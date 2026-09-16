"""Cinematic visual proof: side-by-side geometric-consistency demo.

Inputs: ``original_720p.mp4`` (left) + ``relit_branch_720p.mp4`` (right).
For every frame ``t`` the script extracts DepthAnythingV2 depth maps for both
frames, computes MAE between the normalized maps, and composites an hstack
video with a hacker-style HUD over the right pane::

    DAG Branch: Active | Feature: Luminance (+0.4) | Depth MAE: 0.0123 (PASS)

MAE < 0.05 renders green (PASS); >= 0.05 renders red (WARN). Output is
``netflix_demo_final.mp4`` at 24 fps via ``cv2.VideoWriter``.

Depth comes from :class:`dailies_dag.rl.srpo_guard.DepthRewardModel`, so the
real DepthAnythingV2 weights are used when available (online) and the
deterministic analytic proxy otherwise — same normalized-depth contract.

Example:
    python -m dailies_dag.scripts.render_visual_proof \\
        --original original_720p.mp4 --relit relit_branch_720p.mp4 \\
        --out netflix_demo_final.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:  # cv2 is required for rendering; helpers below stay import-safe without it.
    import cv2

    HAS_CV2: bool = True
except Exception:  # pragma: no cover - headless CI
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

from dailies_dag.rl.srpo_guard import DepthRewardModel

# --------------------------------------------------------------------------- #
# constants                                                                   #
# --------------------------------------------------------------------------- #

DEFAULT_FPS: int = 24
DEFAULT_OUT: str = "netflix_demo_final.mp4"
DEFAULT_FEATURE_LABEL: str = "Luminance (+0.4)"
MAE_PASS_THRESHOLD: float = 0.05

HUD_GREEN_BGR: Tuple[int, int, int] = (0, 255, 0)  # PASS
HUD_RED_BGR: Tuple[int, int, int] = (0, 0, 255)  # WARN
HUD_BANNER_BGR: Tuple[int, int, int] = (0, 0, 0)  # terminal-black backdrop


def _require_cv2() -> None:
    if not HAS_CV2:
        raise RuntimeError(
            "cv2 (opencv-python) is required for rendering but is not installed. "
            "Install it with: pip install opencv-python"
        )


# --------------------------------------------------------------------------- #
# cv2-free helpers (pure logic, unit-testable anywhere)                       #
# --------------------------------------------------------------------------- #


def hud_verdict(mae: float, threshold: float = MAE_PASS_THRESHOLD) -> str:
    """PASS if ``mae < threshold`` else WARN."""
    return "PASS" if float(mae) < float(threshold) else "WARN"


def hud_color(mae: float, threshold: float = MAE_PASS_THRESHOLD) -> Tuple[int, int, int]:
    """HUD text color in BGR: green on PASS, red on WARN."""
    return HUD_GREEN_BGR if hud_verdict(mae, threshold) == "PASS" else HUD_RED_BGR


def format_hud_text(feature_label: str, mae: float, threshold: float = MAE_PASS_THRESHOLD) -> str:
    """One-line HUD string with the measured MAE and verdict."""
    return (
        f"DAG Branch: Active | Feature: {feature_label} "
        f"| Depth MAE: {float(mae):.4f} ({hud_verdict(mae, threshold)})"
    )


def depth_mae(depth_a: torch.Tensor, depth_b: torch.Tensor) -> float:
    """MAE between two normalized depth maps (any broadcastable shape)."""
    if depth_a.shape != depth_b.shape:
        raise ValueError(f"depth shape mismatch {tuple(depth_a.shape)} vs {tuple(depth_b.shape)}")
    with torch.no_grad():
        return float(torch.mean(torch.abs(depth_a.float() - depth_b.float())).item())


def pair_frame_count(n_orig: int, n_relit: int) -> int:
    """Frames to render: the overlap of both clips (must be non-empty)."""
    n = min(n_orig, n_relit)
    if n <= 0:
        raise ValueError("both clips must contain at least one frame")
    return n


@dataclass
class ProofConfig:
    """Render knobs (all paths resolved to absolute on build)."""

    original: Path
    relit: Path
    out: Path = Path(DEFAULT_OUT)
    fps: int = DEFAULT_FPS
    feature_label: str = DEFAULT_FEATURE_LABEL
    threshold: float = MAE_PASS_THRESHOLD
    font_scale: float = 0.7
    thickness: int = 2
    margin: int = 12


# --------------------------------------------------------------------------- #
# video IO (cv2)                                                              #
# --------------------------------------------------------------------------- #


def read_video_frames(path: str | Path) -> Tuple[List[np.ndarray], float]:
    """Read all frames as BGR uint8 + source fps via ``cv2.VideoCapture``."""
    _require_cv2()
    cap = cv2.VideoCapture(str(path))  # type: ignore[union-attr]
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0  # type: ignore[union-attr]
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {path}")
    return frames, fps


def bgr_list_to_rgb_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """BGR uint8 frames -> ``[T, C, H, W]`` RGB float32 in [0, 1] (no grad)."""
    _require_cv2()
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]  # type: ignore[union-attr]
    arr = np.stack(rgb, axis=0)  # [T, H, W, C] uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


# --------------------------------------------------------------------------- #
# depth metric extraction                                                     #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def depth_mae_per_frame(
    orig: torch.Tensor,
    relit: torch.Tensor,
    depth_model: Optional[DepthRewardModel] = None,
) -> List[float]:
    """Per-frame MAE between normalized depth maps of two clips.

    Args:
        orig / relit: ``[T, C, H, W]`` RGB float clips (same T after pairing).
        depth_model: judge (proxy-backed when offline). Built on demand.

    Returns:
        ``[mae_0, ..., mae_{T-1}]`` as Python floats. Grad-free throughout.
    """
    if orig.shape != relit.shape:
        raise ValueError(f"clip shape mismatch {tuple(orig.shape)} vs {tuple(relit.shape)}")
    model = depth_model or DepthRewardModel(use_proxy=True)
    model.eval()
    d_orig = model.estimate(orig)  # [T, 1, H, W], normalized [0, 1]
    d_relit = model.estimate(relit)
    return [depth_mae(d_orig[t], d_relit[t]) for t in range(d_orig.shape[0])]


# --------------------------------------------------------------------------- #
# OpenCV compositing + HUD                                                    #
# --------------------------------------------------------------------------- #


def composite_frame(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    hud_text: str,
    color_bgr: Tuple[int, int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
    margin: int = 12,
) -> np.ndarray:
    """hstack ``[orig | relit]`` + hacker HUD on the top-left of the right pane.

    A solid black banner behind the text keeps it legible over bright footage.
    """
    _require_cv2()
    if left_bgr.shape != right_bgr.shape:
        right_bgr = cv2.resize(  # type: ignore[union-attr]
            right_bgr, (left_bgr.shape[1], left_bgr.shape[0])
        )
    canvas = np.hstack([left_bgr, right_bgr])  # [H, 2W, 3]
    font = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore[union-attr]
    (tw, th), _ = cv2.getTextSize(hud_text, font, font_scale, thickness)  # type: ignore[union-attr]
    x0 = left_bgr.shape[1] + margin  # right pane starts at W
    y0 = margin
    cv2.rectangle(  # type: ignore[union-attr]
        canvas,
        (x0 - 6, y0 - 6),
        (min(x0 + tw + 6, canvas.shape[1] - 1), y0 + th + 10),
        HUD_BANNER_BGR,
        thickness=-1,  # filled
    )
    cv2.putText(  # type: ignore[union-attr]
        canvas, hud_text, (x0, y0 + th), font, font_scale, color_bgr, thickness, cv2.LINE_AA
    )
    return canvas


# --------------------------------------------------------------------------- #
# end-to-end render                                                           #
# --------------------------------------------------------------------------- #


def render(cfg: ProofConfig, depth_model: Optional[DepthRewardModel] = None) -> Dict[str, object]:
    """Render the proof video. Returns per-frame MAEs + summary stats."""
    _require_cv2()
    left_frames, fps_in = read_video_frames(cfg.original)
    right_frames, _ = read_video_frames(cfg.relit)
    n = pair_frame_count(len(left_frames), len(right_frames))
    if len(left_frames) != len(right_frames):
        print(f"[proof] frame mismatch ({len(left_frames)} vs {len(right_frames)}): using {n}")
    left_frames, right_frames = left_frames[:n], right_frames[:n]

    print("[proof] extracting depth maps (DepthAnythingV2 / proxy)...")
    orig_t = bgr_list_to_rgb_tensor(left_frames)
    relit_t = bgr_list_to_rgb_tensor(right_frames)
    maes = depth_mae_per_frame(orig_t, relit_t, depth_model)

    h, w = left_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[union-attr]
    writer = cv2.VideoWriter(str(cfg.out), fourcc, float(cfg.fps), (2 * w, h))  # type: ignore[union-attr]
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {cfg.out}")
    try:
        for t in range(n):
            text = format_hud_text(cfg.feature_label, maes[t], cfg.threshold)
            frame = composite_frame(
                left_frames[t],
                right_frames[t],
                text,
                hud_color(maes[t], cfg.threshold),
                font_scale=cfg.font_scale,
                thickness=cfg.thickness,
                margin=cfg.margin,
            )
            writer.write(frame)
            if t == 0 or (t + 1) % 24 == 0 or t == n - 1:
                print(f"[proof] frame {t + 1}/{n}  MAE={maes[t]:.4f} {hud_verdict(maes[t], cfg.threshold)}")
    finally:
        writer.release()
    mean_mae, max_mae = float(np.mean(maes)), float(np.max(maes))
    n_pass = sum(1 for m in maes if m < cfg.threshold)
    print(f"[proof] wrote {cfg.out} @ {cfg.fps}fps  (src {fps_in:.1f}fps)")
    print(f"[proof] Depth MAE mean={mean_mae:.4f} max={max_mae:.4f}  PASS {n_pass}/{n}")
    return {"per_frame_mae": maes, "mean_mae": mean_mae, "max_mae": max_mae, "frames": n}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> ProofConfig:
    """CLI -> :class:`ProofConfig`."""
    ap = argparse.ArgumentParser(description="Render Dailies-DAG geometric-consistency proof")
    ap.add_argument("--original", type=Path, required=True, help="original_720p.mp4")
    ap.add_argument("--relit", type=Path, required=True, help="relit_branch_720p.mp4")
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="netflix_demo_final.mp4")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--feature", default=DEFAULT_FEATURE_LABEL, help="HUD feature label")
    ap.add_argument("--threshold", type=float, default=MAE_PASS_THRESHOLD)
    args = ap.parse_args(argv)
    if args.fps <= 0:
        raise ValueError("fps must be > 0")
    return ProofConfig(
        original=args.original,
        relit=args.relit,
        out=args.out,
        fps=args.fps,
        feature_label=args.feature,
        threshold=args.threshold,
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    """CLI entry: parse args -> render -> stats."""
    return render(parse_args(argv))


if __name__ == "__main__":
    main()
