"""Single-clip LoRA overfit for Wan2.1-14B (Dailies-DAG final pipeline).

What this does
-------------
Overfits ONE 4-second cinematic clip into LoRA adapters (rank 128) on
``WanTransformer3DModel`` so branched edits start from a perfectly memorized
baseline. Three stages:

1. **4K proxy ingestion** — read the user's 4K ProRes ``.mp4``, take exactly
   4 s at 24 fps (96 frames), center-crop to 16:9, downscale to strict
   720p (1280x720). No augmentation of any kind (no flips, no crops beyond
   the deterministic 16:9 center window, no color jitter) — the model must
   memorize this exact clip.
2. **Offline VAE + text encode, cached to disk** — the (frozen) Wan VAE and
   text encoder run exactly once; ``{latents, prompt_embeds, meta}`` are
   ``torch.save``-ed. The training loop never touches raw pixels or the
   text encoder again.
3. **Overfit loop** — LoRA (r=128, alpha=128) on ``to_q``/``to_k``/``to_v``/
   ``to_out`` of every spatio-temporal attention block, flow-matching
   velocity MSE, ``AdamW8bit`` (bitsandbytes) + gradient checkpointing.

Heavy dependencies (``diffusers``, ``bitsandbytes``, ``ffmpeg-python``) are
imported lazily inside the functions that need them, so this module imports
cleanly on a CPU-only box and the pure helpers (crop math, frame indexing,
flow-matching targets, LoRA config) are unit-testable without a GPU.

Example:
    python -m dailies_dag.scripts.train_overfit_lora \\
        --clip /data/cine_4k_prores.mp4 --out ./overfit_run --steps 800
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# constants: the single-clip contract                                         #
# --------------------------------------------------------------------------- #

TARGET_FPS: int = 24
TARGET_SECONDS: float = 4.0
TARGET_FRAMES: int = int(TARGET_SECONDS * TARGET_FPS)  # 96, exact
TARGET_W: int = 1280
TARGET_H: int = 720

LORA_RANK: int = 128
LORA_ALPHA: int = 128
LORA_TARGETS: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out")


# --------------------------------------------------------------------------- #
# run config                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class OverfitConfig:
    """All knobs for the single-clip overfit run."""

    clip: Path
    out_dir: Path
    pretrained_model: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    prompt: str = "cinematic dolly shot, golden hour street, photoreal"
    start_sec: float = 0.0
    steps: int = 800
    lr: float = 1e-5
    seed: int = 0
    cache_path: Optional[Path] = None  # defaults to <out_dir>/latents_cached.pt
    save_every: int = 200
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")


# --------------------------------------------------------------------------- #
# Phase 1: 4K proxy ingestion (CPU-only, no model weights)                    #
# --------------------------------------------------------------------------- #


def uniform_frame_indices(num_available: int, num_wanted: int = TARGET_FRAMES) -> List[int]:
    """Evenly spaced frame indices covering ``[0, num_available)``.

    Pure integer math (round-half-up via floor(x + 0.5)) so the 96-frame
    contract is bit-exact and reproducible across machines.
    """
    if num_available < num_wanted:
        raise ValueError(
            f"clip segment has {num_available} frames, need {num_wanted} "
            f"({TARGET_SECONDS}s @ {TARGET_FPS}fps)"
        )
    if num_wanted == 1:
        return [0]
    return [
        min(int(math.floor(i * (num_available - 1) / (num_wanted - 1) + 0.5)), num_available - 1)
        for i in range(num_wanted)
    ]


def center_crop_16x9(frames: torch.Tensor) -> torch.Tensor:
    """Deterministic 16:9 center window (the ONLY spatial crop allowed).

    Args:
        frames: ``[T, C, H, W]``.

    Returns:
        ``[T, C, Hc, Wc]`` with ``Wc / Hc == 16 / 9`` (largest centered box).
    """
    if frames.dim() != 4:
        raise ValueError(f"expected [T, C, H, W], got {tuple(frames.shape)}")
    _, _, h, w = frames.shape
    target_ratio = 16.0 / 9.0
    if w / h > target_ratio:
        # Too wide: trim width.
        wc = int(h * target_ratio)
        x0 = (w - wc) // 2
        return frames[:, :, :, x0 : x0 + wc]
    # Too tall (or exact): trim height.
    hc = int(w / target_ratio)
    y0 = (h - hc) // 2
    return frames[:, :, y0 : y0 + hc, :]


def load_4k_proxy_clip(
    path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float = TARGET_SECONDS,
    fps: int = TARGET_FPS,
    width: int = TARGET_W,
    height: int = TARGET_H,
) -> torch.Tensor:
    """Read a 4K ProRes ``.mp4`` and return a strict 720p proxy tensor.

    Pipeline: decode segment ``[start_sec, start_sec + duration_sec)`` ->
    uniform-resample to exactly ``duration_sec * fps`` frames -> 16:9 center
    crop -> bilinear downscale to ``(width, height)``.

    Primary decoder is ``torchvision.io.read_video``; if that is unavailable
    (or cannot open the file), falls back to ``ffmpeg-python`` rawvideo pipe.
    No augmentation is applied — output is a deterministic function of the
    input file.

    Args:
        path: source 4K ``.mp4``.
        start_sec: segment start in seconds.
        duration_sec: segment length (4.0 -> exactly 96 frames at 24 fps).
        fps: target frame rate.
        width / height: strict output size (1280x720).

    Returns:
        ``[T, C, H, W]`` float32 in [0, 1], ``T == duration_sec * fps``.
    """
    num_wanted = int(round(duration_sec * fps))
    frames = _decode_segment(path, start_sec, duration_sec)  # [N, H, W, C] uint8
    idx = uniform_frame_indices(frames.shape[0], num_wanted)
    sel = frames[idx].permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    cropped = center_crop_16x9(sel)
    # Bilinear resize to the strict contract size. antialias=True keeps thin
    # cinematic detail from shimmering on the 4K -> 720p downscale.
    resized = F.interpolate(
        cropped, size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )
    assert tuple(resized.shape[1:]) == (3, height, width), tuple(resized.shape)
    assert resized.shape[0] == num_wanted, (resized.shape[0], num_wanted)
    return resized.clamp(0.0, 1.0)


def _decode_segment(path: str | Path, start_sec: float, duration_sec: float) -> torch.Tensor:
    """Decode ``[start, start + duration)`` to ``[N, H, W, C]`` uint8."""
    # --- preferred: torchvision (no extra dependency) ---
    try:
        from torchvision.io import read_video  # type: ignore[import]

        try:
            video, _, _ = read_video(
                str(path),
                start_pts=start_sec,
                end_pts=start_sec + duration_sec,
                pts_unit="sec",
                output_format="THWC",
            )
            if video.numel() > 0:
                return video  # uint8 [N, H, W, C]
        except Exception:
            pass  # fall through to ffmpeg
    except ImportError:
        pass
    # --- fallback: ffmpeg-python rawvideo pipe ---
    try:
        import ffmpeg  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError(
            "No video decoder available: torchvision.io.read_video failed and "
            "ffmpeg-python is not installed. Install one of them."
        ) from e
    probe = ffmpeg.probe(str(path))
    src_w, src_h = _probe_size(probe)
    out, _ = (
        ffmpeg.input(str(path), ss=start_sec, t=duration_sec)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            vsync="cfr",  # constant frame rate: exact frame count
        )
        .run(capture_stdout=True, capture_stderr=True)
    )
    import numpy as np

    arr = np.frombuffer(out, dtype=np.uint8)
    frame_bytes = src_h * src_w * 3
    n = arr.size // frame_bytes
    if n == 0:
        raise RuntimeError(f"ffmpeg decoded 0 frames from {path}")
    return torch.from_numpy(arr[: n * frame_bytes].reshape(n, src_h, src_w, 3))


def _probe_size(probe: Dict[str, Any]) -> Tuple[int, int]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and "width" in stream:
            return int(stream["width"]), int(stream["height"])
    raise RuntimeError("ffmpeg probe found no video stream")


# --------------------------------------------------------------------------- #
# Phase 2: offline VAE + text encode, cached to disk (runs ONCE)              #
# --------------------------------------------------------------------------- #


def build_lora_config(
    rank: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
    targets: Tuple[str, ...] = LORA_TARGETS,
) -> Any:
    """PEFT LoRA config for Wan attention projections (no model needed)."""
    from peft import LoraConfig  # type: ignore[import]

    # r == alpha == 128: full-strength adaptation for single-clip memorization.
    # Targeting ONLY the attention projections keeps the adapter small while
    # capturing the clip's geometry/appearance (norms + MLPs stay frozen).
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,  # overfitting: no stochasticity
        bias="none",
        target_modules=list(targets),
        task_type="FEATURE_EXTRACTION",
    )


@torch.no_grad()
def encode_and_cache_latents(
    clip: torch.Tensor,
    prompt: str,
    pretrained_model: str,
    cache_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Run the frozen VAE (+ text encoder) once and cache latents to disk.

    Args:
        clip: ``[T, C, 720, 1280]`` float [0, 1] proxy from
            :func:`load_4k_proxy_clip`.
        prompt: caption describing the clip (conditions the overfit).
        pretrained_model: diffusers checkpoint id or local path.
        cache_path: destination ``.pt`` file.
        device / dtype: encode device/precision (compute only; the cache is
            saved in float32 for exact resumption anywhere).

    Returns:
        The cached dict (also written to ``cache_path``).
    """
    from diffusers import AutoencoderKLWan  # type: ignore[import]
    from transformers import T5EncoderModel, T5Tokenizer  # type: ignore[import]

    dev = torch.device(device)
    if clip.shape[1:] != (3, TARGET_H, TARGET_W):
        raise ValueError(f"expected [T, 3, 720, 1280], got {tuple(clip.shape)}")

    # --- text embeddings (frozen T5, exactly like the Wan pipeline) ---
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model, subfolder="text_encoder", torch_dtype=dtype
    ).to(dev)
    text_encoder.eval()
    tokens = tokenizer(
        prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    prompt_embeds = text_encoder(tokens.input_ids.to(dev))[0].float().cpu()

    # --- video latents (frozen Wan VAE; causal 3D: keep time intact) ---
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model, subfolder="vae", torch_dtype=dtype
    ).to(dev)
    vae.eval()
    video = clip.unsqueeze(0).to(dev, dtype=dtype).permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
    latents = vae.encode(video).latent_dist.sample().float().cpu()
    scale = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
    latents = latents * scale

    payload = {
        "latents": latents,  # [1, C, T', H', W'] float32
        "prompt_embeds": prompt_embeds,  # [1, L, D] float32
        "meta": {
            "frames": clip.shape[0],
            "resolution": [TARGET_W, TARGET_H],
            "fps": TARGET_FPS,
            "scaling_factor": scale,
            "prompt": prompt,
            "pretrained_model": pretrained_model,
        },
    }
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


# --------------------------------------------------------------------------- #
# Phase 3: flow-matching overfit loop                                         #
# --------------------------------------------------------------------------- #


def sample_flow_matching_pair(
    x0: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rectified-flow training pair (pure function, model-free).

    ``x_t = (1 - t) * x0 + t * eps``, target velocity ``u = eps - x0``,
    with ``t ~ U[0, 1]`` per batch element — the Wan2.1/OT flow-matching
    convention.

    Returns:
        ``(x_t, target_velocity, t_float)`` with ``t_float: [B]`` in [0, 1].
    """
    b = x0.shape[0]
    t = torch.rand(b, device=x0.device, generator=generator)
    shape = [b] + [1] * (x0.dim() - 1)
    t_b = t.view(*shape)
    eps = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
    x_t = (1.0 - t_b) * x0 + t_b * eps
    return x_t, (eps - x0), t


def flow_matching_mse(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    """Flow-matching objective: plain MSE between velocities."""
    if pred_velocity.shape != target_velocity.shape:
        raise ValueError(
            f"shape mismatch {tuple(pred_velocity.shape)} vs {tuple(target_velocity.shape)}"
        )
    return F.mse_loss(pred_velocity, target_velocity)


def build_optimizer(params: Any, lr: float) -> torch.optim.Optimizer:
    """AdamW8bit via bitsandbytes (VRAM-saver); AdamW fallback with warning."""
    try:
        import bitsandbytes as bnb  # type: ignore[import]

        return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    except ImportError:
        print("[overfit] bitsandbytes missing: falling back to torch AdamW (more VRAM).")
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)


def train(cfg: OverfitConfig) -> Path:
    """Full overfit run: cached latents -> LoRA flow-matching -> save adapter."""
    from diffusers import WanTransformer3DModel  # type: ignore[import]
    from peft import get_peft_model  # type: ignore[import]

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_path or (cfg.out_dir / "latents_cached.pt")

    # --- load the ONCE-encoded cache (no pixels, no text encoder here) ---
    if not cache_path.exists():
        raise FileNotFoundError(
            f"latent cache {cache_path} missing. Run encode_and_cache_latents "
            "first (it needs the GPU + model weights, but runs only once)."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    latents: torch.Tensor = payload["latents"].to(device, weight_dtype)  # [1,C,T',H',W']
    prompt_embeds: torch.Tensor = payload["prompt_embeds"].to(device, weight_dtype)

    # --- transformer + gradient checkpointing (frozen base, LoRA learns) ---
    transformer = WanTransformer3DModel.from_pretrained(
        cfg.pretrained_model, subfolder="transformer", torch_dtype=weight_dtype
    ).to(device)
    transformer.requires_grad_(False)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()  # activations recomputed, not stored
    elif hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable()
    transformer.train()

    # --- LoRA injection: attention projections only ---
    lora_cfg = build_lora_config()
    transformer = get_peft_model(transformer, lora_cfg)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    print(f"[overfit] trainable LoRA params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer = build_optimizer(trainable, cfg.lr)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    # --- the overfit loop: SAME clip every step, fresh noise each step ---
    # No dataloader, no shuffling, no augmentation: x0 is fixed; only the
    # sampled (t, eps) pair varies, so the adapter memorizes this exact clip.
    x0 = latents
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x_t, target, t_float = sample_flow_matching_pair(x0.float())
        x_t = x_t.to(weight_dtype)
        # Wan scheduler convention: integer timestep in [0, 1000).
        timestep = (t_float * 1000.0).clamp(1.0, 999.0).long().to(device)
        pred = transformer(
            hidden_states=x_t,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        loss = flow_matching_mse(pred.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == cfg.steps:
            print(f"[overfit] step {step}/{cfg.steps}  loss={loss.item():.6f}")
        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt = cfg.out_dir / f"lora_step{step:05d}"
            transformer.save_pretrained(ckpt)
    final = cfg.out_dir / "lora_final"
    transformer.save_pretrained(final)
    print(f"[overfit] done. Adapter: {final}")
    return final


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> OverfitConfig:
    """CLI -> :class:`OverfitConfig` (plus encode-only / train-only switches)."""
    ap = argparse.ArgumentParser(description="Overfit Wan2.1-14B LoRA on one 4K clip")
    ap.add_argument("--clip", type=Path, required=True, help="4K ProRes .mp4 input")
    ap.add_argument("--out", type=Path, required=True, help="output run directory")
    ap.add_argument("--pretrained-model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    ap.add_argument("--prompt", default="cinematic dolly shot, golden hour street, photoreal")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--encode-only", action="store_true", help="ingest + VAE cache, then exit")
    ap.add_argument("--train-only", action="store_true", help="skip ingest, train from cache")
    args = ap.parse_args(argv)
    cfg = OverfitConfig(
        clip=args.clip,
        out_dir=args.out,
        pretrained_model=args.pretrained_model,
        prompt=args.prompt,
        start_sec=args.start_sec,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        save_every=args.save_every,
    )
    # Stash the mode switches on the config object for main().
    cfg.encode_only = args.encode_only  # type: ignore[attr-defined]
    cfg.train_only = args.train_only  # type: ignore[attr-defined]
    return cfg


def main(argv: Optional[List[str]] = None) -> Path:
    """Ingest -> (once) encode+cache -> overfit -> save adapter."""
    cfg = parse_args(argv)
    encode_only: bool = getattr(cfg, "encode_only", False)
    train_only: bool = getattr(cfg, "train_only", False)
    cache_path = cfg.cache_path or (cfg.out_dir / "latents_cached.pt")

    if not train_only:
        print(f"[overfit] ingesting {cfg.clip} (4s @ 24fps -> 720p proxy)...")
        clip = load_4k_proxy_clip(cfg.clip, start_sec=cfg.start_sec)
        print(f"[overfit] proxy: {tuple(clip.shape)}")
        print("[overfit] VAE + text encode (once) -> cache...")
        encode_and_cache_latents(clip, cfg.prompt, cfg.pretrained_model, cache_path)
        print(f"[overfit] cache: {cache_path}")
    if encode_only:
        return cache_path
    cfg.cache_path = cache_path
    return train(cfg)


if __name__ == "__main__":
    main()
