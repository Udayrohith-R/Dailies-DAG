"""Stage-Relative GRPO guardrails for Wan2.1 branched edits (SRPO-Guard).

Goal
----
Train an edit policy (lighting / feature steering) while strictly preventing
actor face warping, background melting, and optical-flow distortion. Two
deterministic geometric judges score every rollout; a stage-relative GRPO step
turns those scores into policy gradients.

Pipeline per edit prompt
------------------------
1. Sample ``G`` rollout branches (e.g. different steering strengths).
2. Judges (no grad graphs, ever):
   * :class:`DepthRewardModel` — monocular depth consistency (DepthAnythingV2,
     deterministic analytic proxy offline).
   * :class:`FlowRewardModel` — motion consistency (RAFT via torchvision,
     deterministic analytic proxy offline).
   * ``L_depth = MAE(depth_orig, depth_edit)``,
     ``L_flow  = MAE(flow_orig,  flow_edit)``.
3. Multi-objective reward (vectorized over the group)::

       R = w_edit * R_feature_shift
           - lambda_depth * L_depth * stage_w(t)
           - lambda_flow  * L_flow  * stage_w(t)

4. Vectorized Pareto + hard-threshold filtering (NumPy broadcasting with
   ``np.all`` / ``np.any``): discard any trajectory with ``L_depth > tau``
   regardless of feature score, then drop Pareto-dominated survivors.
5. Group-relative advantages ``A_i = (R_i - mean(R)) / (std(R) + eps)`` over
   survivors and a clipped GRPO surrogate update.

Flow-matching stage relativity
------------------------------
Wan2.1 is a flow-matching transformer: early denoising steps fix global
structure, late steps refine texture. Structural penalties are therefore
scaled by ``stage_w(t)`` — heaviest at ``t = 0`` (early structure), lightest
at ``t = T - 1`` (final refinement). Linear and cosine decays available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GuardConfig",
    "StageSchedule",
    "DepthRewardModel",
    "FlowRewardModel",
    "depth_loss",
    "flow_loss",
    "luminance_shift_score",
    "stage_weight",
    "compute_rewards",
    "pareto_filter",
    "grpo_step",
    "score_rollouts",
]

# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class StageSchedule:
    """Flow-matching stage weighting for structural penalties.

    ``t`` is the denoising-step index in ``[0, num_steps - 1]`` (``0`` = early
    structure formation, ``num_steps - 1`` = final refinement).
    ``stage_w(0) = 1 + boost``, ``stage_w(num_steps - 1) = 1``.
    """

    num_steps: int = 50
    boost: float = 1.0
    mode: Literal["linear", "cosine"] = "linear"

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.boost < 0:
            raise ValueError("boost must be >= 0")


@dataclass
class GuardConfig:
    """Multi-objective reward weights and safety thresholds."""

    w_edit: float = 1.0
    lambda_depth: float = 2.0
    lambda_flow: float = 2.0
    tau_depth: float = 0.08  # hard discard threshold on L_depth (MAE, [0,1] depth)
    epsilon: float = 1e-6  # advantage-normalization epsilon
    clip_eps: float = 0.2  # GRPO ratio-clipping epsilon
    kl_beta: float = 0.01  # KL penalty weight (0 disables)
    stage: StageSchedule = field(default_factory=StageSchedule)


# --------------------------------------------------------------------------- #
# deterministic judge base                                                    #
# --------------------------------------------------------------------------- #


class NoGradRewardModel(nn.Module):
    """Base class: eval-only judges that never build gradient graphs."""

    def __init__(self) -> None:
        super().__init__()
        self._backbone: Optional[nn.Module] = None
        self.using_proxy: bool = True

    def _freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        assert self._backbone is not None
        return self._backbone(x)

    def extra_repr(self) -> str:
        kind = "proxy" if self.using_proxy else "pretrained"
        return f"backend={kind}"


def _as_float_frames(video: torch.Tensor) -> torch.Tensor:
    """Normalize ``[T, C, H, W]`` video to float32 in [0, 1] (no grad)."""
    if video.dim() != 4:
        raise ValueError(f"expected [T, C, H, W], got {tuple(video.shape)}")
    if video.dtype == torch.uint8:
        return video.float() / 255.0
    v = video.detach().float()
    if v.max().item() > 1.5:  # assume [0, 255] floats
        v = v / 255.0
    return v.clamp(0.0, 1.0)


def _rgb_to_luma(video: torch.Tensor) -> torch.Tensor:
    """BT.601 luma, ``[T, C, H, W]`` -> ``[T, 1, H, W]``."""
    v = _as_float_frames(video)
    if v.shape[1] == 1:
        return v
    if v.shape[1] < 3:
        raise ValueError(f"expected >=3 channels for luma, got {v.shape[1]}")
    r, g, b = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------------------- #
# depth judge (DepthAnythingV2, proxy fallback)                               #
# --------------------------------------------------------------------------- #


class DepthRewardModel(NoGradRewardModel):
    """Monocular-depth consistency judge.

    Primary backend: DepthAnythingV2 via ``transformers``
    (``AutoImageProcessor`` + ``AutoModelForDepthEstimation``). Lazy-loaded so
    importing this module never touches the network. When weights are
    unavailable (offline CI) or ``use_proxy=True``, a deterministic analytic
    proxy is used: normalized inverse-luma smoothed by a 3x3 box filter —
    a stable, reproducible stand-in with the same ``[T, 1, H, W]`` contract.

    All inference runs under ``torch.inference_mode`` with frozen params.
    """

    CHECKPOINT = "depth-anything/Depth-Anything-V2-Small"

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        device: Optional[torch.device | str] = None,
        use_proxy: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor: Optional[Any] = None
        if not use_proxy:
            self._try_load()
        self._freeze()

    def _try_load(self) -> None:
        try:
            from transformers import (  # type: ignore[import]
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )

            self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self._backbone = AutoModelForDepthEstimation.from_pretrained(
                self.checkpoint
            ).to(self.device)
            self.using_proxy = False
        except Exception:
            self._processor = None
            self._backbone = None
            self.using_proxy = True

    @torch.inference_mode()
    def estimate(self, video: torch.Tensor) -> torch.Tensor:
        """Return per-frame depth ``[T, 1, H, W]`` in [0, 1] (detached)."""
        v = _as_float_frames(video)
        if self._backbone is not None and self._processor is not None:
            depths: List[torch.Tensor] = []
            for t in range(v.shape[0]):
                inp = self._processor(
                    images=(v[t].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy(),
                    return_tensors="pt",
                )
                pixel_values = inp["pixel_values"].to(self.device)
                pred = self._backbone(pixel_values).predicted_depth  # [1, h, w]
                d = F.interpolate(
                    pred.unsqueeze(1), size=v.shape[-2:], mode="bilinear", align_corners=False
                )
                d = d - d.amin(dim=(2, 3), keepdim=True)
                d = d / (d.amax(dim=(2, 3), keepdim=True) + 1e-6)
                depths.append(d.to("cpu"))
            return torch.cat(depths, dim=0).detach()
        return self._proxy_depth(v).detach()

    @staticmethod
    def _proxy_depth(v: torch.Tensor) -> torch.Tensor:
        """Deterministic analytic depth proxy (offline-safe)."""
        luma = _rgb_to_luma(v)  # [T,1,H,W]
        smooth = F.avg_pool2d(luma, kernel_size=3, stride=1, padding=1)
        d = 1.0 - smooth
        d = d - d.amin(dim=(2, 3), keepdim=True)
        return d / (d.amax(dim=(2, 3), keepdim=True) + 1e-6)


def depth_loss(depth_orig: torch.Tensor, depth_edit: torch.Tensor) -> torch.Tensor:
    """``L_depth = MAE(depth_orig, depth_edit)`` — scalar tensor, no grad."""
    with torch.no_grad():
        return F.l1_loss(depth_edit.detach().float(), depth_orig.detach().float())


# --------------------------------------------------------------------------- #
# flow judge (RAFT via torchvision, proxy fallback)                           #
# --------------------------------------------------------------------------- #


class FlowRewardModel(NoGradRewardModel):
    """Optical-flow consistency judge.

    Primary backend: RAFT (``torchvision.models.optical_flow.raft_small``)
    over consecutive frame pairs. Lazy-loaded; deterministic analytic proxy
    (luma-difference motion energy broadcast to 2 channels) when offline.
    Output contract: ``[T - 1, 2, H, W]`` flow fields (detached).
    """

    def __init__(
        self,
        model: Literal["raft_small", "raft_large"] = "raft_small",
        device: Optional[torch.device | str] = None,
        use_proxy: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if not use_proxy:
            self._try_load()
        self._freeze()

    def _try_load(self) -> None:
        try:
            from torchvision.models.optical_flow import (  # type: ignore[import]
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )

            if self.model_name == "raft_large":
                weights = Raft_Large_Weights.DEFAULT
                self._backbone = raft_large(weights=weights, progress=False).to(self.device)
            else:
                weights = Raft_Small_Weights.DEFAULT
                self._backbone = raft_small(weights=weights, progress=False).to(self.device)
            self.using_proxy = False
        except Exception:
            self._backbone = None
            self.using_proxy = True

    @torch.inference_mode()
    def estimate(self, video: torch.Tensor) -> torch.Tensor:
        """Return inter-frame flow ``[T - 1, 2, H, W]`` (detached)."""
        v = _as_float_frames(video)
        if v.shape[0] < 2:
            raise ValueError("flow needs at least 2 frames")
        if self._backbone is not None:
            flows: List[torch.Tensor] = []
            for t in range(v.shape[0] - 1):
                img1 = (v[t : t + 1].clamp(0, 1) * 255).to(self.device)
                img2 = (v[t + 1 : t + 2].clamp(0, 1) * 255).to(self.device)
                out = self._backbone(img1, img2)
                f = out[-1] if isinstance(out, (list, tuple)) else out
                flows.append(F.interpolate(f, size=v.shape[-2:], mode="bilinear",
                                           align_corners=False).to("cpu"))
            return torch.cat(flows, dim=0).detach()
        return self._proxy_flow(v).detach()

    @staticmethod
    def _proxy_flow(v: torch.Tensor) -> torch.Tensor:
        """Deterministic motion proxy: signed luma deltas as (dx, mag)."""
        luma = _rgb_to_luma(v).squeeze(1)  # [T,H,W]
        diff = luma[1:] - luma[:-1]  # [T-1,H,W]
        mag = diff.abs()
        return torch.stack([diff, mag], dim=1)  # [T-1,2,H,W]


def flow_loss(flow_orig: torch.Tensor, flow_edit: torch.Tensor) -> torch.Tensor:
    """``L_flow = MAE(flow_orig, flow_edit)`` — scalar tensor, no grad."""
    if flow_orig.shape != flow_edit.shape:
        raise ValueError(f"flow shape mismatch {tuple(flow_orig.shape)} vs {tuple(flow_edit.shape)}")
    with torch.no_grad():
        return F.l1_loss(flow_edit.detach().float(), flow_orig.detach().float())


def luminance_shift_score(orig: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
    """Default ``R_feature_shift``: mean absolute luma change in [0, 1].

    Lighting/feature steering should move the image; a zero score means the
    rollout changed nothing. Task-specific callers may supply their own
    ``R_feature_shift`` (e.g. CLIP alignment) instead.
    """
    with torch.no_grad():
        lo = _rgb_to_luma(orig)
        le = _rgb_to_luma(edited)
        if lo.shape != le.shape:
            raise ValueError("orig/edited shape mismatch")
        return (le - lo).abs().mean().detach()


# --------------------------------------------------------------------------- #
# tensor coercion helpers                                                   #
# --------------------------------------------------------------------------- #


def _to_float_detached(x: torch.Tensor | Sequence[float] | np.ndarray) -> torch.Tensor:
    """Flatten to detached float32 CPU tensor (scores, masks, old logprobs)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32).reshape(-1).cpu()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def _to_float_keep_grad(x: torch.Tensor | Sequence[float] | np.ndarray) -> torch.Tensor:
    """Flatten to float32, preserving autograd (current-policy logprobs)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32).reshape(-1)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# stage-relative weighting                                                    #
# --------------------------------------------------------------------------- #


def stage_weight(
    timestep_idx: torch.Tensor | int | Sequence[int] | np.ndarray,
    schedule: StageSchedule | None = None,
) -> torch.Tensor:
    """Penalty multiplier in ``[1, 1 + boost]`` — heaviest at early steps.

    ``t = 0`` (early structure formation) -> ``1 + boost``;
    ``t = num_steps - 1`` (final refinement) -> ``1``.
    """
    sched = schedule or StageSchedule()
    t = torch.as_tensor(
        np.asarray(timestep_idx) if isinstance(timestep_idx, Sequence) else timestep_idx,
        dtype=torch.float32,
    )
    n = sched.num_steps
    if n == 1:
        return torch.ones_like(t) * (1.0 + sched.boost)
    frac = (t / float(n - 1)).clamp(0.0, 1.0)
    if sched.mode == "cosine":
        decay = 0.5 * (1.0 + torch.cos(np.pi * frac))  # 1 -> 0
    else:
        decay = 1.0 - frac
    return 1.0 + sched.boost * decay


def compute_rewards(
    feature_shift: torch.Tensor | Sequence[float] | np.ndarray,
    l_depth: torch.Tensor | Sequence[float] | np.ndarray,
    l_flow: torch.Tensor | Sequence[float] | np.ndarray,
    cfg: GuardConfig,
    timestep_idx: Optional[torch.Tensor | Sequence[int] | np.ndarray] = None,
) -> torch.Tensor:
    """``R = w_edit*R_feat - lam_d*L_d*stage_w(t) - lam_f*L_f*stage_w(t)``.

    All inputs broadcast to ``[G]`` (one value per rollout branch). Structural
    penalties are dynamically normalized by the flow-matching timestep: early
    denoising steps penalize deviations more than late refinement steps.
    """
    feat = _to_float_detached(feature_shift)
    ld = _to_float_detached(l_depth)
    lf = _to_float_detached(l_flow)
    if not (feat.shape == ld.shape == lf.shape):
        raise ValueError("feature_shift / l_depth / l_flow must share shape [G]")
    if timestep_idx is None:
        w = torch.ones_like(feat)
    else:
        w = stage_weight(timestep_idx, cfg.stage).reshape(-1)
        if w.shape != feat.shape:
            raise ValueError("timestep_idx must broadcast to [G]")
    with torch.no_grad():
        return (cfg.w_edit * feat - cfg.lambda_depth * ld * w - cfg.lambda_flow * lf * w).detach()


# --------------------------------------------------------------------------- #
# vectorized Pareto + hard-threshold filtering                                #
# --------------------------------------------------------------------------- #


def pareto_filter(
    feature_shift: np.ndarray | Sequence[float],
    l_depth: np.ndarray | Sequence[float],
    l_flow: np.ndarray | Sequence[float],
    tau_depth: float,
) -> np.ndarray:
    """Return a boolean keep-mask over ``G`` candidates.

    1. **Hard safety gate**: ``L_depth > tau_depth`` is discarded instantly,
       regardless of feature-shift score (face warp / background melt veto).
    2. **Pareto front** (maximize feature shift, minimize both drifts): ``i``
       is dominated iff some ``j`` is at least as good on all three objectives
       and strictly better on one — computed with fully vectorized ``np.all``
       / ``np.any`` broadcasting (``[G, G]`` comparisons, no Python loops).
    """
    f = np.asarray(feature_shift, dtype=np.float64).reshape(-1)
    d = np.asarray(l_depth, dtype=np.float64).reshape(-1)
    fl = np.asarray(l_flow, dtype=np.float64).reshape(-1)
    if not (f.shape == d.shape == fl.shape):
        raise ValueError("inputs must share shape [G]")
    g = f.shape[0]
    if g == 0:
        return np.zeros(0, dtype=bool)
    keep = d <= float(tau_depth)  # hard gate first
    # Pairwise: row j (challenger) vs col i (incumbent).
    # j dominates i iff j is at least as good on all axes (maximize f,
    # minimize d / fl) and strictly better on at least one.
    better_or_equal = (
        (f[:, None] >= f[None, :])
        & (d[:, None] <= d[None, :])
        & (fl[:, None] <= fl[None, :])
    )
    strictly = (
        (f[:, None] > f[None, :])
        | (d[:, None] < d[None, :])
        | (fl[:, None] < fl[None, :])
    )
    # Note: the diagonal is naturally False (a candidate is never strictly
    # better than itself), so no self-domination is possible.
    dominated = np.any(better_or_equal & strictly, axis=0)
    keep = keep & ~dominated
    return keep


# --------------------------------------------------------------------------- #
# rollout scoring                                                             #
# --------------------------------------------------------------------------- #


def score_rollouts(
    orig: torch.Tensor,
    edited: Sequence[torch.Tensor],
    feature_shift: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None,
    depth_model: Optional[DepthRewardModel] = None,
    flow_model: Optional[FlowRewardModel] = None,
) -> Dict[str, torch.Tensor]:
    """Score ``G`` edited rollouts against ``orig`` with the geometric judges.

    Args:
        orig: ``[T, C, H, W]`` reference frames.
        edited: ``G`` edited videos, each ``[T, C, H, W]``.
        feature_shift: optional precomputed ``[G]`` task scores; defaults to
            :func:`luminance_shift_score` per rollout.
        depth_model / flow_model: judges (proxy-backed when offline). Fresh
            defaults are constructed on demand.

    Returns:
        Dict with ``feature_shift``, ``l_depth``, ``l_flow`` tensors ``[G]``.
        Judge inference is grad-free throughout.
    """
    depth_model = depth_model or DepthRewardModel(use_proxy=True)
    flow_model = flow_model or FlowRewardModel(use_proxy=True)
    depth_model.eval()
    flow_model.eval()
    with torch.inference_mode():
        d_orig = depth_model.estimate(orig)
        f_orig = flow_model.estimate(orig)
        feats: List[float] = []
        lds: List[float] = []
        lfs: List[float] = []
        for ed in edited:
            if feature_shift is None:
                feats.append(float(luminance_shift_score(orig, ed).item()))
            lds.append(float(depth_loss(d_orig, depth_model.estimate(ed)).item()))
            lfs.append(float(flow_loss(f_orig, flow_model.estimate(ed)).item()))
        out: Dict[str, torch.Tensor] = {
            "l_depth": torch.tensor(lds, dtype=torch.float32),
            "l_flow": torch.tensor(lfs, dtype=torch.float32),
        }
        if feature_shift is None:
            out["feature_shift"] = torch.tensor(feats, dtype=torch.float32)
        else:
            out["feature_shift"] = torch.as_tensor(
                np.asarray(feature_shift), dtype=torch.float32
            ).reshape(-1)
            if out["feature_shift"].shape != out["l_depth"].shape:
                raise ValueError("feature_shift must have one entry per rollout")
        return out


# --------------------------------------------------------------------------- #
# GRPO step                                                                   #
# --------------------------------------------------------------------------- #


def grpo_step(
    logprobs: torch.Tensor | Sequence[float] | np.ndarray,
    old_logprobs: torch.Tensor | Sequence[float] | np.ndarray,
    feature_shift: torch.Tensor | Sequence[float] | np.ndarray,
    l_depth: torch.Tensor | Sequence[float] | np.ndarray,
    l_flow: torch.Tensor | Sequence[float] | np.ndarray,
    cfg: GuardConfig | None = None,
    timestep_idx: Optional[torch.Tensor | Sequence[int] | np.ndarray] = None,
    policy: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """One stage-relative GRPO step over ``G`` rollout branches.

    Args:
        logprobs / old_logprobs: ``[G]`` token-sequence log-probs under the
            current / sampling policies (detached for the ratio's denominator).
        feature_shift / l_depth / l_flow: ``[G]`` per-branch scores/losses.
        cfg: reward weights, ``tau_depth`` gate, clipping, KL weight.
        timestep_idx: ``[G]`` (or scalar) flow-matching step indices used for
            :func:`stage_weight` normalization of the structural penalties.
        policy / optimizer: when both are given, backpropagate the clipped
            surrogate through ``logprobs`` (which must then carry grad) and
            take an optimizer step. When omitted, the step is dry-run scoring
            (loss is still computed differentiably w.r.t. ``logprobs``).

    Returns:
        Dict with ``rewards[G]``, ``advantages[G]`` (0 for discarded),
        ``keep[G]`` bool mask, ``loss``, ``kl``, ``num_kept``,
        ``all_discarded``. Advantages are group-relative over survivors::

            A_i = (R_i - mean(R_kept)) / (std(R_kept) + eps)
    """
    config = cfg or GuardConfig()
    lp = _to_float_keep_grad(logprobs)
    olp = _to_float_detached(old_logprobs)
    if lp.shape != olp.shape:
        raise ValueError("logprobs / old_logprobs must share shape [G]")
    g = lp.shape[0]
    rewards = compute_rewards(feature_shift, l_depth, l_flow, config, timestep_idx)
    if rewards.shape[0] != g:
        raise ValueError("score inputs must share shape with logprobs [G]")

    feat_np = _to_float_detached(feature_shift).numpy()
    ld_np = _to_float_detached(l_depth).numpy()
    lf_np = _to_float_detached(l_flow).numpy()
    keep = pareto_filter(feat_np, ld_np, lf_np, config.tau_depth)
    num_kept = int(keep.sum())
    all_discarded = num_kept == 0

    advantages = torch.zeros_like(rewards)
    if not all_discarded:
        rk = rewards[torch.from_numpy(keep)]
        mu = rk.mean()
        sigma = rk.std(unbiased=False) if num_kept > 1 else torch.zeros(())
        advantages[torch.from_numpy(keep)] = (rk - mu) / (sigma + config.epsilon)

    # Clipped surrogate over survivors (discarded branches contribute nothing).
    ratio = torch.exp(lp - olp.detach())
    clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
    per_sample = torch.min(ratio * advantages.detach(), clipped * advantages.detach())
    keep_t = torch.from_numpy(keep)
    if num_kept > 0:
        pg_loss = -per_sample[keep_t].mean()
    else:
        pg_loss = torch.zeros((), dtype=torch.float32)
        if lp.requires_grad:
            pg_loss = pg_loss + 0.0 * lp.sum()  # keep graph valid, zero grad
    with torch.no_grad():
        kl = (olp.detach() - lp.detach()).mean().detach() if g > 0 else torch.zeros(())
    loss = pg_loss + config.kl_beta * (olp.detach() - lp).mean() * float(num_kept > 0)

    if policy is not None or optimizer is not None:
        if policy is None or optimizer is None:
            raise ValueError("policy and optimizer must be given together")
        if all_discarded:
            optimizer.zero_grad(set_to_none=True)  # nothing to learn; stay clean
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

    return {
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        "keep": keep,
        "loss": loss.detach(),
        "kl": kl,
        "num_kept": num_kept,
        "all_discarded": all_discarded,
    }
