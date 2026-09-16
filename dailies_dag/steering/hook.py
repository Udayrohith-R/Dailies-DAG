"""Pre-hook installer: steer Wan attention inputs before ``to_qkv``.

Target
------
``WanTransformer3DModel`` transformer blocks — specifically the **input to the
self-attention mechanism, before ``to_qkv`` projection**. The hook is a
``register_forward_pre_hook`` on each block's self-attention submodule, so the
steered tensor is exactly what the attention's ``to_q``/``to_k``/``to_v``
(or fused ``to_qkv``) projections read.

The hook body runs inside ``torch.no_grad()`` and steers **in place**, adding
zero graph nodes and zero activation copies during inference.

Temporal scoping
----------------
``frame_range=(t0, t1)`` with ``num_spatial_patches=S`` restricts steering to
tokens ``[t0*S, t1*S)`` of the sequence axis — e.g. only the dirty branched
slice. ``None`` steers every token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from dailies_dag.steering.triton_steer_kernel import steer_activations

__all__ = [
    "SteeringHandle",
    "install_steering_pre_hook",
    "remove_steering_pre_hook",
]

_ATTN_ATTR_CANDIDATES: Tuple[str, ...] = (
    "attn1",
    "self_attn",
    "attn",
    "temporal_attn",
    "self_attention",
    "attention",
)

_BLOCK_ATTR_CANDIDATES: Tuple[str, ...] = (
    "blocks",
    "layers",
    "transformer_blocks",
    "attn_blocks",
    "res_blocks",
)


def _iter_blocks(transformer: nn.Module) -> List[Tuple[nn.Module, int]]:
    """Return ``[(block_module, index)]`` for a Wan-style transformer."""
    for attr in _BLOCK_ATTR_CANDIDATES:
        seq = getattr(transformer, attr, None)
        if isinstance(seq, (nn.ModuleList, list, tuple)) and len(seq) > 0:
            return [(b, i) for i, b in enumerate(seq) if isinstance(b, nn.Module)]
    if isinstance(transformer, (nn.ModuleList, list, tuple)):
        return [(b, i) for i, b in enumerate(transformer) if isinstance(b, nn.Module)]
    raise AttributeError(
        f"Could not locate transformer blocks on {type(transformer).__name__}; "
        f"looked for {list(_BLOCK_ATTR_CANDIDATES)}"
    )


def _find_attn(block: nn.Module) -> Optional[str]:
    """Return the attribute name of the self-attention submodule, if found."""
    for name in _ATTN_ATTR_CANDIDATES:
        if isinstance(getattr(block, name, None), nn.Module):
            return name
    for name, child in block.named_children():
        if not isinstance(child, nn.Module):
            continue
        if any(hasattr(child, a) for a in ("to_q", "to_k", "q_proj", "k_proj", "to_qkv")):
            return name
    return None


def _steer_hidden_view(
    hidden: torch.Tensor,
    v: torch.Tensor,
    alpha: float,
    frame_range: Optional[Tuple[int, int]],
    num_spatial_patches: Optional[int],
) -> None:
    """Steer ``hidden`` in place, optionally restricted to a temporal slice."""
    if frame_range is None:
        if hidden.dim() == 3:  # [B, L, D]
            steer_activations(hidden.reshape(-1, hidden.shape[1], hidden.shape[2]), v, alpha)
        elif hidden.dim() == 4:  # [B, T, S, D]
            steer_activations(
                hidden.reshape(-1, hidden.shape[1] * hidden.shape[2], hidden.shape[3]),
                v,
                alpha,
            )
        elif hidden.dim() == 2:  # [L, D] unbatched
            steer_activations(hidden.unsqueeze(0), v, alpha)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported hidden shape {tuple(hidden.shape)}")
        return
    if num_spatial_patches is None:
        raise ValueError("frame_range requires num_spatial_patches")
    t0, t1 = frame_range
    if t0 < 0 or t1 < t0:
        raise ValueError(f"invalid frame_range {(t0, t1)}")
    s = int(num_spatial_patches)
    if hidden.dim() == 3:  # [B, L, D], L = T * S
        tok0, tok1 = t0 * s, t1 * s
        view = hidden[:, tok0:tok1, :].reshape(-1, tok1 - tok0, hidden.shape[2])
        steer_activations(view, v, alpha)
    elif hidden.dim() == 4:  # [B, T, S, D]
        view = hidden[:, t0:t1, :, :].reshape(-1, (t1 - t0) * hidden.shape[2], hidden.shape[3])
        steer_activations(view, v, alpha)
    elif hidden.dim() == 2:  # [L, D]
        tok0, tok1 = t0 * s, t1 * s
        steer_activations(hidden[tok0:tok1, :].unsqueeze(0), v, alpha)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported hidden shape {tuple(hidden.shape)}")


@dataclass
class SteeringHandle:
    """Record of installed pre-hooks; call :meth:`remove` to detach."""

    alpha: float
    _hooks: List[RemovableHandle] = field(default_factory=list)
    _alpha_boxes: List[dict] = field(default_factory=list)
    layers: List[int] = field(default_factory=list)

    def set_alpha(self, alpha: float) -> None:
        """Adjust steering strength live (applies to subsequent forwards)."""
        self.alpha = float(alpha)
        for box in self._alpha_boxes:
            box["alpha"] = float(alpha)

    def remove(self) -> None:
        """Detach every installed pre-hook."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._alpha_boxes.clear()
        self.layers.clear()

    def __len__(self) -> int:
        return len(self._hooks)


def install_steering_pre_hook(
    pipeline: Any,
    steer_vector: torch.Tensor,
    alpha: float,
    *,
    layer_ids: Optional[Sequence[int]] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    num_spatial_patches: Optional[int] = None,
    verbose: bool = True,
) -> SteeringHandle:
    """Steer Wan self-attention inputs in place, before ``to_qkv``.

    Args:
        pipeline: a ``diffusers`` Wan pipeline (uses ``.transformer``), a
            ``WanTransformer3DModel`` itself, or any object exposing a block
            sequence (``blocks``/``layers``/``transformer_blocks``) whose
            blocks expose a self-attention submodule. Test doubles work too.
        steer_vector: 1D ``[Hidden_Dim]`` steering direction ``V``.
        alpha: continuous steering strength scalar.
        layer_ids: subset of block indices to hook; ``None`` hooks all blocks
            that expose self-attention.
        frame_range: optional ``(t_start, t_end)`` temporal window (in frames)
            restricting steering to the dirty slice; requires
            ``num_spatial_patches``.
        num_spatial_patches: ``S`` with ``Sequence_Length = T * S``; required
            with ``frame_range``.
        verbose: print a one-line summary per hooked layer.

    Returns:
        A :class:`SteeringHandle` with live ``.set_alpha()`` and ``.remove()``.
    """
    if steer_vector.dim() != 1:
        raise ValueError(f"steer_vector must be 1D [D], got {tuple(steer_vector.shape)}")
    if frame_range is not None and num_spatial_patches is None:
        raise ValueError("frame_range requires num_spatial_patches")
    transformer: Any = getattr(pipeline, "transformer", None)
    if transformer is None:
        transformer = pipeline
    if not isinstance(transformer, nn.Module):
        raise TypeError(
            "install_steering_pre_hook expected a pipeline/transformer nn.Module, "
            f"got {type(pipeline).__name__}"
        )
    blocks = _iter_blocks(transformer)
    wanted = set(layer_ids) if layer_ids is not None else None
    handle = SteeringHandle(alpha=float(alpha))
    # Keep a per-handle strong reference so the vector can't be freed mid-run.
    v_ref = steer_vector.detach().clone()

    for block, idx in blocks:
        if wanted is not None and idx not in wanted:
            continue
        attr = _find_attn(block)
        if attr is None:
            if verbose:
                print(f"[steering] layer {idx}: no self-attention found, skipped")
            continue
        attn = getattr(block, attr)
        alpha_box: dict = {"alpha": float(alpha)}

        def _pre_hook(
            module: nn.Module,
            args: Tuple[Any, ...],
            kwargs: Any,
            _box: dict = alpha_box,
        ) -> None:
            # Resolve the attention input: first positional arg, else kw.
            hidden: Any = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor):
                return None
            if hidden.shape[-1] != v_ref.shape[0]:
                raise ValueError(
                    f"hidden dim {hidden.shape[-1]} != steer vector dim {v_ref.shape[0]}"
                )
            # In-place steer under no_grad: zero graph nodes, zero copies.
            with torch.no_grad():
                _steer_hidden_view(
                    hidden, v_ref, _box["alpha"], frame_range, num_spatial_patches
                )
            return None

        hook = attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        handle._hooks.append(hook)
        handle._alpha_boxes.append(alpha_box)
        handle.layers.append(idx)
        if verbose:
            print(
                f"[steering] layer {idx}: pre-hook on "
                f"{type(block).__name__}.{attr} (alpha={alpha})"
            )
    if not handle._hooks:
        raise RuntimeError("install_steering_pre_hook attached 0 pre-hooks")

    # SteeringHandle.set_alpha mutates each hook's alpha_box dict, which the
    # bound pre-hook closures read on every forward — no re-registration needed.
    return handle


def remove_steering_pre_hook(handle: SteeringHandle) -> None:
    """Convenience alias for ``handle.remove()``."""
    handle.remove()
