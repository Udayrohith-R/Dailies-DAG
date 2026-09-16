"""Fused Triton steering kernel + PyTorch fallback for Wan2.1 activations.

Operation
---------
In-place broadcast addition over batched hidden states::

    X_new = X + (alpha * V)

where ``X: [Batch, Sequence_Length, Hidden_Dim]`` (``Sequence_Length = T * S``,
frames times spatial patches), ``V: [Hidden_Dim]`` is a 1D steering vector
(e.g. a "luminance" latent direction), and ``alpha`` is a continuous float
scalar (steering strength, may be negative).

Hopper/Ada bandwidth notes
--------------------------
* One program per ``(row, hidden-block)`` tile (2-D grid); each program streams
  a single ``BLOCK``-wide chunk (1024/2048) of contiguous ``Hidden_Dim`` —
  fully coalesced 128-bit-friendly accesses, no transposes, no temporaries.
* ``BLOCK`` is chosen from the hidden dim (1024 for ``D <= 1024``, else 2048),
  launched with ``num_warps=8`` to saturate Hopper/Ada LD/ST pipelines.
* In-place: zero extra activation memory (no output allocation).
* ``alpha`` is passed as a runtime fp32 scalar, not baked into the program,
  so sweeping strength never retriggers recompilation.
"""

from __future__ import annotations

from typing import Optional

import torch

try:  # Triton is optional: CPU/Windows CI runs the exact torch fallback.
    import triton
    import triton.language as tl

    HAS_TRITON: bool = True
except Exception:  # pragma: no cover - triton unavailable
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def is_triton_available() -> bool:
    """Return True when the Triton JIT imported successfully."""
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit  # type: ignore[misc]
    def _fused_steer_kernel(
        x_ptr,
        v_ptr,
        hidden_dim,
        alpha,
        x_stride_row,
        BLOCK: tl.constexpr,  # type: ignore[valid-type]
    ):
        """In-place ``X[row, :] += alpha * V`` for one hidden-dim tile.

        Grid: ``(n_rows, ceil(hidden_dim / BLOCK))``. ``X`` must be
        row-contiguous (``stride_dim == 1``); the wrapper guarantees this.
        """
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hidden_dim
        x = tl.load(x_ptr + row * x_stride_row + offs, mask=mask)
        vv = tl.load(v_ptr + offs, mask=mask)
        tl.store(x_ptr + row * x_stride_row + offs, x + alpha * vv.to(x.dtype), mask=mask)


def _pick_block(hidden_dim: int) -> int:
    """Hopper/Ada-tuned tile width: 1024 for narrow, else 2048."""
    return 1024 if hidden_dim <= 1024 else 2048


def steer_activations(
    X: torch.Tensor,
    V: torch.Tensor,
    alpha: float,
    *,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Steer hidden states in place: ``X += alpha * V`` (broadcast over rows).

    Args:
        X: ``[Batch, Sequence_Length, Hidden_Dim]`` activations. Modified
            **in place** and returned for chaining.
        V: ``[Hidden_Dim]`` steering vector. Cast to ``X``'s device/dtype
            (non-destructively) before use.
        alpha: continuous steering strength scalar.
        block_size: Triton tile width override (1024 or 2048). ``None``
            selects from the hidden dim automatically.

    Returns:
        ``X`` itself (same storage; ``out.data_ptr() == X.data_ptr()``).

    Dispatch:
        Triton fused kernel when ``X`` is a contiguous CUDA tensor and Triton
        imported; otherwise a strict, numerically identical
        ``torch.no_grad`` in-place fallback (used for CPU testing).
    """
    if X.dim() != 3:
        raise ValueError(f"X must be [B, L, D], got shape {tuple(X.shape)}")
    if V.dim() != 1:
        raise ValueError(f"V must be 1D [D], got shape {tuple(V.shape)}")
    if V.shape[0] != X.shape[2]:
        raise ValueError(
            f"V dim {V.shape[0]} != X hidden dim {X.shape[2]}"
        )
    if block_size is not None and block_size not in (1024, 2048):
        raise ValueError("block_size must be 1024 or 2048")
    alpha_f = float(alpha)
    v = V.to(device=X.device, dtype=X.dtype)

    use_triton = (
        HAS_TRITON and X.is_cuda and X.is_contiguous() and v.is_contiguous()
    )
    if use_triton:
        assert triton is not None
        n_rows = X.shape[0] * X.shape[1]
        hidden_dim = X.shape[2]
        block = block_size or _pick_block(hidden_dim)
        grid = (n_rows, (hidden_dim + block - 1) // block)
        x_flat = X.reshape(n_rows, hidden_dim)
        _fused_steer_kernel[grid](  # type: ignore[index]
            x_flat,
            v,
            hidden_dim,
            alpha_f,
            x_flat.stride(0),
            block,
            num_warps=8,
        )
        return X
    # Strict native fallback: identical math, in-place, no graph overhead.
    with torch.no_grad():
        X.add_(v * alpha_f)
    return X
