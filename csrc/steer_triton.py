"""Triton latent-steer kernel with torch fallback (Windows/CPU safe)."""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _steer_add_kernel(h_ptr, v_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr = 1024):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        h = tl.load(h_ptr + offs, mask=mask)
        vv = tl.load(v_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, h + alpha * vv, mask=mask)


def steer_add(h: torch.Tensor, vec: torch.Tensor, alpha: float) -> torch.Tensor:
    """h' = h + alpha * vec. Async Triton path on CUDA, exact torch fallback elsewhere."""
    if HAS_TRITON and h.is_cuda:
        out = torch.empty_like(h)
        n = h.numel()
        grid = ((n + 1023) // 1024,)
        _steer_add_kernel[grid](h, vec.broadcast_to(h.shape), out, n, alpha)
        return out
    with torch.no_grad():
        return h + alpha * vec
