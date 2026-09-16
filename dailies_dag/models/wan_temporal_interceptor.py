"""Wan2.1-14B causal temporal interceptor: paged attention over CoW block pool.

Wiring
------
``WanTransformer3DModel`` (HuggingFace ``diffusers``) is a flow-matching DiT:
each block modulates its norms with a timestep embedding (``temb``) and runs
self-attention over patched video latents. This module monkey-patches those
self-attention submodules in place with :class:`CausalPagedAttentionWrapper`,
which serves prefix frames from :class:`DistributedBlockPagedKVCache` instead
of recomputing them.

Differential prefill vs. cached inference
-----------------------------------------
For a branched edit starting at ``branch_frame_t`` with inputs covering
``[start_frame, start_frame + T_in)``:

* frames ``t < branch_frame_t`` (clean prefix): **no** Q projection and **no**
  attention math. Their K/V are read directly from the block pool and used
  only as attention *context*.
* frames ``t >= branch_frame_t`` (dirty slice): full Q/K/V projection, K/V
  committed via ``cache.write_block()`` (CoW if the block is shared), then
  multi-head causal attention of dirty queries over
  ``K_full = [K_cached_prefix ; K_dirty]``.

The wrapper returns dirty-frame outputs only (``[Td, ...]``); callers reuse the
previously computed prefix outputs. Pass ``return_full="passthrough"`` to get
a full-length tensor where prefix slots carry the input hidden states through
(useful for shape-stable pipelines).

Flow matching / precision
-------------------------
* ``temb`` (and ``encoder_hidden_states`` / masks) are accepted explicitly and
  threaded through **untouched**: the fallback path forwards them verbatim to
  the wrapped module; the paged path never consumes them for attention math
  (attention is temb-independent; adaLN modulation lives in the parent block
  and applies identically to dirty frames outside this wrapper).
* Stability: projections run in the input dtype, but QK matmul / softmax run
  in ``float32`` with max-subtraction, then cast back. ``bfloat16`` and
  ``float8`` (e4m3fn/e5m2) inputs are upcast before any accumulation; logits
  are clamped to finite range. No in-place ops touch low-precision views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional: only for typing; runtime is duck-typed so diffusers is not required.
    from diffusers.models.transformers.wan_transformer_3d import (  # type: ignore[import]
        WanTransformer3DModel,
    )
except Exception:  # pragma: no cover - diffusers not installed in CI
    WanTransformer3DModel = Any  # type: ignore[assignment,misc]

from dailies_dag.cache.distributed_block_paged import DistributedBlockPagedKVCache

__all__ = [
    "CausalPagedAttentionWrapper",
    "InterceptorHandle",
    "install_interceptor",
    "remove_interceptor",
]

_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_LOW_PRECISION_DTYPES = {torch.bfloat16, torch.float16, *_FP8_DTYPES}


# --------------------------------------------------------------------------- #
# small utilities                                                             #
# --------------------------------------------------------------------------- #

def _upcast_for_math(x: torch.Tensor) -> torch.Tensor:
    """Upcast low-precision tensors to float32 for numerically safe math."""
    if x.dtype in _LOW_PRECISION_DTYPES:
        return x.float()
    return x


def _linear(mod: nn.Module, attr_candidates: Sequence[str], x: torch.Tensor) -> torch.Tensor:
    """Apply the first present linear layer among ``attr_candidates``."""
    for name in attr_candidates:
        proj = getattr(mod, name, None)
        if isinstance(proj, nn.Linear):
            return proj(x)
    raise AttributeError(
        f"{type(mod).__name__} has none of {list(attr_candidates)}; "
        "expected to_q/to_k/to_v (or q_proj/k_proj/v_proj, or fused to_qkv)"
    )


def _resolve_head_geometry(
    wrapped: nn.Module, cache: DistributedBlockPagedKVCache
) -> Tuple[int, int]:
    """Return ``(num_heads_global, head_dim)`` preferring the wrapped module."""
    heads = getattr(wrapped, "num_heads", None) or getattr(wrapped, "heads", None)
    dh = getattr(wrapped, "head_dim", None) or getattr(wrapped, "dim_head", None)
    num_heads = int(heads) if heads is not None else int(cache.num_heads)
    head_dim = int(dh) if dh is not None else int(cache.head_dim)
    # Fall back to cache geometry when the wrapped module disagrees only by
    # sharding (local vs global heads): normalize to global.
    if num_heads == cache.local_num_heads and cache.world_size > 1:
        num_heads = cache.num_heads
    return num_heads, head_dim


# --------------------------------------------------------------------------- #
# wrapper                                                                     #
# --------------------------------------------------------------------------- #

class CausalPagedAttentionWrapper(nn.Module):
    """Paged causal self-attention interceptor around a Wan attention module.

    Args:
        wrapped: original attention submodule (e.g. ``WanSelfAttention``).
            Must expose query/key/value projections as ``to_q``/``to_k``/
            ``to_v`` (or ``q_proj``/``k_proj``/``v_proj``, or fused
            ``to_qkv``) plus an output projection ``to_out`` (or
            ``out_proj``/``o_proj``). Kept as ``self.wrapped`` and used
            verbatim on the fallback path.
        cache: rank-sharded :class:`DistributedBlockPagedKVCache` holding K/V.
            ``cache.num_spatial_patches`` defines the frame/space split.
        layer_id: transformer-block index addressing the cache's per-layer
            pool (``k_pool[layer_id]``).
        dropout: dropout applied to attention weights (0.0 at inference).
    """

    def __init__(
        self,
        wrapped: nn.Module,
        cache: DistributedBlockPagedKVCache,
        layer_id: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0 <= layer_id < cache.num_layers):
            raise ValueError(f"layer_id {layer_id} out of range [0, {cache.num_layers})")
        self.wrapped: nn.Module = wrapped
        self.cache: DistributedBlockPagedKVCache = cache
        self.layer_id: int = int(layer_id)
        self.dropout: float = float(dropout)

    # -- introspection -------------------------------------------------- #
    @property
    def original(self) -> nn.Module:
        """Alias for the wrapped (pre-patch) attention module."""
        return self.wrapped

    def _project_dirty_qkv(
        self, dirty: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Project dirty hidden states to Q/K/V with global heads.

        Args:
            dirty: ``[..., S, D]`` hidden states for dirty frames only.

        Returns:
            ``(q, k, v, num_heads, head_dim)`` with shapes ``[..., S, H, Dh]``.
        """
        num_heads, head_dim = _resolve_head_geometry(self.wrapped, self.cache)
        inner = num_heads * head_dim
        # Align hidden dtype to the projection weights first: CPU/CUDA linear
        # kernels reject mixed bf16/fp8 inputs against fp32 weights, and fp8
        # GEMMs are unsupported — so project in the weight dtype, then the
        # attention math below upcasts to fp32 anyway.
        for name in ("to_q", "q_proj", "to_qkv"):
            proj = getattr(self.wrapped, name, None)
            if isinstance(proj, nn.Linear):
                if dirty.dtype != proj.weight.dtype:
                    dirty = dirty.to(proj.weight.dtype)
                break
        try:
            q = _linear(self.wrapped, ("to_q", "q_proj"), dirty)
            k = _linear(self.wrapped, ("to_k", "k_proj"), dirty)
            v = _linear(self.wrapped, ("to_v", "v_proj"), dirty)
        except AttributeError:
            qkv = _linear(self.wrapped, ("to_qkv",), dirty)
            if qkv.shape[-1] != 3 * inner:
                raise ValueError(
                    f"fused qkv last dim {qkv.shape[-1]} != 3*H*Dh ({3 * inner})"
                )
            q, k, v = qkv.chunk(3, dim=-1)
        s = dirty.shape[-2]
        q = q.view(*dirty.shape[:-1], num_heads, head_dim).reshape(
            *dirty.shape[:-2], s, num_heads, head_dim
        )
        k = k.view(*dirty.shape[:-1], num_heads, head_dim).reshape(
            *dirty.shape[:-2], s, num_heads, head_dim
        )
        v = v.view(*dirty.shape[:-1], num_heads, head_dim).reshape(
            *dirty.shape[:-2], s, num_heads, head_dim
        )
        return q, k, v, num_heads, head_dim

    def _apply_out_proj(self, ctx_global: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped output projection to full-head context.

        Args:
            ctx_global: ``[..., S, H*Dh]`` attention context (global heads).
        """
        for name in ("to_out", "out_proj", "o_proj"):
            proj = getattr(self.wrapped, name, None)
            if isinstance(proj, nn.Linear):
                if ctx_global.dtype != proj.weight.dtype:
                    return proj(ctx_global.to(proj.weight.dtype)).to(ctx_global.dtype)
                return proj(ctx_global)
            if isinstance(proj, nn.Module):  # e.g. Sequential out-block
                return proj(ctx_global)
        raise AttributeError(
            f"{type(self.wrapped).__name__} has no output projection "
            "(looked for to_out/out_proj/o_proj)"
        )

    def _gather_full_kv(
        self,
        branch_id: str,
        total_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather ``[Tt, S, Hl, Dh]`` K/V for frames ``[0, total_frames)``.

        Block-wise copies (one slice per physical block, no cat/stack in a
        Python token loop). Returned tensors are fresh (safe to use in math).
        """
        c = self.cache
        s, hl, dh = c.num_spatial_patches, c.local_num_heads, c.head_dim
        k_full = torch.empty((total_frames, s, hl, dh), dtype=dtype, device=device)
        v_full = torch.empty((total_frames, s, hl, dh), dtype=dtype, device=device)
        bt = c.block_temporal_frames
        layer = self.layer_id
        with torch.no_grad():
            table = c.branch_table(branch_id)
            for logical, pid in enumerate(table):
                f0 = logical * bt
                if f0 >= total_frames:
                    break
                nf = min(bt, total_frames - f0)
                # Views into the pre-allocated pool; narrow avoids temporaries.
                k_blk = c.k_pool[layer, pid, :nf].to(device=device, dtype=dtype)
                v_blk = c.v_pool[layer, pid, :nf].to(device=device, dtype=dtype)
                k_full.narrow(0, f0, nf).copy_(k_blk)
                v_full.narrow(0, f0, nf).copy_(v_blk)
        return k_full, v_full

    def _stable_causal_attention(
        self,
        q_dirty: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        num_heads_local: int,
        head_dim: int,
        query_frame_ids: torch.Tensor,
        key_frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-causal multi-head attention computed in float32.

        Args:
            q_dirty: ``[Td, S, Hl, Dh]`` dirty queries (any dtype).
            k_full / v_full: ``[Tt, S, Hl, Dh]`` full context (any dtype).
            query_frame_ids: ``[Td]`` logical frame index per dirty query frame.
            key_frame_ids: ``[Tt]`` logical frame index per key frame.

        Returns:
            ``[Td, S, Hl, Dh]`` context in the input hidden dtype.
        """
        in_dtype = q_dirty.dtype
        device = q_dirty.device
        qf = _upcast_for_math(q_dirty).permute(2, 0, 1, 3).reshape(
            num_heads_local, q_dirty.shape[0] * q_dirty.shape[1], head_dim
        )
        kf = _upcast_for_math(k_full).permute(2, 0, 1, 3).reshape(
            num_heads_local, k_full.shape[0] * k_full.shape[1], head_dim
        )
        vf = _upcast_for_math(v_full).permute(2, 0, 1, 3).reshape(
            num_heads_local, v_full.shape[0] * v_full.shape[1], head_dim
        )
        s = q_dirty.shape[1]
        q_frame = query_frame_ids.repeat_interleave(s).to(device)
        k_frame = key_frame_ids.repeat_interleave(s).to(device)
        mask = k_frame[None, :] <= q_frame[:, None]  # [Td*S, Tt*S] bool
        scale = float(head_dim) ** -0.5
        logits = torch.einsum("hqd,hkd->hqk", qf, kf) * scale
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        logits = logits.masked_fill(~mask, float("-inf"))
        # Max-subtraction for bf16/fp8-logit stability, then fp32 softmax.
        logits = logits - logits.amax(dim=-1, keepdim=True).clamp(min=-50.0, max=50.0)
        attn = logits.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("hqk,hkd->hqd", attn, vf)
        td = q_dirty.shape[0]
        out = out.view(num_heads_local, td, s, head_dim).permute(1, 2, 0, 3)
        return out.to(in_dtype)

    # -- forward -------------------------------------------------------- #
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        active_branch_id: Optional[str] = None,
        branch_frame_t: int = 0,
        start_frame: int = 0,
        return_full: str = "dirty_only",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run paged causal attention, or fall back to the wrapped module.

        Args:
            hidden_states: ``[T, S, D]``, ``[B, T, S, D]`` (``B`` must be 1
                on the paged path), or patched ``[B, L, D]`` with
                ``L = T * S`` where ``S = cache.num_spatial_patches``.
            encoder_hidden_states: cross-attention context, forwarded
                untouched (paged path implements self-attention; callers
                needing cross-attention should use the fallback path).
            temb: flow-matching timestep embedding. **Passed through
                untouched** — never consumed or mutated here.
            attention_mask: optional mask, forwarded on the fallback path.
            block_table: optional ``int64`` tensor of physical block ids
                (logical->physical). When given with ``active_branch_id`` it
                is validated against the cache table and updated in place
                with any CoW repointing so callers stay in sync.
            active_branch_id: cache branch to read/write. ``None`` selects
                the fallback (original) path.
            branch_frame_t: first dirty frame index. Frames
                ``[0, branch_frame_t)`` are clean (served from cache);
                ``[branch_frame_t, ...)`` are recomputed.
            start_frame: logical frame index of ``hidden_states[0]``.
            return_full: ``"dirty_only"`` (default) returns dirty outputs
                ``[Td, ...]``; ``"passthrough"`` returns full-length outputs
                where clean slots carry the inputs through.
            **kwargs: forwarded verbatim on the fallback path (e.g.
                ``cross_attention_kwargs``, ``timestep`` aliases).

        Returns:
            Attention outputs (pre-residual), matching the input layout with
            ``T`` replaced by ``Td`` (dirty frames) unless ``return_full``
            is ``"passthrough"``.
        """
        c = self.cache
        if active_branch_id is None:
            # No branch context: exact legacy behavior, conditioning untouched.
            return self.wrapped(
                hidden_states,
                *(() if encoder_hidden_states is None else (encoder_hidden_states,)),
                **{
                    **({"temb": temb} if temb is not None else {}),
                    **({"attention_mask": attention_mask} if attention_mask is not None else {}),
                    **kwargs,
                },
            ) if self._takes_kwargs() else self._call_legacy(
                hidden_states, encoder_hidden_states, temb, attention_mask, kwargs
            )

        # ---- normalize layout to [B, T_in, S, D] ---- #
        s_expected = c.num_spatial_patches
        orig_layout: str
        batch: int
        if hidden_states.dim() == 3:  # [T, S, D] or [B, L, D] patched
            t_or_b, second, d = hidden_states.shape
            if second == s_expected:
                orig_layout = "TSD"
                t_in, s_in = t_or_b, second
                batch = 1
                h4 = hidden_states.unsqueeze(0)
            elif second % s_expected == 0:
                orig_layout = "BLD"
                batch, l_in, d = t_or_b, second, d
                t_in, s_in = l_in // s_expected, s_expected
                h4 = hidden_states.view(batch, t_in, s_in, d)
            else:
                raise ValueError(
                    f"hidden_states shape {tuple(hidden_states.shape)} matches "
                    f"neither [T, S={s_expected}, D] nor [B, L, D] patched layout"
                )
        elif hidden_states.dim() == 4:  # [B, T, S, D]
            orig_layout = "BTSD"
            batch, t_in, s_in, d = hidden_states.shape
            h4 = hidden_states
        elif hidden_states.dim() == 2:  # [L, D] single-batch patched
            orig_layout = "LD"
            batch = 1
            l_in, d = hidden_states.shape
            if l_in % s_expected != 0:
                raise ValueError(f"L={l_in} not divisible by S={s_expected}")
            t_in, s_in = l_in // s_expected, s_expected
            h4 = hidden_states.view(1, t_in, s_in, d)
        elif hidden_states.dim() == 3 and hidden_states.shape[0] != 0 and (
            hidden_states.shape[1] % s_expected == 0 and hidden_states.shape[2] > 0
        ) and False:  # pragma: no cover - retained for layout documentation
            raise ValueError("unreachable")
        else:  # pragma: no cover - defensive; all tensor dims handled above
            raise ValueError(f"unsupported hidden_states shape {tuple(hidden_states.shape)}")
        if s_in != s_expected:
            raise ValueError(
                f"S={s_in} != cache.num_spatial_patches={s_expected}; "
                "paged path requires frame/space split to match the pool"
            )
        if batch != 1:
            raise ValueError("paged path requires batch size 1 (per-branch cache)")
        if t_in == 0:
            return hidden_states[:0] if return_full == "dirty_only" else hidden_states.clone()
        if branch_frame_t < 0 or start_frame < 0:
            raise ValueError("branch_frame_t and start_frame must be >= 0")
        if return_full not in ("dirty_only", "passthrough"):
            raise ValueError("return_full must be 'dirty_only' or 'passthrough'")

        t_total = start_frame + t_in
        first_dirty = max(start_frame, int(branch_frame_t))
        dirty_lo = first_dirty - start_frame  # offset into h4
        num_heads_global, head_dim = _resolve_head_geometry(self.wrapped, c)
        if num_heads_global != c.num_heads or head_dim != c.head_dim:
            raise ValueError(
                f"head geometry mismatch: wrapped=({num_heads_global}x{head_dim}) "
                f"vs cache=({c.num_heads}x{c.head_dim})"
            )

        # ---- optional block_table mirror: validate + keep in sync ---- #
        if block_table is not None:
            if block_table.dtype not in (torch.int32, torch.int64):
                raise ValueError("block_table must be an integer tensor")
            live = c.branch_table(active_branch_id)
            flat = block_table.reshape(-1).to("cpu")
            if flat.numel() < len(live):
                raise ValueError(
                    f"block_table has {flat.numel()} entries but branch "
                    f"{active_branch_id!r} needs {len(live)}"
                )
            for i, pid in enumerate(live):
                if int(flat[i].item()) != int(pid):
                    raise ValueError(
                        f"block_table[{i}]={int(flat[i].item())} != "
                        f"cache pid {int(pid)} for branch {active_branch_id!r}"
                    )

        # ---- full-reuse fast path: nothing dirty ---- #
        if dirty_lo >= t_in:
            if return_full == "dirty_only":
                return hidden_states[:0].clone() if orig_layout != "BTSD" else hidden_states[:, :0].clone()
            return hidden_states.clone()

        dirty = h4[:, dirty_lo:, :, :]  # [1, Td, S, D]
        td = t_in - dirty_lo
        dirty_frames = torch.arange(first_dirty, first_dirty + td)

        # ---- project dirty Q/K/V only (prefix bypasses Q proj + attn) ---- #
        # NOTE: projections run with grad enabled so dirty queries keep
        # gradients; cache commits below detach (pool tensors never track grad).
        q, k, v, hg, dh = self._project_dirty_qkv(dirty.squeeze(0))
        # q/k/v: [Td, S, H, Dh] (global heads); shard to this rank for cache.
        hs, he = c.local_head_start, c.local_head_end
        k_local = k[:, :, hs:he, :].contiguous()
        v_local = v[:, :, hs:he, :].contiguous()
        q_local = q[:, :, hs:he, :]

        # ---- commit dirty K/V (CoW on shared blocks), grow tail as needed ---- #
        c.ensure_frames(active_branch_id, t_total)
        for i in range(td):
            f = first_dirty + i
            c.write_block(active_branch_id, f, k_local[i], v_local[i], layer=self.layer_id)

        # ---- propagate CoW repointing back into the caller's mirror ---- #
        if block_table is not None:
            live = c.branch_table(active_branch_id)
            flat = block_table.reshape(-1)
            for i, pid in enumerate(live):
                if i < flat.numel():
                    flat[i] = int(pid)
            # reshape() may return a copy for non-contiguous inputs; push back.
            try:
                block_table.copy_(flat.view(block_table.shape))
            except RuntimeError:
                pass

        # ---- gather full K/V context (prefix read straight from pool) ---- #
        k_full, v_full = self._gather_full_kv(
            active_branch_id, t_total, device=h4.device, dtype=h4.dtype
        )
        key_frames = torch.arange(t_total)

        # ---- causal attention of dirty queries over full context ---- #
        ctx_local = self._stable_causal_attention(
            q_local.to(h4.device),
            k_full,
            v_full,
            c.local_num_heads,
            dh,
            dirty_frames.to(h4.device),
            key_frames.to(h4.device),
        )  # [Td, S, Hl, Dh]

        # ---- output projection ---- #
        if c.world_size == 1:
            ctx_merged = ctx_local.reshape(td, s_in, hg * dh)
            out_dirty = self._apply_out_proj(ctx_merged.to(h4.dtype))  # [Td, S, D]
        else:
            # Sequence-parallel: partial out-proj on this rank's head slice.
            # The caller sums partials across ranks (all-reduce), matching
            # Megatron column-parallel semantics.
            ctx_flat = ctx_local.reshape(td, s_in, c.local_num_heads * dh)
            out_dirty = self._partial_out_proj(ctx_flat)

        out_dirty = out_dirty.unsqueeze(0)  # [1, Td, S, D]
        if return_full == "passthrough":
            full = h4.clone()
            full[:, dirty_lo:, :, :] = out_dirty.to(full.dtype)
            out4 = full
        else:
            out4 = out_dirty.to(h4.dtype)

        if orig_layout == "TSD":
            return out4.squeeze(0)
        if orig_layout == "BTSD":
            return out4
        if orig_layout == "LD":
            return out4.squeeze(0).reshape(out4.shape[1] * s_in, d)
        # BLD patched
        return out4.reshape(batch, out4.shape[1] * s_in, d)

    # -- sharded out-proj / legacy call helpers ------------------------- #
    def _partial_out_proj(self, ctx_flat: torch.Tensor) -> torch.Tensor:
        """Rank-local slice of the output projection (column-parallel).

        Uses this rank's column slice of ``to_out.weight`` so that summing
        partials across ranks reproduces the dense projection (bias, if any,
        is added once by rank 0 to avoid double-counting).
        """
        proj = None
        for name in ("to_out", "out_proj", "o_proj"):
            mod = getattr(self.wrapped, name, None)
            if isinstance(mod, nn.Linear):
                proj = mod
                break
        if proj is None:
            raise AttributeError("sharded path requires a Linear output projection")
        hl, dh = self.cache.local_num_heads, self.cache.head_dim
        w = proj.weight  # [D, H*Dh]
        s0 = self.cache.local_head_start * dh
        w_local = w[:, s0 : s0 + hl * dh].to(ctx_flat.dtype)
        out = ctx_flat @ w_local.t()
        if proj.bias is not None and self.cache.rank == 0:
            out = out + proj.bias.to(out.dtype)
        return out

    def _takes_kwargs(self) -> bool:
        """Heuristic: diffusers attentions accept temb-style kwargs."""
        import inspect

        try:
            sig = inspect.signature(self.wrapped.forward)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return False
        for p in sig.parameters.values():
            if p.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                return True
        names = set(sig.parameters)
        return bool(names & {"temb", "encoder_hidden_states", "attention_mask", "kwargs"})

    def _call_legacy(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Call a minimal ``(hidden_states)``-style attention module."""
        try:
            return self.wrapped(hidden_states)  # type: ignore[operator]
        except TypeError:
            return self.wrapped(hidden_states, temb)  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# install / remove                                                            #
# --------------------------------------------------------------------------- #

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
        if any(
            hasattr(child, a)
            for a in ("to_q", "to_k", "q_proj", "k_proj", "to_qkv")
        ):
            return name
    return None


@dataclass
class InterceptorHandle:
    """Record of installed wrappers; call :meth:`uninstall` to restore."""

    wrappers: List[CausalPagedAttentionWrapper] = field(default_factory=list)
    _originals: List[Tuple[nn.Module, str, nn.Module]] = field(default_factory=list)
    default_branch_id: Optional[str] = None
    default_branch_frame_t: int = 0

    def set_default_branch(self, branch_id: str, branch_frame_t: int = 0) -> None:
        """Set fallback branch context used by convenience callers."""
        if branch_frame_t < 0:
            raise ValueError("branch_frame_t must be >= 0")
        self.default_branch_id = branch_id
        self.default_branch_frame_t = int(branch_frame_t)

    def uninstall(self) -> None:
        """Restore every wrapped module to its original submodule."""
        for parent, attr, original in self._originals:
            setattr(parent, attr, original)
        self.wrappers.clear()
        self._originals.clear()

    def __len__(self) -> int:
        return len(self.wrappers)


def install_interceptor(
    pipeline: Any,
    cache: DistributedBlockPagedKVCache,
    *,
    layer_ids: Optional[Sequence[int]] = None,
    dropout: float = 0.0,
    verbose: bool = True,
) -> InterceptorHandle:
    """Monkey-patch Wan self-attention modules with paged-CoW wrappers.

    Args:
        pipeline: a ``diffusers`` Wan pipeline (uses ``.transformer``), a
            ``WanTransformer3DModel`` itself, or any object exposing a block
            sequence (``blocks``/``layers``/``transformer_blocks``) whose
            blocks expose a self-attention submodule. A ``FakeWan``-style
            test double works too.
        cache: the :class:`DistributedBlockPagedKVCache` to hook in.
            ``cache.num_layers`` must cover the wrapped layer ids.
        layer_ids: subset of block indices to wrap; ``None`` wraps all.
        dropout: attention dropout for the wrappers (0.0 at inference).
        verbose: print a one-line summary per wrapped layer.

    Returns:
        An :class:`InterceptorHandle` with ``.wrappers`` and ``.uninstall()``.
    """
    transformer: Any = getattr(pipeline, "transformer", None)
    if transformer is None:
        transformer = pipeline
    if not isinstance(transformer, nn.Module):
        raise TypeError(
            f"install_interceptor expected a pipeline/transformer nn.Module, "
            f"got {type(pipeline).__name__}"
        )
    blocks = _iter_blocks(transformer)
    wanted = set(layer_ids) if layer_ids is not None else None
    handle = InterceptorHandle()
    for block, idx in blocks:
        if wanted is not None and idx not in wanted:
            continue
        if idx >= cache.num_layers:
            raise ValueError(
                f"layer {idx} out of cache range [0, {cache.num_layers}); "
                "construct the cache with num_layers >= num transformer blocks"
            )
        attr = _find_attn(block)
        if attr is None:
            if verbose:
                print(f"[interceptor] layer {idx}: no self-attention found, skipped")
            continue
        original = getattr(block, attr)
        if isinstance(original, CausalPagedAttentionWrapper):
            continue  # idempotent: already intercepted
        wrapper = CausalPagedAttentionWrapper(
            wrapped=original, cache=cache, layer_id=idx, dropout=dropout
        )
        setattr(block, attr, wrapper)
        handle.wrappers.append(wrapper)
        handle._originals.append((block, attr, original))
        if verbose:
            print(
                f"[interceptor] layer {idx}: wrapped {type(block).__name__}.{attr} "
                f"({type(original).__name__}) -> CausalPagedAttentionWrapper"
            )
    if not handle.wrappers:
        raise RuntimeError("install_interceptor wrapped 0 attention modules")
    return handle


def remove_interceptor(handle: InterceptorHandle) -> None:
    """Convenience alias for ``handle.uninstall()``."""
    handle.uninstall()
