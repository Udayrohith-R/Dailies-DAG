"""Forked causal spatio-temporal self-attention blocks for Dailies-DAG.

We fork the standard DiT temporal attention (cf. Wan2.1 / LTX-Video `attn2`) into a
version that NEVER materializes the full K/V sequence. Instead it reads/writes through
`TemporalBlockManager` via a `block_table` argument (vLLM PagedAttention style).

Forward protocol (per layer, per sequence):
  1. Project hidden_states -> q, k, v for the *incoming* frames only.
  2. `manager.write_frame(table, frame, k, v, layer)` for each new frame
     (CoW-clones iff the block is shared with another branch).
  3. `manager.read_sequence(table, T, layer)` to gather K/V for causal attention.
  4. Causal scaled-dot-product attention over (T*L) flattened tokens with a
     frame-causal mask (frame f attends to frames <= f).

`block_table` is a REQUIRED argument — there is no non-paged path. Callers obtain it
from `TemporalBlockManager.alloc_seq / fork / ensure_frame`.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalAttention(nn.Module):
    """Causal temporal self-attention with block-paged KV-cache.

    Hidden layout: [T, L, D] (frames, spatial-tokens-per-frame, model dim).
    We keep batch=1 per sequence for clarity; batch by looping over tables.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        block_table: List[int],
        cache_manager,  # TemporalBlockManager (untyped to avoid circulars)
        layer_id: int,
        start_frame: int = 0,
        branch_frame_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Run paged causal attention.

        Args:
            hidden_states: [T_new, L, D] incoming frames to append/read.
            block_table: paged address table for this branch. MUTATED in place on CoW.
            cache_manager: TemporalBlockManager instance.
            layer_id: which transformer layer's pool to use.
            start_frame: logical frame index of hidden_states[0] (for appends).
            branch_frame_idx: if not None, convenience fork trigger — when set, the
                caller is expected to have already called
                `manager.fork(table, branch_frame_idx)` and passed the CHILD table
                here. Documented explicitly so the edit point is visible in the
                forward signature (zero-latency branching @ frame N).

        Returns:
            [T_new, L, D] attended outputs aligned to the input frames.
        """
        T_new, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.to_qkv(hidden_states)  # [T_new, L, 3*H*Dh]
        qkv = qkv.view(T_new, L, 3, H, Dh)
        q, k, v = qkv.unbind(dim=2)  # each [T_new, L, H, Dh]

        # 1) Persist incoming frames into the paged cache (CoW on shared blocks).
        for i in range(T_new):
            f = start_frame + i
            cache_manager.ensure_frame(block_table, f)
            cache_manager.write_frame(block_table, f, k[i], v[i], layer=layer_id)

        # 2) Gather full prefix K/V for causal attention.
        T_total = start_frame + T_new
        K_full, V_full = cache_manager.read_sequence(block_table, T_total, layer=layer_id)
        # K_full: [T_total, L, H, Dh]

        # 3) Flatten frames*spatial into one token stream, frame-major.
        # token t*L+s corresponds to frame t. Causal mask at frame granularity.
        qf = q.reshape(T_new, L, H, Dh).permute(2, 0, 1, 3).reshape(H, T_new * L, Dh)
        Kf = K_full.permute(2, 0, 1, 3).reshape(H, T_total * L, Dh)
        Vf = V_full.permute(2, 0, 1, 3).reshape(H, T_total * L, Dh)

        # Frame-causal mask: query frame fq=start_frame+i attends to key frames <= fq.
        q_frame = torch.arange(start_frame, start_frame + T_new, device=q.device).repeat_interleave(L)
        k_frame = torch.arange(T_total, device=q.device).repeat_interleave(L)
        mask = k_frame[None, :] <= q_frame[:, None]  # [T_new*L, T_total*L] bool

        scale = Dh**-0.5
        attn_logits = torch.einsum("hqd,hkd->hqk", qf, Kf) * scale
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("hqk,hkd->hqd", attn, Vf)  # [H, T_new*L, Dh]
        out = out.view(H, T_new, L, Dh).permute(1, 2, 0, 3).reshape(T_new, L, H * Dh)
        return self.to_out(out)


class SpatioTemporalDiTBlock(nn.Module):
    """Minimal DiT block: spatial-mix (vanilla) + forked causal temporal attn + MLP.

    Mirrors LTX/Wan block structure enough to demonstrate injection. The spatial
    part is intentionally ordinary attention; the temporal part is the paged one.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        block_table: List[int],
        cache_manager,
        layer_id: int,
        start_frame: int = 0,
        branch_frame_idx: Optional[int] = None,
    ) -> torch.Tensor:
        h = hidden_states + self.temporal_attn(
            self.norm1(hidden_states),
            block_table=block_table,
            cache_manager=cache_manager,
            layer_id=layer_id,
            start_frame=start_frame,
            branch_frame_idx=branch_frame_idx,
        )
        return h + self.mlp(self.norm2(h))
