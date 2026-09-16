"""TemporalBlockManager: vLLM-style block-paged KV-cache ported to 3D video latents.

Design
------
Video DiT latents are 5D conceptually: [layers, frames T, spatial tokens L, heads H, head_dim Dh].
We page the *temporal* axis into fixed-size blocks of S frames (vLLM pages tokens;
we page frames so CoW stays frame-aligned for editing).

Terminology (mirrors vLLM / PagedAttention):
  * physical block  — one tensor chunk holding K and V for S consecutive frames:
        K_block, V_block : [S, L, H, Dh]
  * block_table     — per-sequence list mapping logical_block_idx -> physical_block_id.
                      This is the ONLY per-branch state. Forking copies this list (O(n/S) ints).
  * refcount[pid]   — strict reference count of physical blocks. Shared prefix blocks
                      have refcount > 1 after fork. Any write to a shared block triggers
                      Copy-on-Write: clone the block, decrement the old refcount, point
                      the writer's table at the clone. Readers never clone.

Memory guarantee
----------------
fork(src_table, branch_frame_idx):
  * frames [0, branch_frame_idx)  -> pointer copies, refcount++. Cost O(num_blocks_prefix) ints.
  * frames [branch_frame_idx, T)  -> NOT copied. Caller writes fresh frames into newly
    allocated tail blocks (or lazily on demand). Zero tensor memcpy for the prefix.

Physical pool is lazily allocated (list of None | Tensor) so CPU tests with
max_num_blocks=128 don't pre-allocate gigabytes.

Frame layout inside a block
----------------------------
logical frame f  ->  logical_block = f // S,  offset = f % S.
physical block pid = block_table[logical_block]; slot = offset.
K[pid][offset] is the K tensor for that frame: [L, H, Dh].

Multi-layer support: pools are per-layer: self._k[layer][pid], self._v[layer][pid],
refcounts shared across layers? No — refcounts are per (layer, pid) pair because each
layer has its own tensors. We store refcount[layer][pid].

Threading: all mutating ops take self._lock (threading.Lock). Asyncio orchestration
runs forks from a single event loop; Trio scheduler serializes via a mutex — the lock
here is a last line of defence, not the primary protocol.

Example
-------
>>> mgr = TemporalBlockManager(num_layers=2, num_heads=4, head_dim=8,
...     tokens_per_frame=16, block_size=8, max_num_blocks=32)
>>> table_a = mgr.alloc_seq(num_frames=120)   # 15 logical blocks
>>> mgr.write_frame(table_a, 0, k0, v0, layer=0)  # k0: [L,H,Dh]
>>> table_b = mgr.fork(table_a, branch_frame_idx=60)  # prefix pointers shared
>>> assert mgr.shared_blocks(table_a, table_b) == 8   # 60/8 rounded up (partial shares)
>>> mgr.write_frame(table_b, 60, k_new, v_new, layer=0)  # CoW clone if shared
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Dict, List, Optional

import torch


class BlockExhaustedError(RuntimeError):
    """Raised when the physical block pool is out of free blocks."""


class TemporalBlockManager:
    """Block-paged KV-cache manager for causal video DiTs with strict ref-counted CoW."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        tokens_per_frame: int,
        block_size: int = 8,
        max_num_blocks: int = 128,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        if max_num_blocks <= 0:
            raise ValueError("max_num_blocks must be > 0")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tokens_per_frame = tokens_per_frame  # L = spatial tokens per frame
        self.block_size = block_size  # S = frames per physical block
        self.max_num_blocks = max_num_blocks
        self.dtype = dtype
        self.device = torch.device(device)

        # Per-layer pools: list of (max_num_blocks,) entries, None = unallocated.
        # Block tensor shape: [S, L, H, Dh]
        self._k: List[List[Optional[torch.Tensor]]] = [
            [None] * max_num_blocks for _ in range(num_layers)
        ]
        self._v: List[List[Optional[torch.Tensor]]] = [
            [None] * max_num_blocks for _ in range(num_layers)
        ]
        # Strict refcounts, per (layer, pid). 0 = free.
        self._refcount: List[List[int]] = [
            [0] * max_num_blocks for _ in range(num_layers)
        ]
        # Free physical ids are GLOBAL (a pid is free only when refcount==0 in ALL layers).
        # Invariant: pid free <=> every layer refcount[l][pid] == 0 and slots are None.
        self._free: deque[int] = deque(range(max_num_blocks))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _block_shape(self) -> torch.Size:
        return torch.Size(
            [self.block_size, self.tokens_per_frame, self.num_heads, self.head_dim]
        )

    def _alloc_pid(self) -> int:
        """Pop one free physical block id. Caller must hold lock."""
        if not self._free:
            raise BlockExhaustedError(
                f"No free physical blocks (max={self.max_num_blocks}). "
                "Free a sequence or raise max_num_blocks."
            )
        return self._free.popleft()

    def _ensure_materialized(self, layer: int, pid: int) -> None:
        """Lazily allocate the tensor storage for (layer, pid)."""
        if self._k[layer][pid] is None:
            self._k[layer][pid] = torch.zeros(
                self._block_shape(), dtype=self.dtype, device=self.device
            )
            self._v[layer][pid] = torch.zeros(
                self._block_shape(), dtype=self.dtype, device=self.device
            )

    def _retain(self, layer: int, pid: int) -> None:
        self._refcount[layer][pid] += 1

    def _release(self, layer: int, pid: int) -> None:
        rc = self._refcount[layer][pid]
        if rc <= 0:
            raise RuntimeError(f"refcount underflow for layer={layer} pid={pid}")
        self._refcount[layer][pid] = rc - 1
        if self._all_layers_free(pid):
            # Wipe storage so a stale reader can't observe freed memory.
            self._k[layer] = self._k[layer]  # no-op, clarity
            for l in range(self.num_layers):
                self._k[l][pid] = None
                self._v[l][pid] = None
            self._free.append(pid)

    def _all_layers_free(self, pid: int) -> bool:
        return all(self._refcount[l][pid] == 0 for l in range(self.num_layers))

    def _num_logical_blocks(self, num_frames: int) -> int:
        return math.ceil(num_frames / self.block_size) if num_frames > 0 else 0

    # ------------------------------------------------------------------ #
    # public API: allocation                                              #
    # ------------------------------------------------------------------ #

    def alloc_seq(self, num_frames: int) -> List[int]:
        """Allocate a fresh sequence of `num_frames` frames.

        Returns a `block_table` (list of physical pids, one per logical block).
        Every layer's refcount for each pid is set to 1 (one owner).

        NOTE: all layers share the same pids for a logical block (mirrors vLLM,
        where one block-table entry addresses K/V across layers). Refcounts are
        still tracked per layer so per-layer free is exact.
        """
        n_blocks = self._num_logical_blocks(num_frames)
        with self._lock:
            table: List[int] = []
            for _ in range(n_blocks):
                pid = self._alloc_pid()
                for layer in range(self.num_layers):
                    self._ensure_materialized(layer, pid)
                    self._refcount[layer][pid] = 1
                table.append(pid)
            return table

    def free_seq(self, block_table: List[int]) -> None:
        """Release every physical block referenced by `block_table` (refcount--)."""
        with self._lock:
            for pid in block_table:
                for layer in range(self.num_layers):
                    if self._refcount[layer][pid] > 0:
                        self._release(layer, pid)
                    # else: already freed via shared tail — skip (idempotent-ish
                    # for double-free of the same table; strict underflow still
                    # raises inside _release when >0 expected but 0 found twice
                    # via *different* tables — that path is a real bug).

    # ------------------------------------------------------------------ #
    # public API: CoW fork  (THE primitive)                               #
    # ------------------------------------------------------------------ #

    def fork(
        self, src_table: List[int], branch_frame_idx: int, tail_frames: int = 0
    ) -> List[int]:
        """Fork `src_table` at `branch_frame_idx` with strict ref-counted CoW.

        Args:
            src_table: parent sequence block_table.
            branch_frame_idx: first frame index owned by the child branch.
                Frames [0, branch_frame_idx) are SHARED (pointer copy + retain).
                Frames [branch_frame_idx, ...) are FRESH (newly allocated tail).
            tail_frames: how many fresh frames to pre-allocate for the child
                beyond the branch point. 0 = allocate lazily via `ensure_frame`.

        Returns:
            New child block_table. Shared prefix entries are the SAME pids with
            bumped refcounts; tail entries are fresh pids (or absent if lazy).

        Cost: O(ceil(branch_frame_idx / S)) pointer copies + refcount bumps.
        Tensor memory duplicated: ZERO at fork time. Clones happen later, one
        block at a time, inside `write_frame` iff refcount > 1 (true CoW).

        Example: T=120, S=8, branch=60 -> 8 logical prefix blocks shared
        (frames 0..63 cover branch 60; block 7 is partially shared — still CoW
        at slot granularity, see `write_frame`).
        """
        if branch_frame_idx < 0:
            raise ValueError("branch_frame_idx must be >= 0")
        with self._lock:
            # Number of logical blocks whose frames are entirely-or-partially < branch.
            # Partially-overlapped boundary block is STILL shared (slot-level CoW later).
            n_prefix_blocks = math.ceil(branch_frame_idx / self.block_size) if branch_frame_idx > 0 else 0
            n_prefix_blocks = min(n_prefix_blocks, len(src_table))

            child: List[int] = []
            # 1) Shared prefix: duplicate POINTERS, bump refcounts. No memcpy.
            for i in range(n_prefix_blocks):
                pid = src_table[i]
                for layer in range(self.num_layers):
                    self._retain(layer, pid)
                child.append(pid)

            # 2) Fresh tail: pre-allocate if requested.
            n_tail_blocks = self._num_logical_blocks(tail_frames)
            for _ in range(n_tail_blocks):
                pid = self._alloc_pid()
                for layer in range(self.num_layers):
                    self._ensure_materialized(layer, pid)
                    self._refcount[layer][pid] = 1
                child.append(pid)
            return child

    def ensure_frame(self, block_table: List[int], frame_idx: int) -> List[int]:
        """Grow `block_table` (in place) so `frame_idx` is addressable, allocating tail blocks.

        Returns the same (mutated) table for chaining. Fresh blocks have refcount 1.
        """
        need = self._num_logical_blocks(frame_idx + 1)
        with self._lock:
            while len(block_table) < need:
                pid = self._alloc_pid()
                for layer in range(self.num_layers):
                    self._ensure_materialized(layer, pid)
                    self._refcount[layer][pid] = 1
                block_table.append(pid)
        return block_table

    # ------------------------------------------------------------------ #
    # public API: read / write (writes are CoW)                           #
    # ------------------------------------------------------------------ #

    def _cow_for_write(self, layer: int, block_table: List[int], logical: int) -> int:
        """Return a WRITABLE pid for `block_table[logical]` at `layer`.

        If the physical block is shared (refcount > 1 in ANY layer — we check the
        union to keep cross-layer pid aliasing safe), clone it: allocate a fresh
        pid, memcpy this layer's K/V (other layers memcpy too so pid aliasing is
        preserved), decrement old refcounts, set new refcounts to 1, and repoint
        this sequence's table entry. Cost O(S) — one block, not the sequence.

        Must be called with lock held.
        """
        old_pid = block_table[logical]
        shared = any(self._refcount[l][old_pid] > 1 for l in range(self.num_layers))
        if not shared:
            self._ensure_materialized(layer, old_pid)
            return old_pid
        # --- CoW clone path ---
        new_pid = self._alloc_pid()
        for l in range(self.num_layers):
            self._ensure_materialized(l, new_pid)
            if self._k[l][old_pid] is not None:
                self._k[l][new_pid].copy_(self._k[l][old_pid])
                self._v[l][new_pid].copy_(self._v[l][old_pid])
            # move ONE reference from old to new for every layer that held one
            if self._refcount[l][old_pid] > 0:
                self._refcount[l][old_pid] -= 1
                if self._all_layers_free(old_pid):
                    for ll in range(self.num_layers):
                        self._k[ll][old_pid] = None
                        self._v[ll][old_pid] = None
                    self._free.append(old_pid)
            self._refcount[l][new_pid] = 1
        block_table[logical] = new_pid
        return new_pid

    def write_frame(
        self,
        block_table: List[int],
        frame_idx: int,
        k_frame: torch.Tensor,
        v_frame: torch.Tensor,
        layer: int,
    ) -> None:
        """Write one frame's K/V into the paged cache, CoW-cloning iff shared.

        Args:
            block_table: sequence table (mutated in place on CoW repoint).
            frame_idx: logical frame index.
            k_frame, v_frame: [L, H, Dh] tensors for this frame.
            layer: transformer layer index.
        """
        exp = (self.tokens_per_frame, self.num_heads, self.head_dim)
        if tuple(k_frame.shape) != exp or tuple(v_frame.shape) != exp:
            raise ValueError(f"expected frame shape {exp}, got {tuple(k_frame.shape)}")
        with self._lock:
            if frame_idx // self.block_size >= len(block_table):
                raise IndexError(
                    f"frame {frame_idx} out of range (table has {len(block_table)} "
                    f"blocks). Call ensure_frame first."
                )
            logical = frame_idx // self.block_size
            offset = frame_idx % self.block_size
            pid = self._cow_for_write(layer, block_table, logical)
            self._k[layer][pid][offset].copy_(k_frame.to(dtype=self.dtype, device=self.device))
            self._v[layer][pid][offset].copy_(v_frame.to(dtype=self.dtype, device=self.device))

    def read_sequence(
        self, block_table: List[int], num_frames: int, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K, V for frames [0, num_frames) into contiguous tensors.

        Returns: (K, V), each [num_frames, L, H, Dh]. Read-only: never clones.
        """
        if num_frames == 0:
            z = torch.zeros(
                (0, self.tokens_per_frame, self.num_heads, self.head_dim),
                dtype=self.dtype, device=self.device,
            )
            return z, z.clone()
        with self._lock:
            ks: List[torch.Tensor] = []
            vs: List[torch.Tensor] = []
            for f in range(num_frames):
                logical, offset = f // self.block_size, f % self.block_size
                pid = block_table[logical]
                ks.append(self._k[layer][pid][offset])
                vs.append(self._v[layer][pid][offset])
            return torch.stack(ks, dim=0), torch.stack(vs, dim=0)

    # ------------------------------------------------------------------ #
    # introspection                                                       #
    # ------------------------------------------------------------------ #

    def refcount(self, layer: int, pid: int) -> int:
        return self._refcount[layer][pid]

    def shared_blocks(self, a: List[int], b: List[int]) -> int:
        """Count logical positions sharing the same physical pid (prefix overlap)."""
        return sum(1 for x, y in zip(a, b) if x == y)

    def num_free_blocks(self) -> int:
        return len(self._free)

    def memory_stats(self) -> Dict[str, int]:
        used = self.max_num_blocks - len(self._free)
        return {
            "max_blocks": self.max_num_blocks,
            "used_blocks": used,
            "free_blocks": len(self._free),
            "block_size_frames": self.block_size,
        }
