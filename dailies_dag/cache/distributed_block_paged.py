"""Distributed block-paged KV-cache with zero-copy Copy-on-Write branching.

Video latent topology
---------------------
Video tokens are 3D spatio-temporal patches. One *physical block* stores K and V
for ``block_temporal_frames`` consecutive frames::

    K_block, V_block : [Bt, S, Hl, Dh]

where

* ``Bt`` = ``block_temporal_frames`` (default 4),
* ``S``  = ``num_spatial_patches`` (spatial tokens per frame),
* ``Hl`` = ``local_num_heads`` = ``num_heads // world_size`` (head shard held by
  this sequence-parallel rank),
* ``Dh`` = ``head_dim``.

Defaults align with Wan2.1-14B attention geometry::

    WAN21_14B_NUM_HEADS = 40
    WAN21_14B_HEAD_DIM  = 128
    WAN21_14B_NUM_LAYERS = 40

so ``dim = 40 * 128 = 5120``. Tests override these with tiny shapes; production
passes the Wan values explicitly.

Distributed physical memory pool
--------------------------------
* ``k_pool`` / ``v_pool`` are **pre-allocated, contiguous** tensors of shape
  ``[num_layers, num_blocks, Bt, S, Hl, Dh]``. They are allocated exactly once
  in ``__init__``. The hot paths (``fork`` / ``write_block``) never call
  ``torch.cat`` / ``torch.stack`` / ``torch.empty`` — only tensor slicing and
  ``copy_``.
* Rank-aware head sharding (DeepSpeed-Ulysses / Ring-Attention compatible):
  rank ``r`` owns global heads ``[r*Hl, (r+1)*Hl)``. The constructor accepts an
  explicit ``rank`` / ``world_size`` / ``process_group`` triple and otherwise
  falls back to ``torch.distributed`` if initialized, else single-process.
* ``ref_counts: torch.Tensor`` of shape ``[num_blocks]`` (``torch.long``, CPU)
  tracks the number of logical block-table entries pointing at each physical
  block across *all* branches and *all* layers. ``0`` means free.

Copy-on-Write primitives
------------------------
* ``fork(parent_branch_id) -> str``: clones only the logical block mapping
  table (a Python ``list[int]`` of length ``num_logical_blocks``) and bumps
  each referenced ``ref_counts[pid]`` by 1. Zero tensor memcpy.
* ``write_block(branch_id, logical_frame_idx, new_k, new_v)``: resolves
  ``logical = frame_idx // Bt``, ``slot = frame_idx % Bt``. If the physical
  block is exclusively owned (``refcount == 1``) the frame slice is updated
  in place. Otherwise a free physical block is popped, the *entire* source
  block is copied (all layers, K and V), the old refcount is decremented, the
  branch table is repointed, and the new slice is written. Cost is O(one block).

Logical frame layout
--------------------
Logical frame ``f`` lives at ``table[f // Bt]``, slot ``f % Bt``. A branch
table is ``List[int]`` mapping logical-block-idx -> physical-block-idx.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# Wan2.1-14B geometry constants                                                #
# --------------------------------------------------------------------------- #

WAN21_14B_NUM_HEADS: int = 40
WAN21_14B_HEAD_DIM: int = 128
WAN21_14B_NUM_LAYERS: int = 40
WAN21_14B_HIDDEN_DIM: int = WAN21_14B_NUM_HEADS * WAN21_14B_HEAD_DIM  # 5120
DEFAULT_BLOCK_TEMPORAL_FRAMES: int = 4


class BlockExhaustedError(RuntimeError):
    """Raised when the physical block pool has no free blocks."""


class UnknownBranchError(KeyError):
    """Raised when a branch id is not registered."""


class DistributedBlockPagedKVCache:
    """Rank-sharded, pre-allocated block-paged KV-cache with ref-counted CoW.

    Args:
        num_blocks: total physical blocks in this rank's pool.
        block_temporal_frames: ``Bt``, frames per physical block (default 4).
        num_spatial_patches: ``S``, spatial tokens per frame.
        num_heads: global (unsharded) head count. Must be divisible by
            ``world_size``. Defaults to Wan2.1-14B (40).
        head_dim: per-head dim. Defaults to Wan2.1-14B (128).
        num_layers: transformer layers sharing each pid (one pid addresses
            K/V across all layers, mirroring vLLM).
        dtype: pool dtype.
        device: pool device. Defaults to CUDA if available else CPU. Pools are
            allocated once and never resized.
        rank: sequence-parallel rank. If ``None``, inferred from
            ``torch.distributed`` when initialized, else 0.
        world_size: sequence-parallel world size. If ``None``, inferred from
            ``torch.distributed`` when initialized, else 1.
        process_group: optional ``torch.distributed`` process group handle
            (DeepSpeed-Ulysses / Ring Attention). Stored opaquely; only used
            by :meth:`gather_full_heads` when distributed is initialized.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 64,
        block_temporal_frames: int = DEFAULT_BLOCK_TEMPORAL_FRAMES,
        num_spatial_patches: int = 16,
        num_heads: int = WAN21_14B_NUM_HEADS,
        head_dim: int = WAN21_14B_HEAD_DIM,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float32,
        device: Optional[str | torch.device] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        process_group: Optional[Any] = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_temporal_frames <= 0:
            raise ValueError("block_temporal_frames must be > 0")
        if num_spatial_patches <= 0:
            raise ValueError("num_spatial_patches must be > 0")
        if num_heads <= 0 or head_dim <= 0 or num_layers <= 0:
            raise ValueError("num_heads / head_dim / num_layers must be > 0")

        # ---- resolve distributed identity (dist-compatible, dist-optional) ----
        inferred_rank, inferred_world = self._infer_dist_identity()
        if rank is None:
            rank = inferred_rank
        if world_size is None:
            world_size = inferred_world
        if not (0 <= rank < world_size):
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        if num_heads % world_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"world_size ({world_size}) for head sharding"
            )

        self.rank: int = int(rank)
        self.world_size: int = int(world_size)
        self.process_group: Optional[Any] = process_group

        self.num_blocks: int = int(num_blocks)
        self.block_temporal_frames: int = int(block_temporal_frames)
        self.num_spatial_patches: int = int(num_spatial_patches)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = int(head_dim)
        self.num_layers: int = int(num_layers)
        self.dtype: torch.dtype = dtype

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device)

        self.local_num_heads: int = self.num_heads // self.world_size
        self.local_head_start: int = self.rank * self.local_num_heads
        self.local_head_end: int = self.local_head_start + self.local_num_heads

        # ---- pre-allocated contiguous physical pools (allocated ONCE) ----
        # Shape: [L, N, Bt, S, Hl, Dh]. Contiguous by construction.
        pool_shape: Tuple[int, int, int, int, int, int] = (
            self.num_layers,
            self.num_blocks,
            self.block_temporal_frames,
            self.num_spatial_patches,
            self.local_num_heads,
            self.head_dim,
        )
        self.k_pool: torch.Tensor = torch.zeros(pool_shape, dtype=dtype, device=self.device)
        self.v_pool: torch.Tensor = torch.zeros(pool_shape, dtype=dtype, device=self.device)
        assert self.k_pool.is_contiguous() and self.v_pool.is_contiguous()

        # ---- explicit reference counter per physical block index ----
        # CPU tensor: avoids device syncs on the control plane.
        self.ref_counts: torch.Tensor = torch.zeros((self.num_blocks,), dtype=torch.long)

        self._free: Deque[int] = deque(range(self.num_blocks))
        self._branches: Dict[str, List[int]] = {}
        self._branch_counter: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # distributed helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_dist_identity() -> Tuple[int, int]:
        """Return (rank, world_size) from torch.distributed if initialized."""
        try:
            import torch.distributed as dist  # type: ignore[import]

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank()), int(dist.get_world_size())
        except Exception:
            pass
        return 0, 1

    @property
    def local_head_range(self) -> Tuple[int, int]:
        """Global head interval ``[start, end)`` owned by this rank."""
        return (self.local_head_start, self.local_head_end)

    @property
    def block_shape(self) -> torch.Size:
        """Shape of one physical block (this rank's shard): ``[Bt, S, Hl, Dh]``."""
        return torch.Size(
            [
                self.block_temporal_frames,
                self.num_spatial_patches,
                self.local_num_heads,
                self.head_dim,
            ]
        )

    @property
    def frame_shape(self) -> torch.Size:
        """Shape of one frame slice (this rank's shard): ``[S, Hl, Dh]``."""
        return torch.Size(
            [self.num_spatial_patches, self.local_num_heads, self.head_dim]
        )

    def shard_full_heads(self, full: torch.Tensor) -> torch.Tensor:
        """Slice a full-head frame tensor down to this rank's head shard.

        Args:
            full: ``[S, H, Dh]`` tensor with global heads.

        Returns:
            ``[S, Hl, Dh]`` view (narrow, no copy) for this rank.
        """
        if tuple(full.shape) != (
            self.num_spatial_patches,
            self.num_heads,
            self.head_dim,
        ):
            raise ValueError(
                f"expected full-head frame shape "
                f"({self.num_spatial_patches}, {self.num_heads}, {self.head_dim}), "
                f"got {tuple(full.shape)}"
            )
        return full.narrow(1, self.local_head_start, self.local_num_heads)

    def gather_full_heads(self, locals: List[torch.Tensor]) -> torch.Tensor:
        """Reassemble full heads from per-rank shards without cat/stack hot path.

        This is a control-plane helper (not used in ``fork``/``write_block``).
        In single-process tests pass ``[rank0_tensor, rank1_tensor, ...]``.
        With a live process group and one local tensor, an all-gather is used.

        Args:
            locals: list of ``[S, Hl, Dh]`` shard tensors, rank-ordered, or a
                single-element list holding this rank's shard when distributed
                is initialized (all-gather path).

        Returns:
            ``[S, H, Dh]`` tensor with global heads.
        """
        import torch.distributed as dist  # type: ignore[import]

        if len(locals) == 1 and self.world_size > 1 and dist.is_available() and dist.is_initialized():
            local = locals[0].contiguous()
            gathered: List[torch.Tensor] = [
                torch.empty_like(local) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered, local, group=self.process_group)
            out = torch.empty(
                (self.num_spatial_patches, self.num_heads, self.head_dim),
                dtype=local.dtype,
                device=local.device,
            )
            for r, shard in enumerate(gathered):
                out.narrow(1, r * self.local_num_heads, self.local_num_heads).copy_(shard)
            return out
        if len(locals) != self.world_size:
            raise ValueError(
                f"expected {self.world_size} shards, got {len(locals)}"
            )
        out = torch.empty(
            (self.num_spatial_patches, self.num_heads, self.head_dim),
            dtype=locals[0].dtype,
            device=locals[0].device,
        )
        for r, shard in enumerate(locals):
            if tuple(shard.shape) != tuple(self.frame_shape):
                raise ValueError(f"shard {r} has bad shape {tuple(shard.shape)}")
            out.narrow(1, r * self.local_num_heads, self.local_num_heads).copy_(shard)
        return out

    # ------------------------------------------------------------------ #
    # internal allocation helpers (lock must be held unless noted)        #
    # ------------------------------------------------------------------ #

    def _alloc_pid(self) -> int:
        if not self._free:
            raise BlockExhaustedError(
                f"No free physical blocks (num_blocks={self.num_blocks})."
            )
        return self._free.popleft()

    def _new_branch_id(self) -> str:
        bid = f"branch-{self._branch_counter}"
        self._branch_counter += 1
        return bid

    def _require_branch(self, branch_id: str) -> List[int]:
        try:
            return self._branches[branch_id]
        except KeyError:
            raise UnknownBranchError(f"unknown branch {branch_id!r}") from None

    def _num_logical_blocks(self, num_frames: int) -> int:
        if num_frames <= 0:
            return 0
        bt = self.block_temporal_frames
        return (num_frames + bt - 1) // bt

    # ------------------------------------------------------------------ #
    # public API: branch lifecycle                                        #
    # ------------------------------------------------------------------ #

    def create_branch(
        self, num_frames: int, branch_id: Optional[str] = None
    ) -> str:
        """Allocate a fresh branch spanning ``num_frames`` logical frames.

        Allocates ``ceil(num_frames / Bt)`` physical blocks, sets each
        ``ref_counts[pid] = 1``, and registers the logical mapping table.

        Args:
            num_frames: logical frames addressable by the new branch.
            branch_id: optional explicit id; auto-generated if ``None``.

        Returns:
            The branch id.
        """
        if num_frames < 0:
            raise ValueError("num_frames must be >= 0")
        with self._lock:
            bid = branch_id if branch_id is not None else self._new_branch_id()
            if bid in self._branches:
                raise ValueError(f"branch {bid!r} already exists")
            table: List[int] = []
            for _ in range(self._num_logical_blocks(num_frames)):
                pid = self._alloc_pid()
                if int(self.ref_counts[pid].item()) != 0:
                    raise RuntimeError(f"allocated non-free pid {pid}")
                self.ref_counts[pid] = 1
                table.append(pid)
            self._branches[bid] = table
            return bid

    def ensure_frames(self, branch_id: str, num_frames: int) -> None:
        """Grow a branch table so ``num_frames`` frames are addressable."""
        if num_frames < 0:
            raise ValueError("num_frames must be >= 0")
        with self._lock:
            table = self._require_branch(branch_id)
            need = self._num_logical_blocks(num_frames)
            while len(table) < need:
                pid = self._alloc_pid()
                self.ref_counts[pid] = 1
                table.append(pid)

    def fork(self, parent_branch_id: str, child_branch_id: Optional[str] = None) -> str:
        """Zero-copy branch: clone the logical table, bump refcounts.

        Absolutely zero tensor memcpy: only a Python list copy plus one
        integer increment per referenced physical block.

        Args:
            parent_branch_id: existing branch to branch from.
            child_branch_id: optional explicit id for the child.

        Returns:
            The child branch id.
        """
        with self._lock:
            parent = self._require_branch(parent_branch_id)
            cid = child_branch_id if child_branch_id is not None else self._new_branch_id()
            if cid in self._branches:
                raise ValueError(f"branch {cid!r} already exists")
            # Clone ONLY the logical mapping table (list of ints).
            child: List[int] = list(parent)
            # Bump physical reference counters by 1 (shared ownership).
            for pid in child:
                self.ref_counts[pid] += 1
            self._branches[cid] = child
            return cid

    def free_branch(self, branch_id: str) -> None:
        """Release a branch: decrement refcounts, recycle blocks hitting 0."""
        with self._lock:
            table = self._require_branch(branch_id)
            for pid in table:
                rc = int(self.ref_counts[pid].item())
                if rc <= 0:
                    raise RuntimeError(f"refcount underflow for pid {pid}")
                self.ref_counts[pid] = rc - 1
                if int(self.ref_counts[pid].item()) == 0:
                    self._free.append(pid)
            del self._branches[branch_id]

    # ------------------------------------------------------------------ #
    # public API: CoW write                                               #
    # ------------------------------------------------------------------ #

    def _coerce_to_local(
        self, frame: torch.Tensor, name: str
    ) -> torch.Tensor:
        """Accept ``[S, Hl, Dh]`` (local shard) or ``[S, H, Dh]`` (full)."""
        if tuple(frame.shape) == tuple(self.frame_shape):
            return frame
        if tuple(frame.shape) == (
            self.num_spatial_patches,
            self.num_heads,
            self.head_dim,
        ):
            return frame.narrow(1, self.local_head_start, self.local_num_heads)
        raise ValueError(
            f"{name}: expected frame shape {tuple(self.frame_shape)} (local shard) "
            f"or ({self.num_spatial_patches}, {self.num_heads}, {self.head_dim}) "
            f"(full heads), got {tuple(frame.shape)}"
        )

    def write_block(
        self,
        branch_id: str,
        logical_frame_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        layer: int = 0,
    ) -> None:
        """Write one frame's K/V slice with Copy-on-Write.

        Args:
            branch_id: target branch.
            logical_frame_idx: logical temporal frame index ``f``. The owning
                block is ``table[f // Bt]``, slot ``f % Bt``.
            new_k / new_v: ``[S, Hl, Dh]`` tensors for this rank's head shard
                (or ``[S, H, Dh]`` full-head tensors, which are narrowed to
                this rank's shard).
            layer: transformer layer index.

        Behavior:
            * ``refcount == 1``: update in place via tensor slicing, no copy.
            * ``refcount > 1``: pop a free physical block, ``copy_`` the whole
              source block (all layers, K and V), decrement the source
              refcount, set the new refcount to 1, repoint this branch's table
              entry, then write the new slice.
        """
        if logical_frame_idx < 0:
            raise ValueError("logical_frame_idx must be >= 0")
        if not (0 <= layer < self.num_layers):
            raise IndexError(f"layer {layer} out of range [0, {self.num_layers})")
        k_local = self._coerce_to_local(new_k, "new_k")
        v_local = self._coerce_to_local(new_v, "new_v")
        bt = self.block_temporal_frames
        logical = logical_frame_idx // bt
        slot = logical_frame_idx % bt
        with self._lock:
            table = self._require_branch(branch_id)
            if logical >= len(table):
                raise IndexError(
                    f"frame {logical_frame_idx} (logical block {logical}) out of "
                    f"range: branch {branch_id!r} has {len(table)} logical blocks. "
                    f"Call ensure_frames first."
                )
            pid = table[logical]
            rc = int(self.ref_counts[pid].item())
            if rc <= 0:
                raise RuntimeError(f"write to free block pid={pid}")
            if rc == 1:
                # Exclusive ownership: in-place slice update. No extra copy.
                self.k_pool[layer, pid, slot].copy_(
                    k_local.to(dtype=self.dtype, device=self.device)
                )
                self.v_pool[layer, pid, slot].copy_(
                    v_local.to(dtype=self.dtype, device=self.device)
                )
                return
            # --- Shared: CoW clone path (one-block memcpy) ---
            new_pid = self._alloc_pid()
            # Copy the full source block (every layer, K and V) so pid aliasing
            # across layers stays consistent.
            for l in range(self.num_layers):
                self.k_pool[l, new_pid].copy_(self.k_pool[l, pid])
                self.v_pool[l, new_pid].copy_(self.v_pool[l, pid])
            self.ref_counts[pid] = rc - 1
            self.ref_counts[new_pid] = 1
            table[logical] = new_pid
            self.k_pool[layer, new_pid, slot].copy_(
                k_local.to(dtype=self.dtype, device=self.device)
            )
            self.v_pool[layer, new_pid, slot].copy_(
                v_local.to(dtype=self.dtype, device=self.device)
            )

    # ------------------------------------------------------------------ #
    # public API: reads / introspection                                   #
    # ------------------------------------------------------------------ #

    def read_frame(
        self, branch_id: str, logical_frame_idx: int, layer: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cloned ``(K, V)`` frame slices ``[S, Hl, Dh]`` (read-only)."""
        if logical_frame_idx < 0:
            raise ValueError("logical_frame_idx must be >= 0")
        if not (0 <= layer < self.num_layers):
            raise IndexError(f"layer {layer} out of range")
        bt = self.block_temporal_frames
        logical = logical_frame_idx // bt
        slot = logical_frame_idx % bt
        with self._lock:
            table = self._require_branch(branch_id)
            if logical >= len(table):
                raise IndexError(f"frame {logical_frame_idx} out of range")
            pid = table[logical]
            return (
                self.k_pool[layer, pid, slot].clone(),
                self.v_pool[layer, pid, slot].clone(),
            )

    def get_physical_id(self, branch_id: str, logical_block_idx: int) -> int:
        """Return the physical block index for a logical block."""
        with self._lock:
            table = self._require_branch(branch_id)
            return table[logical_block_idx]

    def branch_table(self, branch_id: str) -> List[int]:
        """Return a copy of the branch's logical->physical mapping table."""
        with self._lock:
            return list(self._require_branch(branch_id))

    def get_block_views(
        self, branch_id: str, logical_block_idx: int, layer: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return live views ``(K_block, V_block)`` of shape ``[Bt, S, Hl, Dh]``.

        These are views into the pre-allocated pool (no copy), so
        ``data_ptr()`` equality directly proves physical sharing.
        """
        if not (0 <= layer < self.num_layers):
            raise IndexError(f"layer {layer} out of range")
        with self._lock:
            table = self._require_branch(branch_id)
            pid = table[logical_block_idx]
            # Basic-int indexing on a contiguous tensor yields offset views;
            # identical pids therefore yield identical data_ptr() values.
            return self.k_pool[layer, pid], self.v_pool[layer, pid]

    def refcount(self, pid: int) -> int:
        """Return the current reference count of a physical block."""
        if not (0 <= pid < self.num_blocks):
            raise IndexError(f"pid {pid} out of range")
        return int(self.ref_counts[pid].item())

    def num_free_blocks(self) -> int:
        """Number of physical blocks with ``refcount == 0``."""
        return len(self._free)

    def num_branches(self) -> int:
        """Number of live branches."""
        return len(self._branches)

    def memory_stats(self) -> Dict[str, int]:
        """Pool occupancy summary."""
        used = self.num_blocks - len(self._free)
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": used,
            "free_blocks": len(self._free),
            "num_branches": len(self._branches),
            "block_temporal_frames": self.block_temporal_frames,
            "num_spatial_patches": self.num_spatial_patches,
            "num_heads": self.num_heads,
            "local_num_heads": self.local_num_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "world_size": self.world_size,
        }
