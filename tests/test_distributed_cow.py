"""CPU/single-GPU unit tests for DistributedBlockPagedKVCache CoW.

Covers the three required guarantees:
  a. Pointer equality immediately after fork.
  b. Isolation: child mutation at frame 16 leaves the parent unchanged.
  c. Refcount integrity: 1 -> 2 on fork -> 1 on child CoW mutation.

Plus: contiguity / no-dynamic-alloc invariants and mocked 2-rank head sharding
(DeepSpeed-Ulysses / Ring-Attention style) without requiring NCCL or GPUs.
"""

from __future__ import annotations

import pytest
import torch

from dailies_dag.cache.distributed_block_paged import (
    DEFAULT_BLOCK_TEMPORAL_FRAMES,
    WAN21_14B_HEAD_DIM,
    WAN21_14B_NUM_HEADS,
    DistributedBlockPagedKVCache,
)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _small_cache(
    *,
    num_blocks: int = 16,
    block_temporal_frames: int = 4,
    num_spatial_patches: int = 8,
    num_heads: int = 4,
    head_dim: int = 8,
    num_layers: int = 1,
    rank: int = 0,
    world_size: int = 1,
) -> DistributedBlockPagedKVCache:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DistributedBlockPagedKVCache(
        num_blocks=num_blocks,
        block_temporal_frames=block_temporal_frames,
        num_spatial_patches=num_spatial_patches,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype=torch.float32,
        device=device,
        rank=rank,
        world_size=world_size,
    )


def _frame(cache: DistributedBlockPagedKVCache, fill: float) -> torch.Tensor:
    return torch.full(tuple(cache.frame_shape), fill, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# spec defaults                                                               #
# --------------------------------------------------------------------------- #

def test_wan21_head_geometry_defaults() -> None:
    assert WAN21_14B_NUM_HEADS == 40
    assert WAN21_14B_HEAD_DIM == 128
    assert DEFAULT_BLOCK_TEMPORAL_FRAMES == 4
    c = DistributedBlockPagedKVCache(
        num_blocks=4, num_spatial_patches=2, num_layers=1, device="cpu"
    )
    assert c.num_heads == 40 and c.head_dim == 128
    assert c.block_temporal_frames == 4
    assert tuple(c.block_shape) == (4, 2, 40, 128)


def test_pools_preallocated_contiguous() -> None:
    c = _small_cache()
    assert c.k_pool.is_contiguous() and c.v_pool.is_contiguous()
    assert c.k_pool.shape == (1, 16, 4, 8, 4, 8)
    assert isinstance(c.ref_counts, torch.Tensor)
    assert c.ref_counts.shape == (16,)


# --------------------------------------------------------------------------- #
# (a) pointer equality immediately after fork                                 #
# --------------------------------------------------------------------------- #

def test_fork_pointer_equality_zero_memcpy() -> None:
    c = _small_cache(num_blocks=16)
    parent = c.create_branch(32)  # 8 logical blocks, Bt=4
    # Seed parent with distinctive data.
    for f in range(32):
        c.write_block(parent, f, _frame(c, float(f)), _frame(c, -float(f)))

    # Fail loudly if fork performs any tensor memcpy.
    fork_copies: list[str] = []
    orig_copy = torch.Tensor.copy_
    def _no_copy(self: torch.Tensor, *args, **kwargs):  # type: ignore[no-untyped-def]
        fork_copies.append("copy_")
        return orig_copy(self, *args, **kwargs)
    torch.Tensor.copy_ = _no_copy  # type: ignore[method-assign]
    try:
        child = c.fork(parent)
    finally:
        torch.Tensor.copy_ = orig_copy  # type: ignore[method-assign]
    assert fork_copies == [], "fork must perform zero tensor memcpy"

    # Logical tables cloned; every entry points at the same physical block.
    assert c.branch_table(child) == c.branch_table(parent)
    for logical in range(8):
        parent_k, parent_v = c.get_block_views(parent, logical)
        child_k, child_v = c.get_block_views(child, logical)
        assert child_k.data_ptr() == parent_k.data_ptr()
        assert child_v.data_ptr() == parent_v.data_ptr()
        assert c.get_physical_id(child, logical) == c.get_physical_id(parent, logical)


# --------------------------------------------------------------------------- #
# (b) isolation: child write at frame 16 leaves parent frame 16 unchanged     #
# --------------------------------------------------------------------------- #

def test_isolation_child_write_frame_16() -> None:
    c = _small_cache(num_blocks=16)
    parent = c.create_branch(32)
    for f in range(32):
        c.write_block(parent, f, _frame(c, 0.0), _frame(c, 0.0))
    child = c.fork(parent)

    c.write_block(child, 16, _frame(c, 1.0), _frame(c, 2.0))

    pk, pv = c.read_frame(parent, 16)
    ck, cv = c.read_frame(child, 16)
    assert torch.all(pk == 0.0) and torch.all(pv == 0.0), "parent mutated by child CoW write"
    assert torch.all(ck == 1.0) and torch.all(cv == 2.0)

    # Sibling frames in the same CoW-cloned block were carried over, then only
    # slot 0 (frame 16) was overwritten: frames 17..19 still read back 0.
    for f in (17, 18, 19):
        kf, _ = c.read_frame(child, f)
        assert torch.all(kf == 0.0)


# --------------------------------------------------------------------------- #
# (c) refcount integrity: 1 -> 2 on fork -> 1 on child mutation               #
# --------------------------------------------------------------------------- #

def test_refcount_integrity() -> None:
    c = _small_cache(num_blocks=16)
    parent = c.create_branch(32)
    pids_before = c.branch_table(parent)
    assert all(c.refcount(p) == 1 for p in pids_before)

    child = c.fork(parent)
    assert all(c.refcount(p) == 2 for p in c.branch_table(child))

    # Frame 16 -> logical block 4 (Bt=4), slot 0.
    old_pid = c.get_physical_id(child, 4)
    assert old_pid == c.get_physical_id(parent, 4)
    c.write_block(child, 16, _frame(c, 3.0), _frame(c, 4.0))
    new_pid = c.get_physical_id(child, 4)
    assert new_pid != old_pid, "CoW must repoint the writer's table entry"
    assert c.get_physical_id(parent, 4) == old_pid
    assert c.refcount(old_pid) == 1, "parent refcount must drop back to 1"
    assert c.refcount(new_pid) == 1
    # Untouched shared blocks stay at 2.
    for logical in [0, 1, 2, 3, 5, 6, 7]:
        assert c.refcount(c.get_physical_id(parent, logical)) == 2


def test_exclusive_write_is_inplace_no_alloc() -> None:
    c = _small_cache(num_blocks=8)
    b = c.create_branch(8)  # 2 blocks, refcount 1 each
    pid = c.get_physical_id(b, 0)
    free_before = c.num_free_blocks()
    c.write_block(b, 0, _frame(c, 5.0), _frame(c, 6.0))
    assert c.get_physical_id(b, 0) == pid
    assert c.num_free_blocks() == free_before
    k, _ = c.read_frame(b, 0)
    assert torch.all(k == 5.0)


# --------------------------------------------------------------------------- #
# mocked 2-rank distributed head sharding                                     #
# --------------------------------------------------------------------------- #

def test_mocked_two_rank_sharding_and_gather() -> None:
    """Simulate a 2-rank Ulysses group without NCCL: two cache instances."""
    kwargs = dict(
        num_blocks=16,
        block_temporal_frames=4,
        num_spatial_patches=8,
        num_heads=4,
        head_dim=8,
        num_layers=1,
    )
    r0 = _small_cache(rank=0, world_size=2, **kwargs)  # type: ignore[arg-type]
    r1 = _small_cache(rank=1, world_size=2, **kwargs)  # type: ignore[arg-type]
    assert r0.local_num_heads == 2 == r1.local_num_heads
    assert r0.local_head_range == (0, 2)
    assert r1.local_head_range == (2, 4)

    # Full-head frame narrows cleanly to each rank's shard.
    full = torch.arange(8 * 4 * 8, dtype=torch.float32).reshape(8, 4, 8)
    assert torch.equal(r0.shard_full_heads(full), full[:, 0:2, :])
    assert torch.equal(r1.shard_full_heads(full), full[:, 2:4, :])

    # Each rank caches its own shard; reassembly recovers full heads.
    b0 = r0.create_branch(8)
    b1 = r1.create_branch(8)
    r0.write_block(b0, 0, full[:, 0:2, :], full[:, 0:2, :])
    r1.write_block(b1, 0, full[:, 2:4, :], full[:, 2:4, :])
    k0, _ = r0.read_frame(b0, 0)
    k1, _ = r1.read_frame(b1, 0)
    assert k0.shape == (8, 2, 8) and k1.shape == (8, 2, 8)
    assert torch.equal(r0.gather_full_heads([k0, k1]), full)

    # CoW fork is per-rank consistent: same logical->physical aliasing.
    c0 = r0.fork(b0)
    assert r0.branch_table(c0) == r0.branch_table(b0)
    pk, _ = r0.get_block_views(b0, 0)
    ck, _ = r0.get_block_views(c0, 0)
    assert pk.data_ptr() == ck.data_ptr()


def test_sharding_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _small_cache(num_heads=5, world_size=2)


def test_dist_identity_fallback_without_init() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        pytest.skip("distributed already initialized in this env")
    c = DistributedBlockPagedKVCache(
        num_blocks=2, num_spatial_patches=2, num_heads=2, head_dim=2,
        device="cpu",
    )
    assert (c.rank, c.world_size) == (0, 1)
