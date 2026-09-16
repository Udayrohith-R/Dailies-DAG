import torch
from dailies_dag.cache.temporal_block_manager import TemporalBlockManager
from dailies_dag.models.causal_temporal_attention import CausalTemporalAttention


def _mgr(S=8, T=120, layers=1):
    m = TemporalBlockManager(num_layers=layers, num_heads=2, head_dim=4,
                             tokens_per_frame=4, block_size=S, max_num_blocks=64)
    return m


def test_fork_shares_pointers_not_memory():
    m = _mgr()
    a = m.alloc_seq(120)
    n_prefix = 60 // 8 + (1 if 60 % 8 else 0)
    b = m.fork(a, branch_frame_idx=60)
    assert m.shared_blocks(a, b) == 8  # ceil(60/8)
    assert all(m.refcount(0, pid) == 2 for pid in b[:8])
    assert m.num_free_blocks() == 64 - 15  # only parent's 15 blocks used; fork added 0


def test_cow_clone_on_write():
    m = _mgr()
    a = m.alloc_seq(16)
    L, H, Dh = 4, 2, 4
    k = torch.ones(L, H, Dh)
    v = torch.ones(L, H, Dh)
    m.write_frame(a, 0, k, v, layer=0)
    b = m.fork(a, branch_frame_idx=8)
    old_pid = b[0]
    m.write_frame(b, 0, torch.zeros(L, H, Dh), torch.zeros(L, H, Dh), layer=0)
    assert b[0] != old_pid  # CoW repointed
    assert b[0] != a[0]
    K_a, _ = m.read_sequence(a, 1, layer=0)
    assert K_a[0].abs().sum() > 0  # parent untouched


def test_attention_accepts_block_table():
    torch.manual_seed(0)
    m = TemporalBlockManager(num_layers=1, num_heads=2, head_dim=4,
                             tokens_per_frame=4, block_size=8, max_num_blocks=32)
    attn = CausalTemporalAttention(dim=16, num_heads=2, head_dim=4)
    # temporal dim must satisfy L*H*Dh == dim? project via to_qkv handles any dim.
    table = m.alloc_seq(0)
    h = torch.randn(4, 4, 16)
    out = attn(h, block_table=table, cache_manager=m, layer_id=0, start_frame=0)
    assert out.shape == (4, 4, 16)
    # branch at frame 2, append 2 more frames on child — prefix reused
    child = m.fork(table, branch_frame_idx=2)
    h2 = torch.randn(2, 4, 16)
    out2 = attn(h2, block_table=child, cache_manager=m, layer_id=0,
                start_frame=2, branch_frame_idx=2)
    assert out2.shape == (2, 4, 16)
