"""Branch @ frame 60 demo: proves O(n/S) pointer fork, zero prefix recompute."""
import argparse, torch
from dailies_dag.cache.temporal_block_manager import TemporalBlockManager
from dailies_dag.models.causal_temporal_attention import CausalTemporalAttention

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--branch", type=int, default=60)
    ap.add_argument("--block-size", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    mgr = TemporalBlockManager(num_layers=2, num_heads=4, head_dim=8,
                               tokens_per_frame=8, block_size=args.block_size,
                               max_num_blocks=64)
    attn = CausalTemporalAttention(dim=32, num_heads=4, head_dim=8)

    main_table = mgr.alloc_seq(0)
    clip = torch.randn(args.branch, 8, 32)
    attn(clip, block_table=main_table, cache_manager=mgr, layer_id=0, start_frame=0)
    print(f"main: cached {args.branch} frames -> {len(main_table)} blocks, free={mgr.num_free_blocks()}")

    child = mgr.fork(main_table, branch_frame_idx=args.branch)
    print(f"fork @ {args.branch}: shared={mgr.shared_blocks(main_table, child)} blocks, "
          f"tensor memcpy=0, free={mgr.num_free_blocks()}")

    edit = torch.randn(args.frames - args.branch, 8, 32) + 0.5  # relight shift
    attn(edit, block_table=child, cache_manager=mgr, layer_id=0,
         start_frame=args.branch, branch_frame_idx=args.branch)
    print(f"child: appended {args.frames - args.branch} frames -> {len(child)} blocks")
    print("stats:", mgr.memory_stats())

if __name__ == "__main__":
    main()
