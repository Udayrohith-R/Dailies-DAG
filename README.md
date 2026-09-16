# Dailies-DAG — Zero-Latency Branching Edits for Video DiTs

> Built for Virtual Production and continuous editing. Target reader: creative-tech
> researchers evaluating whether diffusion-based relighting can hold cinematic
> continuity at interactive latency.

## 0. The problem: hardware starvation per edit

A standard video DiT re-encodes the full spatio-temporal context on every edit.
For a clip with T frames of H \times W latents, each self-attention layer
costs O((T \times H \times W)^2) — and a single slider nudge on frame 60
pays that price across all 40 Wan2.1-14B blocks before the artist sees a pixel.
The GPUs are not computing anything new. They are recomputing everything old.
That is hardware starvation, and it is what kills interactive dailies review.

**Dailies-DAG** is a system–algorithmic co-design that reduces a branch edit to
O(n/S) pointer copies, where n is the cached prefix length and S the
temporal block size. Forking a 96-frame 720p prefix across
B_t = 4-frame physical blocks copies **24 integers and zero tensors**.
Only blocks the edit actually touches are ever cloned (Copy-on-Write); only
frames at or after the branch point are ever recomputed (differential
prefill). Everything else is served from a pre-allocated, rank-sharded block
pool. No recompute of unchanged geometry. Ever.

```
frames 0..59 ──► shared physical blocks (refcount=2, pointers only)
frame 60 ──────► CoW fork: new tail blocks, Triton steer vector, RL-guarded
```

## 1. Architecture: the 3 pillars

### Pillar 1 — `DistributedBlockPagedKVCache` (temporal branching, zero latency)

`dailies_dag/cache/distributed_block_paged.py`

Video tokens are 3D spatio-temporal patches, so physical blocks are
`[B_t=4, S, H_l, D_h]` K/V tiles — the temporal analogue of vLLM's paged
tokens, frame-aligned so edits stay CoW-clean. The pools (`k_pool`/`v_pool`,
`[L, N, B_t, S, H_l, D_h]`) are allocated **once**, contiguous, and never
resized: the hot paths (`fork` / `write_block`) use only slicing + `copy_`,
never `cat`/`stack`/`empty`. Ownership is an explicit per-block
`ref_counts: torch.Tensor`.

- `fork(parent) -> child`: clones the logical table (`List[int]`), bumps
refcounts. **Zero memcpy** — enforced by test (fails on any `copy_`).
- `write_block(branch, frame, k, v)`: `refcount == 1` → in-place slice;
`refcount > 1` → pop one free block, copy one block, repoint one entry.
- **Distributed**: heads sharded across sequence-parallel ranks
(DeepSpeed-Ulysses / Ring-Attention compatible; rank `r` owns
`[r·H_l, (r+1)·H_l)`), rank/world auto-detected from `torch.distributed`.
Shard-local pools keep the design FSDP-friendly — no replicated cache state
beyond the owned head slice. Defaults match Wan2.1-14B geometry
(40 heads × 128 dim, 40 layers); tests run shrunken shapes.
- `CausalPagedAttentionWrapper` (`dailies_dag/models/wan_temporal_interceptor.py`)
monkey-patches `WanTransformer3DModel` attention in place via
`install_interceptor(pipeline, cache)`: prefix frames bypass Q-projection
and attention entirely (K/V read from the pool), dirty frames project →
commit via `write_block` → attend over the combined context. Timestep
(`temb`) embeddings pass through untouched; all matmul/softmax runs in
fp32 with max-subtraction for bf16/fp8 stability.

### Pillar 2 — Mechanistic latent steering (deterministic sliders, no prompts)

`dailies_dag/steering/triton_steer_kernel.py` · `dailies_dag/steering/hook.py`

Text prompts are a lottery for lighting continuity. Dailies-DAG steers
mechanistically: a learned 1-D direction `V` (e.g. *luminance*) is added
in place — `X += α·V` over `[B, T·S, D]` — through the custom
`_fused_steer_kernel` (2-D grid over `(row, hidden-tile)`, `BLOCK` of
1024/2048, `num_warps=8` for Hopper/Ada bandwidth saturation, `α` as a
runtime scalar so strength sweeps never recompile; exact `torch.no_grad`
fallback on CPU). `install_steering_pre_hook(pipeline, V, α)` plants the edit
as a forward pre-hook on every block's attention input — **before `to_qkv`** —
with live `set_alpha()` and optional temporal scoping (`frame_range`) so only
the dirty branch slice is relit. Deterministic, slider-quantized, reproducible
frame-to-frame: the same `α` is the same photons.

### Pillar 3 — Stage-relative GRPO guardrails (geometry must not melt)

`dailies_dag/rl/srpo_guard.py`

Steering without constraints warps faces and melts backgrounds. Every rollout
branch is scored by two frozen geometric judges (never any grad graph):
**DepthAnythingV2** (monocular depth consistency,
`L_depth = MAE(depth_orig, depth_edit)`) and **RAFT** via torchvision
(`L_flow = MAE(flow_orig, flow_edit)`), with deterministic analytic proxies
when weights are offline. `grpo_step` computes group-relative advantages
`A_i = (R_i − mean(R)) / (std(R) + ε)` over `G` branches for
`R = w_edit·R_feat − λ_depth·L_depth − λ_flow·L_flow`, with vectorized
Pareto filtering (`np.all`/`np.any` broadcasting) plus a hard gate —
**any trajectory with depth drift past `τ` is discarded regardless of its
feature score**. Penalties are normalized by the flow-matching timestep:
early denoising steps (structure formation) penalize deviation up to `1+boost`
× harder than final refinement steps. The policy learns relighting that the
geometry judges cannot detect.

## 2. The pipeline (usage)

Prerequisites: CUDA host with the Wan2.1-14B Diffusers checkpoint,
`pip install -e ".[triton,orchestration,training]"` plus
`diffusers`, `bitsandbytes`, `ffmpeg-python`, `opencv-python`.

```bash
# --- Stage 0: 4K ProRes -> exact 96-frame 720p proxy + one-time VAE/text cache ---
python -m dailies_dag.scripts.train_overfit_lora \
  --clip /data/cine_4k_prores.mp4 \
  --out ./overfit_run \
  --prompt "cinematic dolly shot, golden hour street, photoreal" \
  --start-sec 0.0 --steps 800 --lr 1e-5
# Flags: --encode-only  (ingest + VAE cache, then exit)
#        --train-only   (resume training from latents_cached.pt)
# Overfit contract: LoRA r=128/α=128 on to_q/k/v/out, flow-matching velocity
# MSE on the SAME clip every step (no augmentation), AdamW8bit + grad checkpointing.
# Adapter lands at ./overfit_run/lora_final.

# --- Stage 1: branch + steer (Python; zero-latency fork, slider relight) ---
python - <<'EOF'
import torch
from dailies_dag.cache import DistributedBlockPagedKVCache
from dailies_dag.models import install_interceptor
from dailies_dag.steering import install_steering_pre_hook

cache = DistributedBlockPagedKVCache(
    num_blocks=256, num_spatial_patches=11520, num_layers=40, device="cuda")
handle = install_interceptor(pipeline, cache)          # Wan attentions -> paged CoW
child = cache.fork("main")                             # O(n/S) pointers, zero memcpy
luminance_V = torch.load("vectors/luminance.pt")       # [5120] learned direction
steer = install_steering_pre_hook(
    pipeline, luminance_V, alpha=0.4,
    frame_range=(60, 96), num_spatial_patches=11520)    # relight dirty slice only
denoise_branch(pipeline, branch_id=child, branch_frame_t=60)  # prefix served from pool
steer.set_alpha(0.55)                                  # live slider, no re-hook
EOF

# --- Stage 2: geometric proof (side-by-side + depth-MAE HUD, 24fps) ---
python -m dailies_dag.scripts.render_visual_proof \
  --original original_720p.mp4 \
  --relit relit_branch_720p.mp4 \
  --out netflix_demo_final.mp4 --fps 24
# HUD: "DAG Branch: Active | Feature: Luminance (+0.4) | Depth MAE: 0.0123 (PASS)"
# green PASS < 0.05, red WARN >= 0.05, per-frame + mean/max printed to stdout.
```

## 3. Layout

```
dailies_dag/
  cache/
    distributed_block_paged.py  # Pillar 1: sharded paged KV + CoW (CORE)
    temporal_block_manager.py   # single-process paged KV predecessor
  models/
    wan_temporal_interceptor.py # CausalPagedAttentionWrapper + install_interceptor
    causal_temporal_attention.py# reference paged attention block
  steering/
    triton_steer_kernel.py      # Pillar 2: _fused_steer_kernel + torch fallback
    hook.py                     # install_steering_pre_hook (pre-to_qkv)
  rl/
    srpo_guard.py               # Pillar 3: depth/flow judges + stage-relative GRPO
  scripts/
    train_overfit_lora.py       # 4K ingest -> VAE cache -> LoRA overfit
    render_visual_proof.py      # hstack + depth-MAE HUD -> netflix_demo_final.mp4
tests/
  test_distributed_cow.py       # pointer equality / isolation / refcounts / sharding
  test_temporal_block_manager.py
examples/
  branch_edit.py                # O(n/S) fork demo (CPU, no weights)
```

## 4. Quickstart (CPU, no weights needed)

```bash
pip install -e ".[test]"
pytest tests/ -v
python examples/branch_edit.py --frames 120 --branch 60 --block-size 8
```

## 5. The guarantees (tested, not promised)

- **Fork**: `child.data_ptr() == parent.data_ptr()` on every shared block;
refcounts `1 → 2` on fork, `→ 1` on first CoW write
(`tests/test_distributed_cow.py`).
- **Isolation**: mutating child frame 16 leaves parent frame 16 bit-identical.
- **Steering**: attention observes exactly `X + α·V`, pre-`to_qkv`, grad-free,
temporally scoped.
- **Guardrails**: tau-violating trajectories vetoed before advantages;
advantages group-normalized (mean 0, std 1); all-discarded groups emit
zero loss, never NaN gradients.

