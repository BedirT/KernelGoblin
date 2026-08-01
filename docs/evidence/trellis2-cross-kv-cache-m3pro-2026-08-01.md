# TRELLIS.2 Exact Cross-K/V Cache Evaluation - M3 Pro - 2026-08-01

## What Is Safe To Cache

Each TRELLIS.2 cross-attention block projects K and V from the fixed DINO image
conditioning. Those tensors do not depend on the diffusion latent or timestep.
KernelGoblin can round conditioning once per positive or negative CFG branch
and retain each block's post-normalization, post-BF16 K and post-BF16 V. This
path is opt-in. The production default does not retain the cache because the
measured memory/performance trade is poor.

Self-attention K/V is deliberately not cached. Its input is the evolving latent,
so ordinary autoregressive K/V reuse would be incorrect.

## Memory And Work Removed

At 1,029 conditioning tokens, 1,536 channels, 30 blocks, and two CFG branches:

| Field | Value |
| --- | ---: |
| F32 K/V cache entries | 60 |
| Cache bytes | 758,661,120 |
| Cached sparse-flow arena peak | 1,318,781,016 bytes |
| Uncached sparse-flow arena peak | 560,119,896 bytes |
| Hisar one-step shape-flow cached peak | 1,183,823,880 bytes |
| Sparse 12-step model calls | 22 |
| Sparse cache hits / builds | 600 / 60 |
| Shape 12-step expected hits / builds | 570 / 60 |

Two single-sample debug diagnostics ran the same pinned 12-step sparse workload.
The cached flow took `99.335s`; the uncached flow took `99.573s`. The `0.238s`
difference is not an accepted benchmark result: there was no warmup, only one
sample per mode, and both runs failed the same pre-existing BF16 trajectory
quality gates described below. It is enough to reject a default that reserves
an extra 758,661,120 bytes for no established end-to-end win.

## Verification

The focused two-step real shape sampler executes 60 cache builds and 60 hits,
then executes the same run uncached and requires byte-identical final F32
latents. The 12-step sparse diagnostic asserts 60 builds and 600 hits. Cached
and uncached sparse runs printed identical per-step error metrics, decoder
metrics, occupancy count (`63,562`), and occupancy IoU (`0.6862779`). This
isolates the cache from the trajectory drift.

The byte-exact claim is deliberately scoped to that focused two-token test. The
production-sized cached and uncached diagnostics agree in every reported metric,
but they were separate processes and therefore do not establish byte equality
for all 4,096-token intermediate tensors.

The focused exact-hit check is reproducible with:

```sh
KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT=/path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  swift test --no-parallel --filter realSLatShapeSamplerGolden
```

The 12-step test uses the production uncached path by default. Set
`KG_TRELLIS2_ENABLE_CROSS_KV_CACHE=1` alongside its two checkpoint variables to
exercise and assert the opt-in 600-hit path.

The earlier one-step Hisar geometry pipeline also passed Metal API validation,
native GLB reload, stage teardown, and zero-swap process accounting:

- Wall time: `81.53s`, including a warm 0.17s release build.
- Maximum RSS: 2,671,722,496 bytes.
- Sparse flow: `10.995s`.
- Shape flow: `12.262s`.
- Output: 1,621,251 vertices and 3,481,360 faces.
- Output SHA-256: `1951b993fb470d129a9ad09b26245e116fecc1f698099c64035aace9d12f255d`.

A one-step run has one positive and one negative call, so it initializes but
does not hit either branch's cache.

## Correctness Blocker Exposed By The Evaluation

Both cached and uncached 12-step runs fail three existing structural gates after
the previously landed BF16 dense production change: step-four maximum scale
ratio is `0.301624` against `0.20`, occupancy IoU is `0.686278` against `0.70`,
and occupancy count ratio is `0.694030` against `0.70`. Tolerances were not
loosened. The cache is byte-identical in the focused hit test and behaviorally
identical in the full diagnostic, so this is a BF16 trajectory issue rather than
a K/V reuse issue.

Modern temporal feature caches can save much more only by approximating stale
diffusion features. They must be offered as an explicitly approximate quality
mode and judged on sparse geometry, thin-feature retention, topology, and final
texture, not on image-model FID alone.
