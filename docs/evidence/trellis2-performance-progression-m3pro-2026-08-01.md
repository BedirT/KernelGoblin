# TRELLIS.2 Native Performance Progression - M3 Pro - 2026-08-01

## The Short Version

The authenticated 12-step sparse-structure flow moved from `840.812s`
(`14m 0.8s`) on the first conformant F32 route to a current three-run range of
`118.888s` to `139.654s`. The median diagnostic is `125.813s` (`2m 5.8s`), a
`6.68x` reduction against that recorded baseline.

These are synchronized real-checkpoint diagnostics on the same Apple M3 Pro,
not a controlled benchmark series. The current complete trajectory-and-decoder
test takes `133.690s` to `151.066s`; its latest run was `140.698s`. Do not compare
that complete-test number directly with the flow-only `840.812s` baseline.

An earlier conversational estimate or observed run around 22 minutes was not
captured with a durable timing boundary. We therefore use the slower port's
authenticated `840.812s` flow measurement as the accepted baseline instead of
turning recollection into evidence.

## What Was Actually Slow

The runtime was dispatching the Apple GPU, but the first custom kernels did not
map the dominant matrix operations to the hardware efficiently:

1. The custom F32 attention kernel assigned work per query and head, then
   streamed every key through an online softmax. At the production self-attention
   shape, one call took a `803.669ms` median. TRELLIS.2 performs self-attention
   and cross-attention in every one of 30 blocks, across 22 CFG model calls.
2. The first dense path used tiled or SIMD-group Metal kernels. The production
   `4096 x 1536 -> 8192` MLP-up projection took `216.315ms` on the tiled path.
   This projection also repeats in every block and model call.
3. LayerNorm and RMSNorm initially put an entire row or head on one GPU thread.
   This left SIMD lanes idle, although normalization was a secondary bottleneck.
4. The CPU was mainly orchestrating synchronized dispatches. The problem was
   not that the model ran on the CPU or that unified memory itself was slow; it
   was poor GPU work decomposition and excessive cost in repeatedly executed
   kernels.

## What Changed

| Change | Exact-shape result | Model consequence |
| --- | --- | --- |
| MPSGraph dense routing | MLP up `216.315ms -> 28.524ms`, `7.584x` | Removed the largest dense bottleneck while retaining custom Metal fallbacks |
| MPSGraph SDPA | Self attention `803.669ms -> 132.285ms`, `6.075x`; cross attention `140.610ms -> 25.679ms`, `5.476x` | Replaced the dominant custom online-softmax path with Apple's optimized GPU graph |
| SIMD-group normalization | LayerNorm `1.958ms -> 1.018ms`; RMSNorm `1.027ms -> 0.576ms` | Parallelized reductions across 32 SIMD lanes, with scalar fallbacks |
| BF16 model boundaries | Faster dense projections and lower temporary pressure | Kept where the full trajectory remained conformant |
| Selective F32 accumulation | F32 SDPA and final Euler velocity projection | Recovered the unchanged 12-step geometry gates after an all-BF16 route drifted too far |

The one-step full geometry integration shows the same progression under one
stable command boundary:

| Production route | Total process | Sparse flow | Shape flow |
| --- | ---: | ---: | ---: |
| Original custom Metal | `188.16s` | `80.362s` | `51.513s` |
| MPSGraph dense only | `166.00s` | `65.429s` | `43.171s` |
| MPSGraph dense + SDPA | `97.60s` | `20.982s` | `16.364s` |

This table is one-step geometry generation, not the current 12-step sparse-only
trajectory test. It exists to attribute the speedup without mixing timing
boundaries.

## Why We Did Not Keep The Fastest Precision Route

BF16 MPSGraph attention and dense both passed microbenchmarks, complete blocks,
and one-step integration. Together they reduced a 12-step sparse diagnostic to
about `99.573s`, but failed the immutable free-running geometry gates:

- Step-four maximum scale ratio: `0.301624`, limit `0.20`.
- Occupancy IoU: `0.686278`, minimum `0.70`.
- Occupancy count ratio: `0.694030`, minimum `0.70`.

The accepted mixed-precision route keeps BF16 dense projections, uses F32 SDPA,
and computes the final Euler velocity projection in F32. It reaches `0.718923`
occupancy IoU and `0.739343` count ratio without changing any tolerance. The
extra 20 to 40 seconds versus the rejected route buys a trajectory that still
meets the pinned TRELLIS.2 contract.

## What The Number Does Not Mean

- `140.698s` is the latest sparse trajectory plus sparse decoder test, not full
  image-to-PBR generation.
- The complete native pipeline also runs DINO conditioning, shape flow, shape
  decoding, mesh extraction, UV preparation, texture flow, texture decoding,
  PBR baking, and GLB export depending on the selected mode.
- Three noisy diagnostics are enough to show a large engineering improvement,
  but not enough for a publication-quality benchmark claim.
- The sub-two-minute end-to-end goal has not been reached yet. Temporal feature
  caching and a faster F32-accumulating attention path remain the next targets.

## Related Evidence

- [MPSGraph dense](trellis2-mpsgraph-dense-m3pro-2026-08-01.md)
- [MPSGraph attention](trellis2-mpsgraph-attention-m3pro-2026-08-01.md)
- [SIMD-group normalization](trellis2-simdgroup-normalization-m3pro-2026-08-01.md)
- [Mixed-precision 12-step trajectory](trellis2-mixed-precision-trajectory-m3pro-2026-08-01.md)
- [Cache acceleration decision](../TRELLIS2_CACHE_ACCELERATION.md)
