# Caching TRELLIS.2 Without Fooling Ourselves

Autoregressive transformers make K/V caching look like the obvious answer. A
diffusion transformer is different: every model call receives a new latent, so
self-attention K and V change too. Reusing them would not be an optimization of
the same computation. It would be an approximation.

There is one exact exception. TRELLIS.2 cross-attention reads the same DINO
conditioning at every timestep. We implemented that cache, proved the hit path
against an uncached path, and measured it over the real 12-step sparse flow. It
was not worth enabling:

| 12-step sparse diagnostic | Uncached | Exact cross-K/V cache |
| --- | ---: | ---: |
| Flow time, one debug sample | 99.573s | 99.335s |
| Arena peak | 560,119,896 B | 1,318,781,016 B |
| Cache builds / hits | 0 / 0 | 60 / 600 |

That is 758,661,120 extra bytes for no established end-to-end speedup. The
implementation remains available as an opt-in research control, but production
defaults to the smaller uncached path. See the
[full evidence](evidence/trellis2-cross-kv-cache-m3pro-2026-08-01.md).

## What The Newer Methods Actually Do

The useful recent methods are temporal *feature* caches. They skip some or all
transformer blocks and estimate the missing result from nearby diffusion steps.
They are approximate even when a paper describes the quality as "lossless."

| Method | Basic idea | Training | Fit for our 12-step sparse 3D flow |
| --- | --- | --- | --- |
| [TeaCache](https://github.com/ali-vilab/TeaCache/tree/7c10efc4702c6b619f47805f7abe4a7a08085aa0) | Reuse a whole-model residual when a timestep-aware change score stays below a threshold | No weight training; model-specific calibration | Best first prototype. Small cache, simple boundary, but published runs are usually much longer than 12 steps. |
| [DiCache](https://arxiv.org/abs/2508.17356) | Run shallow probe blocks to decide when to cache, then align multiple cached trajectories | Training-free | Stronger long-term candidate because it adapts per sample instead of trusting fixed coefficients. More implementation and tuning work. |
| [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer/tree/704ee98c74f7f04da443daa3c0aa2cc7803d86e3) | Forecast future features from finite-difference history | Training-free | Attractive published speedups, but a 12-step run provides little history and per-block derivative storage is large. Official code is GPL-3.0, so we can study the paper but cannot copy code into this MIT repo. |
| [DuCa](https://github.com/Shenyi-Z/DuCa/tree/0a7151b7bfea3822295e8d29f26cd2e10756670c) | Alternate aggressive whole-feature reuse with conservative token refreshes | Training-free | Better operational fit than attention-map methods, but full per-block feature caches are expensive. Official code is GPL-3.0. |
| [ToCa](https://github.com/Shenyi-Z/ToCa/tree/e84096ffd85af4540a6a1f64e3334e428e1b7377) | Recompute only tokens judged important | Training-free | Poor first fit. Its 2D spatial score and attention-map access do not transfer cleanly to irregular sparse 3D tokens. Thin geometry is easy to erase. |
| [TokenCache](https://arxiv.org/abs/2409.18523) | Learn which tokens, blocks, and timesteps can be reused | Predictor training plus LoRA | A separately trained model variant, not a faithful runtime port. Its 10-step DiT result is also much less dramatic than longer-step claims. |
| [DiTFastAttn](https://github.com/thu-nics/DiTFastAttn/tree/91bfbaee962d839c433e68c6fab1aeaa78c30c28) | Calibrated sparse/window attention plus attention reuse across timesteps and CFG | Offline calibration | Needs a real 3D sparse-window design. Published end-to-end gains show that shrinking attention alone can leave the MLP bottleneck untouched. |

[Cache-DiT](https://github.com/vipshop/cache-dit) is also a useful engineering
reference: it packages several of these ideas behind block adapters. It is a
PyTorch/CUDA-oriented engine, not something we can install into the native
Swift runtime, but its adapter boundary is close to what we need.

## Our Order Of Attack

1. Restore the exact 12-step quality gate after the BF16 dense optimization.
   Speed numbers do not count while the pinned geometry contract is red.
2. Add a `quality` profile that always runs all blocks. This remains the
   reference and the default until an approximate profile earns promotion.
3. Prototype TeaCache at the complete 30-block flow boundary, with independent
   state for sparse, shape, texture, positive CFG, and negative CFG.
4. Calibrate with TRELLIS.2 trajectories, not coefficients borrowed from FLUX
   or a video model. Twelve steps means forced refreshes consume a much larger
   share of the schedule.
5. Promote only after fixed-seed comparisons cover occupancy IoU, component
   count, mesh topology, thin parts such as bows and limbs, UV validity, and PBR
   texture metrics. A pretty turntable is evidence too, but not the only gate.
6. If TeaCache cannot skip enough work safely, move to a shallow-probe DiCache
   design. TaylorSeer and token-wise methods stay behind that until their memory
   and sparse-3D semantics make sense on Apple Silicon.

The target remains a useful generation under two minutes. The important part is
that `fast` will be an honest, named quality/performance trade, not a silent
change to what "TRELLIS.2 compatible" means.
