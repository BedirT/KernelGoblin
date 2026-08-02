# TRELLIS.2 Mixed-Precision Trajectory Evidence - M3 Pro - 2026-08-01

## Why This Exists

The first BF16 MPSGraph production route passed dense, attention, complete-block,
and one-step sampler checks, but failed after 22 feedback calls in the pinned
12-step sparse trajectory. We kept the oracle and every tolerance unchanged and
isolated dense and attention precision independently.

All runs used the same authenticated Torch/MPS fixture, sparse-flow checkpoint,
decoder checkpoint, 4,096 latent tokens, 1,029 conditioning tokens, 12 Euler
steps, and 22 model calls. Cross-K/V caching was disabled.

## Diagnostic Matrix

These are single-sample debug diagnostics, not throughput benchmarks. Their
purpose is precision attribution. Timings include the complete sparse flow and
are reported only to reveal the cost direction.

| Dense route | Self SDPA | Cross SDPA | Flow | Step-4 max ratio | Occupancy IoU | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BF16 automatic | BF16 | BF16 | 99.573s | 0.301624 | 0.686278 | Fail |
| F32 terminals + input | BF16 | BF16 | 104.352s | 0.218788 | 0.700967 | Fail |
| All MPSGraph F32 | BF16 | BF16 | 108.420s | 0.218788 | 0.700967 | Fail |
| F32 terminals + input | F32 | F32 | 122.971s | 0.185057 | 0.726330 | Pass |
| F32 terminals + input | F32 | BF16 | 121.495s | 0.227098 | 0.702941 | Fail |
| F32 terminals + input | BF16 | F32 | 114.553s | 0.235225 | 0.693456 | Fail |
| Final flow output F32 only | F32 | F32 | 118.888s | 0.182033 | 0.718923 | **Pass** |
| All dense BF16 | F32 | F32 | 132.426s | 0.212441 | 0.698326 | Fail |

The elapsed values are noisy: the final all-BF16 row was slower than several
F32 rows despite doing less nominal work. We do not use these single samples as
speed claims. The deterministic tensor and occupancy gates are the evidence.

## Selected Production Precision

- Dense projections remain model-precision BF16 by default.
- Self-attention and cross-attention prefer MPSGraph SDPA with F32 inputs and
  outputs, with a native F32 Metal fallback where MPSGraph is unavailable.
- The final flow output projection prefers F32 MPSGraph accumulation because it
  becomes the Euler velocity without a later BF16 rounding boundary. It also
  retains the native F32 Metal fallback.
- Tiny single-row projections continue to use the native Metal path.

The selected route passed all unchanged gates:

| Gate | Limit | Result |
| --- | ---: | ---: |
| Steps 0-4 normalized RMS | <= 0.10 | 0.062064 maximum |
| Steps 0-4 maximum scale ratio | <= 0.20 | 0.182033 maximum |
| Final normalized RMS | structural diagnostic | 0.515869 |
| Occupancy IoU | >= 0.70 | 0.718923 |
| Occupancy count ratio | 0.70...1.30 | 0.739343 |
| Decoded coordinates | nonempty | pass |
| Sparse arena peak | <= 600 MiB | 560,119,896 bytes |
| Swaps | 0 | 0 |

Three final-route diagnostics reported `118.888s`, `139.654s`, and `125.813s`
for the sparse flow, with complete-test times of `133.690s`, `151.066s`, and
`140.698s`. The spread is why these measurements are not presented as a
benchmark or speed claim. This is a correctness recovery, not the sub-two-minute
end-to-end target. The next exact attention work must retain F32 accumulation
while reducing the SDPA and dispatch cost.

## Reproduction

```sh
KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT=/path/to/ss_flow_img_dit_1_3B_64_bf16.safetensors \
KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT=/path/to/ss_dec_conv3d_16l8_fp16.safetensors \
  swift test --no-parallel \
  --filter realSparseStructureFullTrajectoryAndHandoffGolden
```
