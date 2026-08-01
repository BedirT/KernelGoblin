# TRELLIS.2 BF16 MPSGraph Dense Evidence - M3 Pro - 2026-08-01

## Scope

The first MPSGraph dense path widened BF16 checkpoint weights and already
BF16-rounded activations to F32 inside the graph. This revision preserves the
upstream BF16 matrix boundary and returns F32 storage to the surrounding native
runtime. Small row-one projections still use custom Metal because graph dispatch
costs more than it saves.

## Benchmark

```sh
./kg model native-benchmark trellis2 --warmup 2 --iterations 5
```

| Workload | Tiled Metal | BF16 MPSGraph | Speedup | MPSGraph NRMS |
| --- | ---: | ---: | ---: | ---: |
| Sparse input `4096x8 -> 1536` | 1.789 ms | 0.739 ms | 2.420x | `2.95e-4` |
| Cross K/V `1029x1024 -> 3072` | 13.866 ms | 2.082 ms | 6.661x | `1.36e-3` |
| MLP up `4096x1536 -> 8192` | 216.079 ms | 24.492 ms | 8.823x | `1.39e-3` |

The benchmark validates before timing. Custom Metal retains its F32 CPU
accumulation bound. The BF16 graph is checked independently with maximum
absolute error and normalized RMS limits of `0.002`, derived for the lower
precision matrix boundary. Observed maximum absolute error was `0.001680`.
Maximum RSS was 660,439,040 bytes, peak footprint was 784,270,416 bytes, and
swaps were zero.

Against the earlier F32 MPSGraph medians, BF16 moved sparse input from 1.033 ms
to 0.739 ms, cross K/V from 2.454 ms to 2.082 ms, and MLP up from 28.524 ms to
24.492 ms.

## Real Checkpoint Gate

The complete one-step, 4,096-token sparse sampler ran all 30 blocks and both CFG
calls against the real checkpoint. The existing Torch/MPS-derived call and final
output gates passed:

- Positive normalized RMS: `0.009921`; maximum scale ratio: `0.02551`.
- Negative normalized RMS: `0.008495`; maximum scale ratio: `0.02652`.
- Final RMS: `0.019059`; maximum absolute error: `0.09214`.
- Inference timing boundary: `10.109s`, down from `10.602s` with BF16 SDPA alone.
- Arena peak: `550,400,008` bytes, unchanged.

This is a measured 4.7% sampler improvement. It is worthwhile and more faithful
to upstream precision, but it does not by itself close the 12-step latency gap.
