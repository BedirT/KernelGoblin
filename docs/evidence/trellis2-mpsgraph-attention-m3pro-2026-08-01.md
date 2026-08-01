# TRELLIS.2 MPSGraph Attention Evidence - M3 Pro - 2026-08-01

## What This Proves

This record covers Apple MPSGraph scaled dot-product attention on the exact
TRELLIS.2 head shape. The optimized path is selected for one large 128-wide
segment on macOS 15 or newer. macOS 14, small inputs, and segmented batches
retain the custom Metal online-softmax implementation.

- Device: Apple M3 Pro
- OS: macOS 26.5.2 (25F84)
- Layout at the model boundary: contiguous `[B,N,H,D]`
- MPSGraph layout: explicit transpose to and from `[B,H,N,D]`
- Timing: dispatch through synchronized completion; graph construction, first
  compilation, allocation, and correctness validation excluded

## Exact-Shape Benchmark

```sh
./kg model native-attention-benchmark trellis2 --warmup 1 --iterations 3
```

| Workload | Custom Metal median | MPSGraph median | Speedup | Normalized RMS |
| --- | ---: | ---: | ---: | ---: |
| Self `B1 H12 Q4096 K4096 D128` | 803.669 ms | 132.285 ms | 6.075x | `1.94e-6` |
| Cross `B1 H12 Q4096 K1029 D128` | 140.610 ms | 25.679 ms | 5.476x | `8.42e-7` |
| Cross `B1 H12 Q1024 K1029 D128` | 35.626 ms | 9.879 ms | 3.606x | `1.05e-6` |

The benchmark checks every output element against the custom Metal kernel and
27 sampled outputs against a stable CPU softmax. The largest sampled CPU
absolute error was `3.68e-8`. A separate run recorded about 348 MiB maximum RSS,
446 MiB peak footprint, and zero swaps.

## Real-Checkpoint Conformance

The pinned one-step 4,096-token sparse sampler ran all 30 blocks and both CFG
passes in `16.60s` inside the test timing boundary. Both call traces remained
within the existing Torch/MPS-derived gates:

- Positive normalized RMS: `0.009257`; maximum scale ratio: `0.02666`.
- Negative normalized RMS: `0.007672`; maximum scale ratio: `0.02688`.
- Final RMS: `0.018001`; maximum absolute error: `0.08965`.
- Arena peak: `550,400,008` bytes, unchanged from the accepted custom path.

## Real Model Integration

The same Hisar one-step geometry contract used for the dense baseline completed
in `97.60s`, down from `188.16s` before MPSGraph optimization and `166.00s` with
dense optimization alone.

| Stage | Before MPSGraph | Dense only | Dense + SDPA |
| --- | ---: | ---: | ---: |
| Sparse flow | 80.362s | 65.429s | 20.982s |
| Shape flow | 51.513s | 43.171s | 16.364s |
| Total process | 188.16s | 166.00s | 97.60s |

The generated GLB reloaded successfully with 1,678,203 vertices and 3,692,356
faces. Maximum RSS was `2,665,201,664` bytes and swaps remained zero.

The artifact is not bit-identical to the custom-attention run because the
parallel softmax reduction order differs. That difference is expected and was
accepted only after the exact-shape, CPU-sampled, pinned MPS fixture, and full
real-checkpoint gates above passed. This result does not yet prove the 12-step
two-minute target; no 12-step rerun was performed.
