# TRELLIS.2 BF16 MPSGraph Attention Evidence - M3 Pro - 2026-08-01

## Why This Path Exists

TRELLIS.2 stores and executes this transformer in BF16 upstream. KernelGoblin
already rounded attention inputs and outputs at the same model boundaries, but
its first MPSGraph SDPA path widened the internal graph to F32. The revised
graph casts the already BF16-exact inputs back to BF16, runs SDPA in BF16, and
returns an F32 buffer containing BF16-rounded values for the surrounding native
runtime.

This is a precision-boundary correction and a performance optimization. It is
not K/V quantization, stale feature reuse, reduced sampling, or a relaxed model
gate.

## Exact-Shape Benchmark

```sh
./kg model native-attention-benchmark trellis2 --warmup 2 --iterations 5
```

| Workload | Custom Metal | BF16 MPSGraph | Speedup | Metal comparison NRMS |
| --- | ---: | ---: | ---: | ---: |
| Self `B1 H12 Q4096 K4096 D128` | 767.699 ms | 31.393 ms | 24.455x | `1.68e-3` |
| Cross `B1 H12 Q4096 K1029 D128` | 130.744 ms | 7.725 ms | 16.925x | `1.65e-3` |
| Cross `B1 H12 Q1024 K1029 D128` | 33.254 ms | 2.663 ms | 12.485x | `1.65e-3` |

The benchmark validates before timing. The custom Metal output is sampled
against the F32 CPU softmax with the original `2e-5` absolute gate. The BF16
graph uses a separate `4e-5` absolute gate derived for its lower-precision
reduction and output rounding. The full comparison also retains the existing
`0.005` normalized-RMS gate, and the unchanged pinned upstream fixture remains
the acceptance authority. Maximum RSS was 278,708,224 bytes, peak footprint
was 405,783,344 bytes, and swaps were zero.

Against the previous F32 MPSGraph graph, the same self-attention workload moved
from 132.285 ms to 31.393 ms, a 4.21x reduction.

## Pinned Upstream And Real-Checkpoint Gates

The unchanged 4,096-token fixture generated from the pinned upstream MPS SDPA
passed with normalized RMS `0.004902` and maximum scale ratio `0.00535`.

The complete one-step sparse sampler then ran all 30 blocks and both CFG calls
against the real checkpoint. It passed the existing call and final-output gates:

- Positive call normalized RMS: `0.009192`; maximum scale ratio: `0.02629`.
- Negative call normalized RMS: `0.007665`; maximum scale ratio: `0.02773`.
- Final RMS: `0.017871`; maximum absolute error: `0.08188`.
- Inference timing boundary: `10.788s`, down from `16.054s` after SIMD-group
  normalization.
- Arena peak: `550,400,008` bytes, unchanged.

## Real Geometry Integration

The Hisar one-step geometry run passed native GLB reload validation under Metal
API validation. Sparse flow moved from 20.982s to 15.393s and shape flow from
16.364s to 14.235s. The output contains 1,625,268 vertices and 3,530,558 faces;
maximum RSS was 2,671,181,824 bytes and swaps remained zero.

The measured process wall time was 98.97s, but that includes a 7.53s release
rebuild and is therefore not compared directly with the previous warm-build
97.60s result. The stage evidence is the stable integration comparison.

The output hash changed because BF16 SDPA changes reduction and rounding order.
It was accepted only after the pinned attention and real-checkpoint sampler gates
passed. This improvement still does not prove the 12-step two-minute target.
