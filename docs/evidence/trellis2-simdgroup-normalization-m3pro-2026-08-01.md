# TRELLIS.2 SIMD-Group Normalization Evidence - M3 Pro - 2026-08-01

The original LayerNorm and multi-head RMSNorm kernels assigned one GPU thread
to an entire 1,536-channel row or 128-wide attention head. The optimized path
uses one 32-lane SIMD group, parallel partial sums, `simd_sum`, and strided
output writes. Small and unsupported shapes retain the scalar kernels.

```sh
./kg model native-normalization-benchmark trellis2 \
  --warmup 2 --iterations 5
```

| Workload | Scalar median | SIMD-group median | Speedup | Normalized RMS |
| --- | ---: | ---: | ---: | ---: |
| LayerNorm `4096x1536` | 1.958 ms | 1.018 ms | 1.923x | `3.12e-7` |
| RMSNorm `4096x12x128` | 1.027 ms | 0.576 ms | 1.781x | `1.09e-7` |

The benchmark checks every output element before timing. Maximum absolute
differences were `2.51e-6` for LayerNorm and `8.35e-7` for RMSNorm.

The pinned one-step 4,096-token, 30-block sparse sampler also passed its real
checkpoint gates. Its inference timing boundary moved from `16.60s` before the
SIMD-group path to `16.05s`; the arena peak stayed `550,400,008` bytes. This is
an incremental end-to-end improvement, not evidence that normalization is the
remaining dominant bottleneck.
