# TRELLIS.2 MPSGraph Dense Evidence - M3 Pro - 2026-08-01

## Scope

This record verifies the optional Apple MPSGraph backend for large BF16-weight,
F32-output dense projections. It is not an attention benchmark and it does not
claim the two-minute 12-step generation target has been reached.

- Device: Apple M3 Pro
- OS: macOS 26.5.2 (25F84)
- Production availability: macOS 15.2 and newer
- Fallback: custom SIMD-group/tiled Metal dense kernels
- Timing: dispatch through synchronized completion; graph construction,
  allocation, and correctness validation excluded from microbenchmark samples

## Correctness Gate

The benchmark compares every MPSGraph output element with the tiled custom
Metal oracle, then checks sampled values against the CPU reference. The largest
observed normalized RMS was `7.55e-7`. Tests also exercise nonzero offsets into
a shared checkpoint buffer, optional bias, tail shapes, and output canaries.

```sh
swift run -c release kg-trellis2-dense-bench --warmup 1 --iterations 3
```

| Workload | Shape `(M,N,K)` | Tiled median | SIMD median | MPSGraph median | MPSGraph / tiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse input | `4096,1536,8` | 1.931 ms | 1.682 ms | 1.033 ms | 1.868x |
| Cross K/V | `1029,3072,1024` | 13.853 ms | 8.506 ms | 2.454 ms | 5.645x |
| Conditioning | `1,9216,1536` | 0.492 ms | 0.476 ms | 2.388 ms | 0.206x |
| Sparse MLP up | `4096,8192,1536` | 216.315 ms | 132.262 ms | 28.524 ms | 7.584x |

Automatic routing deliberately retains custom Metal for the one-row
conditioning case.

## Real Model Integration

The one-step geometry run executes both 30-block flows, both decoders, mesh
extraction, GLB export, checkpoint mapping, and bounded arena teardown.

```sh
.build/arm64-apple-macosx/release/kg-trellis2 generate \
  --input build/hisar-native-20260801/hisar-alpha.png \
  --output build/hisar-native-20260801/hisar-geometry-mpsgraph-1step-v2.glb \
  --seed 42 --steps 1 --require-alpha --geometry-only \
  --checkpoint-root build/trellis2-geometry-install-20260801-1353 \
  --evidence build/hisar-native-20260801/hisar-geometry-mpsgraph-1step-v2.evidence.json
```

- Total: `166.00s`, down from the accepted `188.16s` baseline.
- Sparse flow: `65.429s`, down from `80.362s`.
- Shape flow: `43.171s`, down from `51.513s`.
- Maximum RSS: `2,654,404,608` bytes.
- Swaps: `0`.
- Arena peaks remained unchanged at `550,400,008` bytes for sparse flow and
  `421,215,752` bytes for shape flow.
- Output SHA-256 remained bit-identical:
  `9b543e1a8b957d5ca573ac6b0728739054c5b98d0716d63ba18f676cd81fdaa1`.

The integration result confirms a real improvement but also isolates attention
as the next dominant target. A 12-step run is intentionally deferred until the
attention path has a large correctness-gated gain.
