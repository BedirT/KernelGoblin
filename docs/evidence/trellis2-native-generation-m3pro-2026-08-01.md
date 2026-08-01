# TRELLIS.2 Native Image-To-PBR Generation Evidence

This is the immutable human-readable record for the first accepted default-step
native 512 image-to-PBR artifact. The 361 MB GLB and model weights remain
ignored; their hashes identify the exact accepted input and output.

## Runtime

| Field | Value |
| --- | --- |
| KernelGoblin runtime commit | `d95c680b470be65dc36be5d6523b5795411ad4d2` |
| Build | SwiftPM release, `kg-trellis2` prebuilt before timing |
| Device | MacBook Pro `Mac15,6`, Apple M3 Pro, 11 CPU cores, 14 GPU cores, 36 GB unified memory |
| Metal | Metal 4; Apple metal `32023.883` |
| OS | macOS 26.5.2, build `25F84` |
| Xcode | 26.6, build `17F113` |
| Swift | Apple Swift 6.3.3, `swiftlang-6.3.3.1.3` |
| Metal diagnostics | `MTL_DEBUG_LAYER=1` |
| Upstream source | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| TRELLIS weights | `af44b45f2e35a493886929c6d786e563ec68364d` |

No thermal or power-state normalization was recorded. Wall time is an observed
acceptance-run duration, not a portable benchmark.

## Input And Command

```text
image SHA-256 = db468bad8a04f1474a8d68140c07501b013b3ec6124b911fb7852675d64c05ee
```

```sh
MTL_DEBUG_LAYER=1 /usr/bin/time -l \
  .build/arm64-apple-macosx/release/kg-trellis2 generate \
  --input docs/assets/trellis2-input-t.png \
  --output build/native-e2e-generate-d95c680/trellis2-512-pbr.glb \
  --checkpoint-root build/trellis2-native-install-smoke-20260801-105852 \
  --steps 12 \
  --texture-size 2048 \
  --seed 42 \
  --require-alpha
```

The timing boundary starts at release-process launch and ends after checkpoint
authentication, all nine recorded stages, GLB encoding and atomic write,
native reload validation, evidence JSON write, and CLI reporting. It excludes
release build, weight installation, preview rendering, and input acquisition.

## Artifact Contract

| Field | Value |
| --- | ---: |
| Conditioning tokens | 1,029 |
| Sparse occupancy coordinates | 3,556 |
| Shape coordinates / decoded mesh vertices | 1,575,509 |
| Decoded mesh faces | 3,370,530 |
| GLB accessor vertices | 10,111,590 |
| GLB faces | 3,370,530 |
| Assimp imported vertices / faces | 10,111,590 / 3,370,530 |
| Assimp meshes / materials | 12 / 1 |
| Embedded PNG textures | 2 |
| Texture dimensions | 2048 x 2048 |
| Covered texels | 2,959,012 |
| GLB bytes | 378,336,688 |
| GLB reload | Passed natively and independently through Assimp |
| GLB SHA-256 | `b3e941111c1209f86311ec575ef09d110076c885e978e7063ad140d3049d79bf` |

The GLB is split into 12 primitives to stay under practical accessor sizes.
Assimp preserved the total face and vertex counts and loaded two embedded
textures. The UV implementation was
`deterministic-native-per-face-atlas-fallback`; exact upstream/CuMesh tier was
false. UV fingerprint:
`bc57575431a2293c0eeea208a41c9fb3575fa2fa021d8eb07b92ced1c1f2af7f`.

The native seed was `42` under `splitmix64-box-muller-v1`. It is stable native
reproducibility, not a PyTorch RNG identity claim.

## Raw Stage Ledger

| Stage | Seconds | Checkpoint SHA-256 | Arena capacity | Arena peak | Live after close | Cumulative requested | Allocations |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DINOv3 | 4.497636 | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` | 268,435,456 | 75,866,112 | 0 | 1,626,906,624 | 314 |
| Sparse structure flow | 1067.636654 | `ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6` | 671,088,640 | 555,905,112 | 0 | 614,499,154,008 | 20,072 |
| Sparse structure decoder | 10.140037 | `1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a` | 335,544,320 | 234,881,024 | 0 | 1,116,733,440 | 64 |
| Shape flow | 711.740772 | `ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f` | 4,294,967,296 | 499,056,212 | 0 | 511,894,731,348 | 19,161 |
| Texture flow | 473.530676 | `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d` | 4,294,967,296 | 505,883,696 | 0 | 292,528,051,248 | 10,974 |
| Shape decoder | 35.265612 | `e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581` | 6,442,450,944 | 1,731,914,492 | 0 | 22,175,298,380 | 208 |
| Mesh extraction | 7.961863 | none | 67,108,864 | 29,934,671 | 0 | 29,934,671 | 3 |
| Texture decoder | 35.013056 | `97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c` | 6,442,450,944 | 1,728,631,324 | 0 | 22,191,896,208 | 205 |
| PBR export | 10.837283 | none | 536,870,912 | 439,852,768 | 0 | 439,852,768 | 8 |

Arena capacity is a ceiling, not resident allocation. Cumulative requested is
allocation traffic across repeated denoising calls and must not be summed into
a memory peak.

## Raw Process Accounting

```text
2357.20 real
31.02 user
10.55 sys
2994143232 maximum resident set size
282940 page reclaims
672044 page faults
0 swaps
0 block input operations
0 block output operations
419 voluntary context switches
525779 involuntary context switches
474481617739 instructions retired
143737065225 cycles elapsed
7245727112 peak memory footprint
```

The process maximum RSS and macOS peak memory footprint are reported separately
because unified-memory mappings and resident accounting are not interchangeable.

## Independent Reload

The native validator reported 10,111,590 vertices and indices, two 2048-by-2048
textures, `OPAQUE`, and `doubleSided=true`. Assimp independently loaded 12
triangle primitives, two embedded textures, one PBR material, 10,111,590
vertices, and 3,370,530 faces.
