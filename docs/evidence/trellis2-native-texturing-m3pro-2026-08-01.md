# TRELLIS.2 Native Existing-Mesh Texturing Evidence

This is the immutable human-readable record for the first accepted default-step
native existing-mesh texturing artifact. Generated GLBs and model weights remain
ignored; their hashes make the accepted inputs and output identifiable.

## Runtime

| Field | Value |
| --- | --- |
| KernelGoblin runtime commit | `9f634f38d2e9ba1e735ff75c720eb7e85606c45c` |
| Build | SwiftPM release, `kg-trellis2` prebuilt before timing |
| Device | MacBook Pro `Mac15,6`, Apple M3 Pro, 11 CPU cores, 14 GPU cores, 36 GB unified memory |
| Metal | Metal 4; Apple metal `32023.883` |
| OS | macOS 26.5.2, build `25F84` |
| Xcode | 26.6, build `17F113` |
| Swift | Apple Swift 6.3.3, `swiftlang-6.3.3.1.3` |
| Metal diagnostics | `MTL_DEBUG_LAYER=1` |
| Upstream source | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| TRELLIS weights | `af44b45f2e35a493886929c6d786e563ec68364d` |

No thermal or power-state normalization was recorded, so wall time is an
observed acceptance-run duration, not a portable benchmark.

## Inputs And Command

```text
mesh SHA-256  = 56b16fba10017445de9529b2c2c1f68a97d2ca883293b27fa5efce116c6489b1
image SHA-256 = db468bad8a04f1474a8d68140c07501b013b3ec6124b911fb7852675d64c05ee
```

```sh
MTL_DEBUG_LAYER=1 /usr/bin/time -l \
  .build/arm64-apple-macosx/release/kg-trellis2 texture \
  --mesh build/trellis2/upstream/assets/example_texturing/the_forgotten_knight.ply \
  --input docs/assets/trellis2-input-t.png \
  --output build/native-e2e-texture-topology-fixed-9f634f3/forgotten-knight.glb \
  --checkpoint-root build/trellis2-native-install-smoke-20260801-105852 \
  --steps 12 \
  --texture-size 2048 \
  --uv-policy regenerate \
  --seed 42 \
  --require-alpha
```

The `/usr/bin/time` boundary starts at release-process launch and ends after
checkpoint authentication, all stages, GLB encoding/atomic write, reload
validation, evidence JSON write, and CLI reporting. It excludes release build,
weight installation, and input acquisition.

## Artifact Contract

| Field | Value |
| --- | ---: |
| Source vertices | 153,723 |
| Source faces | 223,711 |
| Voxel coordinates | 822,875 |
| Shape latent coordinates | 1,905 |
| PBR coordinates | 822,875 |
| GLB accessor vertices | 671,133 |
| GLB faces | 223,711 |
| Assimp imported faces | 223,711 |
| Embedded PNG textures | 2 |
| Texture dimensions | 2048 x 2048 |
| Covered texels | 2,959,061 |
| GLB bytes | 36,571,340 |
| GLB reload | Passed |
| GLB SHA-256 | `37ad68cca494628cf29dafdfa1989200dd448048af9346dbe4b126ec710083c3` |

The UV implementation was
`deterministic-native-per-face-atlas-fallback`; exact upstream/CuMesh tier was
false. UV fingerprint:
`e4aaa20b5e2a1484cfdf06f42bb94c399fdbff02fed1bccef8ea72e3f5f720e4`.

The native seed was `42` under `splitmix64-box-muller-v1`. It is not a PyTorch
RNG identity claim.

## Raw Stage Ledger

| Stage | Seconds | Checkpoint SHA-256 | Arena capacity | Arena peak | Live after close | Cumulative requested | Allocations |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Mesh voxelization | 1.013938 | none | 0 | 0 | 0 | 0 | 0 |
| DINOv3 | 4.384777 | `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` | 268,435,456 | 75,866,112 | 0 | 1,626,906,624 | 314 |
| Shape encoder | 16.836726 | `f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41` | 6,442,450,944 | 937,105,448 | 0 | 9,471,100,848 | 201 |
| Texture flow | 156.470822 | `8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d` | 4,294,967,296 | 273,000,240 | 0 | 162,024,980,784 | 10,974 |
| Texture decoder | 18.617414 | `97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c` | 6,442,450,944 | 849,578,732 | 0 | 10,856,013,112 | 205 |
| PBR export | 4.914790 | none | 536,870,912 | 201,239,656 | 0 | 201,239,656 | 8 |

Arena capacity is a ceiling, not resident allocation. Cumulative requested is
allocation traffic and must not be added to peak memory.

## Raw Process Accounting

```text
202.92 real
12.34 user
3.14 sys
3042181120 maximum resident set size
96836 page reclaims
316346 page faults
0 swaps
0 block input operations
0 block output operations
146 voluntary context switches
174287 involuntary context switches
157207953620 instructions retired
54345360087 cycles elapsed
7477478456 peak memory footprint
```

The process maximum RSS and macOS peak memory footprint are reported separately
because unified-memory mappings and resident accounting are not interchangeable.

## Independent Reload

The native GLB validator reported 671,133 vertices, 671,133 indices, two
2048-by-2048 textures, `OPAQUE`, and `doubleSided=true`. Assimp independently
loaded one triangle mesh, two embedded textures, one material, and exactly
223,711 faces. Assimp merged duplicate imported vertices, which is allowed; it
did not remove faces.
