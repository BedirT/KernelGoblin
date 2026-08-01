# TRELLIS.2 Native Geometry-Only Evidence

This is the durable record for the first native geometry-only control-path
artifact. The 81 MiB GLB and model weights remain ignored; their hashes identify
the exact accepted input and output. One diffusion step proves routing and the
artifact contract, not default-quality geometry.

## Runtime

| Field | Value |
| --- | --- |
| KernelGoblin implementation commit | `b22bed2` |
| Device | Apple M3 Pro, 36 GB unified memory |
| OS | macOS 26.5.2, build `25F84` |
| Runtime | Release Swift + Metal; no Torch or Python |
| Metal diagnostics | `MTL_DEBUG_LAYER=1` |
| Upstream source | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| TRELLIS weights | `af44b45f2e35a493886929c6d786e563ec68364d` |

## Install And Command

The run used a fresh `--feature geometry` install. It authenticated and linked
exactly five checkpoints: DINOv3, sparse-structure flow, sparse-structure
decoder, shape flow, and shape decoder. No texture-flow or texture-decoder file
was present in the install root.

```sh
./kg model setup trellis2 \
  --feature geometry \
  --root build/trellis2-geometry-install-20260801-1353

MTL_DEBUG_LAYER=1 /usr/bin/time -l ./kg model run trellis2 \
  --input build/hisar-native-20260801/hisar-alpha.png \
  --output build/hisar-native-20260801/hisar-geometry-final-smoke.glb \
  --checkpoint-root build/trellis2-geometry-install-20260801-1353 \
  --steps 1 \
  --seed 42 \
  --require-alpha \
  --geometry-only \
  --evidence build/hisar-native-20260801/hisar-geometry-final-smoke.evidence.json
```

## Artifact Contract

| Field | Value |
| --- | ---: |
| Input SHA-256 | `b92e1a1f7a86ab50107cb32f981b413fdf492d214d1aa2529953b5bdc6709137` |
| Sparse coordinates | 3,114 |
| Vertices | 1,684,057 |
| Faces | 3,671,130 |
| GLB bytes | 84,471,988 |
| GLB SHA-256 | `9b543e1a8b957d5ca573ac6b0728739054c5b98d0716d63ba18f676cd81fdaa1` |
| UV accessor | Absent |
| Embedded textures | 0 |
| Material | Neutral factor-only PBR material |
| Native reload | Passed, including accessor capacity and index-range validation |
| Independent reload | Assimp: 4 meshes, 3,671,130 faces, 0 textures, 1 material |

Assimp reports 1,689,051 imported vertices because it splits the indexed source
across four primitives. The native GLB accessor retains 1,684,057 vertices and
11,013,390 valid indices.

## Executed Stages

| Stage | Seconds | Arena peak | Live after close |
| --- | ---: | ---: | ---: |
| DINOv3 | 2.541 | 75,866,112 | 0 |
| Sparse-structure flow | 80.362 | 550,400,008 | 0 |
| Sparse-structure decoder | 8.747 | 234,881,024 | 0 |
| Shape flow | 51.513 | 421,215,752 | 0 |
| Shape decoder | 33.335 | 1,740,121,156 | 0 |
| Mesh extraction | 9.247 | 31,997,083 | 0 |
| Geometry export | 1.504 | 0 | 0 |

The evidence stage list contains no texture flow, texture decoder, UV
preparation, rasterization, inpainting, or PBR bake stage.

## Process Accounting

```text
188.16 real
2651209728 maximum resident set size
0 swaps
```

The timing includes process launch, checkpoint authentication, inference, mesh
extraction, GLB export, native reload validation, evidence writing, and CLI
reporting. It excludes weight installation and release compilation. A quality
run uses the default 12 steps and is expected to take roughly 30 to 32 minutes
on this machine based on the accepted 12-step stage timings.
