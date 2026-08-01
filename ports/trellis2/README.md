# TRELLIS.2 Reference Runtime

This directory is the pinned **reference and conformance runtime** for the
native Swift + Metal TRELLIS.2 port. It uses Python, Torch, and MPS on purpose:
we need a working upstream-shaped graph that can produce checkpoints,
intermediate fixtures, and end-to-end artifacts for the no-Torch runtime. The
native graph now exists; this directory remains because independent oracles are
still how we catch semantic drift.

It is not the intended shipping runtime. Native code lives under
`Sources/KernelGoblinTrellis2/`; the architecture boundary is documented in
[`docs/NATIVE_TRELLIS2_ARCHITECTURE.md`](../../docs/NATIVE_TRELLIS2_ARCHITECTURE.md).

| Reference path | Status | What the evidence proves |
| --- | --- | --- |
| `512` image-to-3D | Verified, default 12 steps | Complete pinned graph, finite geometry, reloadable vertex-color GLB |
| `1024_cascade` image-to-3D | Verified, default 12 steps | Stage-wise memory fit at 15.28 GB maximum RSS |
| Experimental PBR bake | Reference component tests pass | xatlas, sparse sampling, material packing, GLB reload; upstream mesh parity pending |
| Existing-mesh texturing | Orchestration implemented | Full reference execution and artifact evidence pending |
| `1024` and `1536_cascade` | Experimental | No default-step end-to-end claim |

## Setup

You need Apple Silicon, Python 3.11, and
[`uv`](https://docs.astral.sh/uv/). Everything is installed below
`build/trellis2/`; nothing is added to the global Python environment.

```sh
./kg model oracle-setup trellis2
./kg model oracle-test trellis2
```

`oracle-test` runs the compatibility and primitive suite. It is not a full
generation. The stronger model gate is an explicit `oracle-run` with the
default 12 sampling steps and a validated output artifact.

Setup clones TRELLIS.2 revision
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`, verifies the checkout, and injects
the selected MPS shims through an isolated `PYTHONPATH`.

## Checkpoints And Access

| Artifact | Revision | Access |
| --- | --- | --- |
| `microsoft/TRELLIS.2-4B` | `af44b45f2e35a493886929c6d786e563ec68364d` | Open MIT weights |
| `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` | Open MIT weights |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` | Gated DINOv3 license |
| Optional `briaai/RMBG-2.0` | `5df4c9c76d8170882c34f6986e848ee07fd0ba43` | Gated CC BY-NC 4.0 |

TRELLIS.2's weights are open. DINOv3 is a separate image encoder used by the
upstream graph, so image-conditioned generation still needs the owner to
accept Meta's terms and authenticate with `hf auth login`. KernelGoblin does
not accept a license on your behalf.

RMBG is needed only when an opaque input needs background removal. A useful
alpha channel or `--no-preprocess` avoids that optional component.

Checkpoint loading rejects missing and unexpected state keys. The sole
allowlist entry is `rope_phases`, a non-learned buffer regenerated from the
pinned configuration.

## Run The Proven Reference Paths

```sh
./kg model oracle-run trellis2 \
  --input image.png \
  --output build/trellis2/output-512

./kg model oracle-run trellis2 \
  --pipeline-type 1024_cascade \
  --input image.png \
  --output build/trellis2/output-1024
```

These default commands retain the previously verified vertex-color export.
Run the portable material path only through its explicit experimental gate:

```sh
./kg model oracle-run trellis2 --experimental-pbr \
  --input image.png --output build/trellis2/output-pbr
```

`--seed` defaults to `42`. `--steps N` is useful for failure-finding, but a
one-step smoke run is never presented as default-quality evidence.

Each run writes a GLB and `evidence.json`. The evidence records revisions,
hashes, effective steps, preprocessing, device and framework details,
CPU-fallback policy, geometry counts, memory/runtime data when measured, and
the actual export boundary. The CLI rejects empty or non-finite geometry,
invalid indices, and artifacts that do not reload.

## Why It Fits

The model name says 4B, but unified memory also holds DINO, several flow and
decoder components, activations, sparse topology, allocator state, a large
mesh, UV data, and textures. Loading everything together is unnecessary.

The reference cascade therefore runs one lifetime at a time:

1. Materialize one component lazily.
2. Run all consumers under `torch.inference_mode()`.
3. Synchronize MPS.
4. Release the component and bounded sparse scratch.
5. Continue with the small semantic output that the next stage needs.

The verified 1024 cascade peaked at 15,275,048,960 bytes RSS with no process
swaps on a 36 GB Apple M3 Pro. It took 3,078.152 seconds. This proves memory fit
and correctness for that device and input, not production speed or a universal
minimum-memory claim.

## Experimental PBR And Mesh Texturing

The reference now contains the pieces needed for a real material pipeline:

- deterministic UV preservation or xatlas regeneration;
- a physical Metal UV position/face-ID rasterizer;
- half-voxel sparse trilinear PBR sampling;
- bounded closest-surface projection and texture fill;
- base-color RGBA plus glTF metallic-roughness packing; and
- a staged existing-mesh texturing CLI in `texture.py`.

That is meaningful progress, but it is not yet upstream mesh-processing
parity. The current experimental path substitutes portable simplification and
xatlas for CuMesh cleanup/unwrap behavior. Exact topology, normal, overlap,
projection, and nvdiffrast coverage fixtures are still required before this
becomes the default verified export.

## Compatibility Surface

| Path | Responsibility |
| --- | --- |
| `overlays/conv_mps.py` | Submanifold sparse convolution with bounded MPS chunks |
| `shims/trellis2_mps/` | MPS routing and sparse attention |
| `shims/flex_gemm/` | Half-voxel nearest/trilinear sparse sampling |
| `shims/o_voxel/` | Dual-grid reference extraction and export integration |
| `shims/cumesh/` | Portable CPU compatibility methods |
| `streaming_loader.py` | Pinned strict loading, lazy materialization, eviction, and cleanup |
| `pbr/` | Experimental UV, sampling, texture, and GLB reference components |

For exact hashes, timings, known gaps, and the native migration plan, read
[`docs/TRELLIS2_PORT.md`](../../docs/TRELLIS2_PORT.md).
