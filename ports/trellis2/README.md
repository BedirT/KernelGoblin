# TRELLIS.2 Apple GPU Runtime

This directory is the isolated compatibility layer that turns the pinned
TRELLIS.2 CUDA-oriented inference graph into a runnable PyTorch MPS pipeline.
It contains runtime preparation, strict checkpoint loading, backend shims,
memory controls, conformance tests, and the evidence-producing inference CLI.

| Pipeline | Status | Intended use |
| --- | --- | --- |
| `512` | Verified, default 12 steps | Faster correctness and quality baseline |
| `1024_cascade` | Verified, default 12 steps | High-resolution, memory-bounded generation |
| `1024` | Experimental | Direct high-resolution path, not end-to-end verified |
| `1536_cascade` | Experimental | Larger cascade, not end-to-end verified |

## Setup

Host requirements are macOS on Apple Silicon, Python 3.11, and
[`uv`](https://docs.astral.sh/uv/). Dependencies remain isolated under
`build/trellis2/.venv`; no global Python packages are installed.

```sh
./kg model setup trellis2
./kg model test trellis2
```

Setup clones upstream revision
`75fbf0183001ed9876c8dbb35de6b68552ee08bd` under the ignored build directory
and copies the selected MPS sparse-convolution backend into that checkout.
Compatibility shims are injected through an isolated `PYTHONPATH`; they do not
replace system or global packages.

## Checkpoint Provenance

| Artifact | Revision |
| --- | --- |
| `microsoft/TRELLIS.2-4B` | `af44b45f2e35a493886929c6d786e563ec68364d` |
| `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |
| Optional `briaai/RMBG-2.0` | `5df4c9c76d8170882c34f6986e848ee07fd0ba43` |

Checkpoint loading rejects missing and unexpected state keys. The sole
allowlist entry is `rope_phases`, a non-learned sparse-flow buffer regenerated
deterministically from the pinned configuration.

TRELLIS.2's weights are open, but its upstream image-conditioning graph uses
Meta's gated DINOv3 model. Accept the terms at
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and authenticate with `hf auth login`. The CLI preflights a small DINOv3 file so
missing access fails before TRELLIS weights download.

RMBG-2.0 is separately gated and licensed CC BY-NC 4.0. It is loaded only for
opaque inputs that need background removal. Useful input transparency bypasses
it, and `--no-preprocess` bypasses preprocessing entirely.

## Run

```sh
# Verified 512 path
./kg model run trellis2 \
  --input path/to/image.png \
  --output build/trellis2/output-512

# Verified low-memory 1024 cascade
./kg model run trellis2 \
  --pipeline-type 1024_cascade \
  --input path/to/image.png \
  --output build/trellis2/output-1024
```

`--seed` defaults to `42`. `--steps N` overrides all three samplers and is
useful for smoke tests. A one-step run is not a substitute for default-step
quality evidence.

Each run writes `trellis2-<pipeline>.glb` and `evidence.json`. Evidence records
the input and output hashes, exact revisions, geometry counts, runtime,
framework/platform details, CPU-fallback status, and export boundary. The CLI
rejects empty or non-finite vertices, out-of-range faces, and GLBs that do not
reload with geometry.

## Memory-Bounded Cascade

The 1024 cascade does not keep all selected checkpoints resident:

1. Lazy proxies postpone checkpoint loading until a component is first used.
2. DINOv3 is released after both conditioning resolutions are computed.
3. Sparse structure, 512 shape, decoder upsampling, 1024 shape, texture flow,
   shape decoding, and texture decoding execute as separate eviction stages.
4. `torch.inference_mode()` prevents latent tensors from retaining autograd
   graphs and earlier weights.
5. MPS synchronization precedes component deletion and allocator cleanup.
6. Sparse neighbor maps and matrix products use bounded chunks.

The verified 12-step 1024 run peaked at 15,275,048,960 bytes RSS on a 36 GB
Apple M3 Pro. It took 3,078.152 seconds, so this establishes memory fit and
correctness, not production performance.

## Disk-Bounded Checkpoint Streaming

Only one temporary safetensors file is stored at a time. After its parameters
load, the file and Hugging Face local-dir cache are removed in `finally`,
including failure paths. Allow at least 4 GB free for the largest checkpoint
plus metadata and enough room for the final GLB.

Use another writable filesystem for scratch downloads when the repo volume is
tight:

```sh
KG_TRELLIS2_DOWNLOAD_DIR=/path/to/scratch \
  ./kg model run trellis2 --pipeline-type 1024_cascade \
  --input image.png --output build/trellis2/output-1024
```

## Compatibility Surface

| Shim / overlay | Responsibility |
| --- | --- |
| `overlays/conv_mps.py` | Exact submanifold sparse convolution with memory-bounded MPS chunks |
| `shims/trellis2_mps/` | MPS device routing, sparse attention, and imported call-site patching |
| `shims/flex_gemm/` | Sparse nearest/trilinear grid-sampling semantics |
| `shims/o_voxel/` | Dual-grid extraction, sparse lookup, and portable GLB export |
| `shims/cumesh/` | CPU mesh-cleanup compatibility methods used by inference |
| `shims/nvdiffrast/` | Import-time placeholder for unused CUDA rendering imports |
| `streaming_loader.py` | Pinned strict loading, lazy materialization, eviction, and scratch cleanup |

The portable exporter stores TRELLIS-predicted base color and alpha as vertex
colors. It does not claim parity with upstream's CUDA-only UV unwrapping,
nvdiffrast texture bake, or renderer.

For exact end-to-end hashes, timings, memory measurements, troubleshooting, and
remaining work, see [`docs/TRELLIS2_PORT.md`](../../docs/TRELLIS2_PORT.md).
