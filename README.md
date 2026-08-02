# KernelGoblin

> Correct first. Fast second. Measured always.

GPU ports are easy to announce and surprisingly hard to trust.

A kernel compiles. A tiny input runs. Maybe a fallback quietly did the real
work. Somewhere between those facts, the whole model gets called "ported."
KernelGoblin exists because we want a better answer than that.

This is a reusable home for custom model kernels and complete accelerator
ports: CUDA to Metal when we want a model to feel native on Apple Silicon,
CUDA to CUDA when the original kernel is correct but leaves performance on the
table, and CPU references wherever we need an honest baseline. Every port pins
its upstream source, carries a known-good oracle, executes the physical backend,
and reconnects to the real model before we make the larger claim.

Our first not-at-all-small experiment is
[`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2): image-to-3D,
existing-mesh texturing, UVs, six-channel PBR decoding, texture baking, and
reloadable GLB export on Apple Silicon.

## The Short Version

- The production Apple runtime is **Swift + Metal**, with optimized Apple
  MPSGraph dense and attention paths on newer macOS releases. It does not import or link
  Python, PyTorch, MLX, or third-party Swift packages; macOS 14 keeps the
  custom Metal fallback.
- Model stages are installed independently, hash-verified, memory-mapped, run,
  synchronized, and released before the next heavyweight stage is opened.
- Existing-mesh texturing has completed a real, default 12-step native run:
  223,711 source faces remained 223,711 output faces, two 2048 PBR textures
  reloaded from the GLB, maximum RSS was 3.04 GB, and the process recorded zero
  swaps.
- Native 512 image-to-PBR generation completed a real default 12-step run: a
  3,370,530-face GLB with two embedded 2048 PBR textures reloaded both natively
  and through Assimp, with zero swaps.
- The optional Torch/MPS environment is an isolated oracle. Default install,
  test, generation, and texturing commands are native.

That last distinction is intentional. Implemented is not the same as verified,
and a verified component is not automatically a verified model.

## What We Have Made So Far

The earlier reference journey gave us a real target instead of a collection of
guesses:

| Input | Reloaded TRELLIS.2 reference output |
| :---: | :---: |
| <img src="docs/assets/trellis2-input-t.png" alt="Steampunk machine used as the TRELLIS.2 input" width="420"> | <img src="docs/assets/trellis2-1024-cascade-preview.png" alt="Reloaded 1024-cascade TRELLIS.2 geometry" width="420"> |

The right-hand preview is the pinned Torch/MPS **oracle**, not a disguised
native result. Its 1024-cascade GLB contains 6,717,817 vertices and 13,774,312
faces. A historical acceptance run observed 15.28 GB maximum RSS and roughly
51 minutes on a 36 GB M3 Pro. Treat that as prior reference evidence, not a
portable benchmark: the original run predates the stricter committed raw-run
record we now require. It was still useful - it proved the graph and open
TRELLIS weights work and gave us immutable outputs to port against.

The native runtime now covers the complete 512 graph:

```mermaid
flowchart LR
    I["Image"] --> V["Apple image prep"]
    V --> D["DINOv3 conditioner"]
    D --> S["Sparse structure flow"]
    S --> O["Occupancy decoder"]
    O --> F["Shape flow"]
    F --> T["Texture flow"]
    T --> G["Shape decoder + O-Voxel mesh"]
    G --> P["Guided six-channel PBR decoder"]
    P --> U["UV raster + bake"]
    U --> B["Reloadable GLB"]
```

Existing-mesh texturing enters through a second front door:

```mermaid
flowchart LR
    M["Mesh"] --> N["Normalize + UV policy"]
    N --> X["Native O-Voxel voxelizer"]
    X --> E["Sparse shape encoder"]
    I["Reference image"] --> D["DINOv3"]
    E --> T["Guided texture flow + decoder"]
    D --> T
    T --> P["Metal PBR bake + GLB"]
```

## Evidence, Without The Victory-Lap Version

We use four evidence tiers so a useful low-level result cannot quietly become a
full-model claim.

| Tier | What it means | TRELLIS.2 today |
| --- | --- | --- |
| End-to-end verified | Default settings, real checkpoints, physical backend, artifact reload | Native 512 image-to-PBR, native existing-mesh texturing; Torch/MPS 512 and 1024-cascade oracles |
| Production-stage verified | Complete real stage and checkpoint execute with authenticated comparisons | DINOv3, both 30-block flows, sparse structure flow/decoder, shape encoder |
| Analytic or tiny-fixture verified | Real backend and exact contract, deliberately small workload | Shape and guided texture decoders, Morton coding, UV raster, sparse attention, O-Voxel mesh extraction, PBR bake and GLB packing |
| Outside the current native scope | Available only through an explicitly named reference path | Native 1024/cascade modes remain oracle-only |

Some useful numbers from the physical M3 Pro verification:

- 103 tests in 15 suites passed on the physical Metal backend. Separately,
  checkpoint-gated tests passed complete real-checkpoint stages and the
  production sparse trajectory.
- The full 12-step sparse-structure trajectory executed all 22 model calls;
  teacher-forced probes stayed below `0.01376` normalized RMS.
- The selected mixed-precision trajectory reached `0.7189` occupancy IoU with
  the captured oracle and passes the unchanged structural gates. Dense stays
  BF16; SDPA and the final Euler velocity projection use F32 accumulation.
- The full shape encoder reaches `0.000700` normalized RMS. The authenticated
  mesh-to-O-Voxel-to-encoder handoff reaches `0.000555`, with exact sparse
  coordinates and all four subdivision guides.
- Shape and texture decoder PBR outputs reach `0.000628` and `0.001251`
  normalized RMS on their pinned fixtures.
- Every recorded heavyweight stage returns its bounded Metal arena to zero live
  bytes before its checkpoint mapping is closed.

The detailed ledger, exact hashes, timing boundaries, and semantic gaps live in
[`docs/TRELLIS2_PORT.md`](docs/TRELLIS2_PORT.md). The machine-readable source of
truth is [`ports/trellis2/model.toml`](ports/trellis2/model.toml).
The performance story, including the original bottleneck and the honest
`840.812s -> 125.813s` sparse-flow comparison, is documented separately in
[`docs/evidence/trellis2-performance-progression-m3pro-2026-08-01.md`](docs/evidence/trellis2-performance-progression-m3pro-2026-08-01.md).

![Three deterministic views of the native existing-mesh PBR result](docs/assets/trellis2-native-texturing-preview.png)

The preview above is rendered from the accepted native GLB. Its exact command,
hardware/toolchain disclosure, raw stage ledger, process accounting, and hashes
are committed in
[`docs/evidence/trellis2-native-texturing-m3pro-2026-08-01.md`](docs/evidence/trellis2-native-texturing-m3pro-2026-08-01.md).

![Three deterministic views of the native image-to-PBR result](docs/assets/trellis2-native-generation-preview.png)

This second preview is the accepted native image-to-PBR GLB, not the oracle.
Its 12-step run took 2,357.20 seconds on this M3 Pro, reached 2.99 GB maximum
RSS and 7.25 GB macOS peak footprint, and recorded zero swaps. The command,
stage ledger, hashes, and independent reload are in
[`docs/evidence/trellis2-native-generation-m3pro-2026-08-01.md`](docs/evidence/trellis2-native-generation-m3pro-2026-08-01.md).

## Native Quick Start

You need an Apple Silicon Mac, Xcode with the Metal toolchain, Swift 6.2+,
CMake 3.25+, and Ninja.

The runtime supports macOS 14. On macOS 15+, production SLat attention uses
Apple's GPU-backed MPSGraph SDPA with F32 accumulation because the faster BF16
route fails the pinned 12-step geometry gates. On macOS 15.2+, large dense
projections preserve the BF16 model boundary in MPSGraph, except for the final
Euler velocity projection. Small, segmented, or unavailable cases stay on our
custom Metal kernels, where graph dispatch overhead would cost more than it
saves.

```sh
./kg doctor
./kg list
./kg model list
./kg validate
```

For the normal case, install once and then give KernelGoblin an image:

```sh
./kg model setup trellis2
./kg generate trellis2 image.png
```

That is the complete default 512 image-to-PBR path: Apple Vision foreground
masking for opaque images, seed 42, 12 steps, 2048 textures, and a reloadable
GLB. The result is written to `build/trellis2/outputs/image.glb`, with its
evidence next to it. Repeated `generate` calls reuse the release binary instead
of asking SwiftPM to rebuild it. The longer `./kg model run trellis2 ...` form
remains the reproducibility and experiment interface.

Generation output is meant to be read, not decoded. It announces the resolved
input, output, foreground policy, and quality settings; numbers every stage in
plain language; reports each completed stage's wall time; and finishes with the
total duration, model path, file size, mesh size, PBR coverage, GLB validation,
device, and evidence path. Exact hashes and detailed allocation accounting stay
in the neighboring JSON record.

TRELLIS.2's own model weights are open under MIT. Its image pipelines also use
Meta's separately gated DINOv3 encoder. Accept the DINOv3 terms on Hugging Face
once and provide a read token for installation:

```sh
export HF_TOKEN=hf_...

# Install all eight pinned components. Existing verified HF cache files are
# linked rather than downloaded again.
./kg model setup trellis2

# Or install only what one workflow needs.
./kg model setup trellis2 --feature geometry
./kg model setup trellis2 --feature generate
./kg model setup trellis2 --feature texture
```

Generate a PBR GLB from an image:

```sh
./kg model run trellis2 \
  --input image.png \
  --output build/trellis2/my-model.glb \
  --steps 12 \
  --texture-size 2048
```

Generate only geometry when you want to inspect the shape before spending time
on texture flow, UVs, and baking:

```sh
./kg model run trellis2 \
  --input image.png \
  --output build/trellis2/my-mesh.glb \
  --steps 12 \
  --geometry-only
```

`--geometry-only` stops after shape decoding, small-hole repair, and mesh
extraction. The GLB gets normals and a neutral factor-only material, but no
UV attribute or model-generated textures. A geometry-only install needs five
pinned checkpoints instead of the seven used for full image generation. It does
not load or execute texture flow, the texture decoder, UV preparation, Metal
rasterization, inpainting, or PBR baking.

Texture an existing mesh from a reference image:

```sh
./kg model texture trellis2 \
  --mesh model.ply \
  --input reference.png \
  --output build/trellis2/textured.glb \
  --uv-policy preserve-or-generate \
  --steps 12 \
  --texture-size 2048
```

Opaque inputs use Apple Vision foreground masking by default. Use
`--require-alpha` when an alpha matte is part of your input contract, or
`--accept-opaque` only when the background is already harmless.

Run the native verification ladder:

```sh
./kg model test trellis2
./kg model native-audit trellis2
./kg model native-benchmark trellis2
./kg model native-attention-benchmark trellis2
./kg model native-normalization-benchmark trellis2
./kg model native-pbr-benchmark trellis2
```

`./kg model test` is intentionally substantial: it authenticates real
checkpoints and executes complete stages, including the production 12-step
sparse trajectory. Use focused `swift test --filter ...` commands while
developing one component.

## Why A 4B Model Needs More Than 8 GB

"4B" counts parameters in one headline model. It is not a memory budget.
TRELLIS.2 also needs DINO conditioning, multiple flow checkpoints, sparse
activations, decoder features, coordinate maps, mesh topology, UV seams, and
multi-million-texel PBR buffers. On a unified-memory Mac, the CPU and GPU share
the same pool too.

KernelGoblin avoids keeping the entire graph resident:

```mermaid
flowchart LR
    A["Map one verified checkpoint"] --> B["Run the stage"]
    B --> C["Wait for Metal"]
    C --> D["Copy the small semantic result"]
    D --> E["Release arena + unmap weights"]
    E --> F["Open the next stage"]
```

This is where the ideas from
[`drumih/turbo-fieldfare`](https://github.com/drumih/turbo-fieldfare/tree/1859181ae26eb39c9698437f806be62adc01367c)
fit beautifully: authenticated receipts, mapped buffers, explicit ownership,
and a memory ledger. We did **not** copy its expert cache. TRELLIS stages are
dense and touch every block on every denoising step; streaming those same
weights from SSD twelve times would replace a memory problem with an I/O
problem.

We also tested the obvious K/V trick instead of assuming it would help. Positive
and negative cross-attention K/V are timestep-invariant, so an opt-in exact cache
can remove 600 repeated block projections from the 12-step sparse flow. On M3
Pro, though, it raised the arena peak from 560 MB to 1.319 GB and changed a
99.57-second diagnostic run to 99.33 seconds. That is a poor default trade: lots
of memory, no established end-to-end win. Self-attention K/V cannot be reused
exactly because the diffusion latent changes at every call. The next useful
cache work is therefore an explicit approximate mode, measured against geometry
rather than borrowed image-model quality claims.

The newer cache methods, their tradeoffs, and our implementation order are in
[`docs/TRELLIS2_CACHE_ACCELERATION.md`](docs/TRELLIS2_CACHE_ACCELERATION.md).

The full rationale is in
[`docs/NATIVE_TRELLIS2_ARCHITECTURE.md`](docs/NATIVE_TRELLIS2_ARCHITECTURE.md).

## The Optional Oracle

Python and Torch are kept on purpose, but only behind explicitly named oracle
commands in an ignored environment under `build/`:

```sh
./kg model oracle-setup trellis2
./kg model oracle-test trellis2

./kg model oracle-run trellis2 \
  --pipeline-type 1024_cascade \
  --input image.png \
  --output build/trellis2/oracle-output
```

The oracle generates fixtures and answers questions such as "what did pinned
upstream do at this exact boundary?" It is not installed by native setup and is
not loaded by native inference.

## Known Differences We Still Care About

This is a proper port, not a claim that Apple and CUDA take identical floating-
point paths.

- Native `--seed` uses documented SplitMix64 plus Box-Muller noise. It is
  deterministic across native runs, but the same number does not reproduce
  PyTorch's RNG stream.
- Apple Vision foreground masking is a portable native policy, not numerical
  parity with upstream RMBG.
- Supplied UVs can be preserved as a useful extension. Pinned upstream
  texturing regenerates its atlas through CuMesh.
- Model I/O unwrap is used only when it preserves all valid faces. A
  deterministic per-face atlas is the topology-safe native fallback; it is
  reproducible, but it is not CuMesh atlas-quality parity.
- Metal and Torch use different BF16 reduction trees. Teacher-forced stage
  comparisons are tight, while a complete free-running trajectory accumulates
  measurable sparse occupancy drift.
- The PBR bake benchmark is an analytic synchronized bake-stage benchmark, not
  a full export or representative-model performance claim.

These are tracked as contracts, not buried as footnotes. The next work is to
replace portability tiers with authenticated upstream-equivalent tiers where
that improves the artifact, then optimize the kernels that profiling says are
actually expensive.

## How A Kernel Enters The Repo

```mermaid
flowchart LR
    A["Pin source + license"] --> B["Capture the real call contract"]
    B --> C["Write or preserve an oracle"]
    C --> D["Port for behavior"]
    D --> E["Differential test on the GPU"]
    E --> F["Reconnect the model"]
    F --> G["Benchmark honestly"]
```

Each kernel is independently selectable. `./kg setup trellis2/z_order` should
not install a model, and adding a future CUDA optimization should not make an
Apple user build it. The checked-in `$port-gpu-kernel` skill and `AGENTS.md`
give Codex the same verification rules we use manually.

Full-model runtimes follow the same idea at a higher level: each
`ports/<model>/model.toml` registers the runtime and names its thin adapter in
`kg`. The manifest owns provenance and policy; the adapter owns the commands
that are inherently model-specific.

`kernel.toml` is the registration boundary. A new kernel brings its own
`CMakeLists.txt`, tests, and benchmark; root CMake and CI discover it without a
new switch. The manual equivalent of `./kg test trellis2/z_order` is:

```sh
cmake -S . -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DKG_SELECTED_KERNEL_DIR=kernels/trellis2/z_order
cmake --build build/manual
ctest --test-dir build/manual --output-on-failure
./build/manual/kernel/kg_trellis2_z_order_bench
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `kg`, `tools/kg.py` | Dependency-free selection, setup, validation, model, and benchmark CLI |
| `kernels/<model>/<operation>/` | One independently buildable CPU/Metal/CUDA kernel contract |
| `Sources/KernelGoblinTrellis2/` | Native checkpoint, model, geometry, PBR, memory, and coordinator runtime |
| `ports/trellis2/` | Optional pinned Torch/MPS oracle and fixture exporters |
| `.agents/skills/port-gpu-kernel/` | Reusable agent workflow for kernel intake and verification |
| `.codex/agents/` | Read-only upstream, correctness, and benchmark specialists |
| `docs/` | Architecture decisions, provenance, evidence, and toolchain notes |
| `build/`, `.build/` | Ignored weights, environments, artifacts, and build products |

## Where We Are Going

1. Add image-level comparisons between native 512 and the pinned oracle, not
   just structural GLB validation.
2. Add rendered native before/after views for more meshes and conditioning
   images.
3. Improve the portable UV fallback toward CuMesh-quality charting while
   preserving strict topology gates.
4. Profile real end-to-end runs, tile the expensive sparse kernels, and publish
   representative workload benchmarks only after correctness gates.
5. Add the next model port without turning setup into one giant dependency
   bucket.

No shortcuts, no mystery fallbacks, and no pretending one green kernel means
the whole model is done. That is the goblin's job.

## Provenance And License

- TRELLIS.2 source: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- TRELLIS.2-4B weights: `af44b45f2e35a493886929c6d786e563ec68364d`
- TRELLIS image-large weights: `25e0d31ffbebe4b5a97464dd851910efc3002d96`
- DINOv3 weights: `ea8dc2863c51be0a264bab82070e3e8836b02d51`
- Turbo Fieldfare research pin: `1859181ae26eb39c9698437f806be62adc01367c`

KernelGoblin's code is MIT licensed. Upstream source and model licenses remain
theirs; the selected kernel and model manifests record exact provenance.
