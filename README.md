# KernelGoblin

> **Correct first. Fast second. Measured always.**

GPU ports are easy to announce and surprisingly hard to trust.

I kept running into the same problem: a useful model ships with a custom CUDA
kernel, someone rewrites something that looks similar, a tiny demo compiles,
and suddenly the whole model is described as "ported." What ran? On which GPU?
Did it match upstream? Was CPU fallback hiding in the graph? Nobody quite
knows.

KernelGoblin is our attempt to make that work less fuzzy. We pin the real
upstream source, preserve a reference, execute the physical backend, reconnect
the kernel to the model, and only then talk about speed.

The first journey is ambitious on purpose: run
[`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2), including its
full PBR mesh workflow, on Apple Silicon. The production Apple runtime is being
built in **Swift + Metal with no PyTorch dependency**. The working Torch/MPS
port stays around as a pinned conformance oracle while the native runtime earns
its way to parity.

## So, What Did We Actually Run?

This input went through the pinned TRELLIS.2 graph on an Apple M3 Pro:

| Input image | Verified 1024-cascade output |
| :---: | :---: |
| <img src="docs/assets/trellis2-input-t.png" alt="TRELLIS.2 steampunk machine input" width="420"> | <img src="docs/assets/trellis2-1024-cascade-preview.png" alt="Verified TRELLIS.2 1024 cascade geometry" width="420"> |

The output is a reloadable GLB with **6,717,817 vertices** and **13,774,312
faces**. The default 12-step run fit in **15.28 GB maximum RSS** on a 36 GB Mac
with CPU fallback disabled.

The good news: it works, the geometry is coherent, and the stage-wise memory
model fits comfortably.

The less exciting news: the reference MPS run took about **51 minutes**. This
is a correctness result, not a victory lap about speed. That slow but honest
baseline is exactly what the native Metal runtime is here to replace.

### Accomplished So Far

| Surface | Status | Strongest evidence |
| --- | --- | --- |
| O-Voxel Morton encode/decode | **Verified native Metal** | Bit-exact differential tests plus 65,536 randomized round trips |
| UV-space PBR raster | **Verified analytic Metal slice** | Physical Metal coverage, winding, degenerate, shared-edge, face-ID, and interpolation tests; CUDA nvdiffrast goldens remain |
| Swift safetensors runtime | **Verified native foundation** | Parsed the real 1.21 GB DINOv3 checkpoint and mapped it into one no-copy `MTLBuffer` |
| Real DINOv3 dense projection | **Verified native Metal slice** | `layer.0.attention.q_proj` from the pinned checkpoint, max absolute error `4.77e-7` against a CPU oracle |
| Real TRELLIS shape-flow projection | **Verified native Metal slice** | Pinned 2.58 GB checkpoint, BF16 `[1536,32]` input layer, 17 rows, zero BF16 bit mismatches |
| CPU mesh to flexible dual grid | **Verified reference extension** | Pinned O-Voxel algorithm through LibTorch, AppleClang portability patch, tetrahedron fixtures; native Swift bridge remains |
| Sparse PBR sampling and glTF packing | **Verified reference component** | Bounded sampling, xatlas seams, RGBA and metallic-roughness packing, GLB reload; native assembly remains |
| TRELLIS.2 512 image-to-3D | **Verified Torch/MPS oracle** | Default 12 steps, reloadable 61 MB GLB |
| TRELLIS.2 1024 cascade | **Verified Torch/MPS oracle** | Default 12 steps, 15.28 GB maximum RSS, reloadable 272.8 MB GLB |
| Full PBR image-to-3D | **In progress** | Native UV and synthetic bake pass; full model artifact still needs final end-to-end proof |
| Existing-mesh texturing | **In progress** | CPU voxelizer, UV policy, staged reference CLI, and PBR baker exist; full native model path remains |
| Swift + Metal full model | **In progress** | Checkpoint mapping and first real model layer pass; remaining operators and stages are explicit below |

That distinction matters. A kernel can be verified while a pipeline is still
unfinished. We do not promote the larger claim just because a nearby test is
green.

## How It Works

Every kernel follows the same path:

```mermaid
flowchart LR
    A["Pin it"] --> B["Understand it"]
    B --> C["Port it"]
    C --> D["Compare it"]
    D --> E["Run it in the model"]
    E --> F["Measure it"]
```

1. **Pin it.** Record the exact repository, revision, source paths, license,
   model checkpoint, and call site.
2. **Understand it.** Capture shapes, layouts, dtypes, edge behavior, overflow,
   errors, and synchronization boundaries before translating code.
3. **Port it.** Preserve the algorithm first. Clever changes can wait until we
   have something correct to compare against.
4. **Compare it.** Run deterministic references, randomized differential
   fixtures, invalid inputs, boundaries, and the real accelerator.
5. **Run it in the model.** A standalone kernel is not a model port. The real
   call site has to dispatch it and produce a valid artifact.
6. **Measure it.** Benchmarks disclose the device, workload, warmup,
   synchronization, transfers, allocations, and exactly what the timer covers.

## Fitting TRELLIS.2 Into Smaller Macs

TRELLIS.2 is called a 4B model, but that does not mean inference needs only the
bytes occupied by four billion weights. Image conditioning, multiple flow
models, shape and texture decoders, sparse topology, activations, allocator
state, mesh copies, UV charts, and a multi-million-texel bake all share the
same unified memory.

The current reference runtime already loads one component at a time:

```mermaid
flowchart LR
    A["Load one stage"] --> B["Run every consumer"]
    B --> C["Synchronize GPU"]
    C --> D["Evict weights and scratch"]
    D --> E["Load next stage"]
```

The native runtime goes further. Each installed stage will be page-aligned,
hash-verified, mapped from disk, and wrapped with
`MTLBuffer(bytesNoCopy:)`. Fixed scratch buffers refuse oversized work instead
of quietly expanding. The important granularity is a **whole TRELLIS stage**,
not one transformer layer at a time.

This choice comes from investigating
[`drumih/turbo-fieldfare`](https://github.com/drumih/turbo-fieldfare/tree/1859181ae26eb39c9698437f806be62adc01367c),
an immaculate Swift/Metal runtime that runs a large mixture-of-experts model on
small Macs. Its bounded installer, mapped buffers, verified receipts, and
memory ledger fit our case beautifully. Its expert cache does not: TRELLIS is
dense and touches every layer during every denoising step, so streaming the
same stage from SSD twelve times would trade a memory problem for an I/O
problem.

The complete decision and package design live in
[`docs/NATIVE_TRELLIS2_ARCHITECTURE.md`](docs/NATIVE_TRELLIS2_ARCHITECTURE.md).

## Try The Native Work

You need Apple Silicon, Xcode with the Metal toolchain, Swift 6.2+, CMake 3.25+,
and Ninja.

```sh
./kg doctor
./kg list
./kg validate

# Native Swift + Metal checkpoint and model-layer tests
./kg model native-setup trellis2
./kg model native-test trellis2

# Two independently buildable Metal kernels
./kg test trellis2/z_order
./kg test trellis2/uv_raster
./kg benchmark trellis2/uv_raster
```

The native CLI can inspect a real safetensors checkpoint without Torch:

```sh
swift run -c release kg-trellis2 \
  inspect-checkpoint /path/to/model.safetensors
```

It can also map the pinned DINOv3 checkpoint without a heap-sized weight copy
and dispatch a real attention projection on Metal:

```sh
swift run -c release kg-trellis2 \
  verify-dino-linear /path/to/dinov3/model.safetensors
```

And the first layer from the real 2.58 GB TRELLIS shape-flow checkpoint:

```sh
swift run -c release kg-trellis2 \
  verify-slat-input-layer /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
```

Both verification commands authenticate the complete checkpoint SHA-256
before reporting a pinned-model result. The TRELLIS slice maps 2.584 GB,
copies zero weight bytes into a second heap allocation, and compares all
26,112 outputs with an independent CPU calculation before the BF16 cast.

On the development M3 Pro, the current UV raster benchmark reports:

```text
backend=Metal device="Apple M3 Pro"
workload=1024x1024 faces=2 warmup=3 iterations=20
timing=host allocation + upload + draw + synchronized readback
median_ms=6.810
```

This is an analytic two-triangle workload, not a full texture-bake timing.

## Run The Reference Model Today

Until the native graph reaches end-to-end parity, the pinned Torch/MPS runtime
remains available as a reference. It is isolated under `build/`; it does not
install packages globally.

TRELLIS.2's own weights are open. The upstream image pipeline separately uses
Meta's gated DINOv3 encoder, so you must accept its terms on Hugging Face and
authenticate once.

```sh
./kg model setup trellis2
./kg model test trellis2

./kg model run trellis2 \
  --input image.png \
  --output build/trellis2/output-512

./kg model run trellis2 \
  --pipeline-type 1024_cascade \
  --input image.png \
  --output build/trellis2/output-1024
```

Those commands preserve the proven vertex-color export. The portable PBR path
is deliberately explicit until its upstream mesh fixtures pass:

```sh
./kg model run trellis2 --experimental-pbr \
  --input image.png --output build/trellis2/output-pbr
```

The experimental flag runs real Metal rasterization and glTF material packing,
but it does not yet carry an upstream-parity or completed full-run claim. The
reference code itself is not the desired shipping architecture. Its job is to
provide real checkpoints, intermediate fixtures, end-to-end artifacts, and a
known behavior contract while we remove Torch from the production path.

## What “Verified” Means Here

Compiling is not execution. Execution is not conformance. Conformance for one
kernel is not full-model success.

A verified kernel includes:

- immutable upstream provenance and license;
- a CPU reference or independently captured golden fixture;
- representative, boundary, randomized, and invalid-input coverage;
- a test that dispatches the named physical backend;
- error checking and explicit synchronization; and
- a correctness-gated benchmark with honest timing boundaries.

A verified model run additionally records input and output hashes, exact
checkpoint revisions, steps, device and framework details, geometry and
material validation, elapsed time, memory evidence, and a reload of the final
artifact.

See [`docs/TRELLIS2_PORT.md`](docs/TRELLIS2_PORT.md) for the original reference
runs and [`kernels/trellis2/uv_raster/README.md`](kernels/trellis2/uv_raster/README.md)
for the newest native kernel boundary.

## Where We Are Going

### Now

- Finish the reusable Swift tensor runtime and page-aligned stage installer.
- Port BF16/FP16 dense math, normalization, activations, attention, and sparse
  tensor primitives to Metal with real-checkpoint fixtures.
- Complete the native DINOv3, TRELLIS flow, decoder, sampler, and PBR stages.

### Next

- Produce native 512 image-to-PBR-GLB and existing-mesh texturing artifacts.
- Compare native intermediates and rendered views with the pinned reference.
- Profile the complete pipeline, then fuse the actual bottlenecks instead of
  guessing which kernel looks interesting.

### Later

- Verify direct 1024 and 1536 independently.
- Add more real model kernels across Metal and CUDA without making every
  dependency mandatory.
- Grow a library of ports that upstream model projects can actually trust.

Now that we know the reference graph works, we can make it native and fast.

## Repository Map

| Path | Purpose |
| --- | --- |
| `Sources/KernelGoblinTrellis2/` | No-Torch Swift checkpoint, memory, and model runtime |
| `kernels/<model>/<operation>/` | Independently buildable CPU, Metal, or CUDA kernels |
| `ports/trellis2/` | Pinned Torch/MPS oracle, geometry reference, and parity fixtures |
| `kg`, `tools/kg.py` | Dependency-free selection, setup, test, benchmark, and model CLI |
| `.agents/skills/` | Reusable repository-local kernel-port workflow |
| `.codex/agents/` | Narrow read-only research and review specialists |
| `docs/` | Architecture, evidence, toolchain, and harness decisions |
| `build/`, `.build/` | Ignored native builds, weights, scratch, and generated evidence |

## Building With Agents, Without Outsourcing Trust

The agent harness helps us trace upstream code, split bounded research, and run
independent reviews. It is not the evidence.

Deterministic tests, physical backend dispatch, final artifact validation, and
reproducible manifests remain the evidence. Start with [`AGENTS.md`](AGENTS.md)
and the repository skill in
[`docs/AGENT_HARNESS.md`](docs/AGENT_HARNESS.md) if you are contributing a port.

## Upstream And License

- TRELLIS.2 source: [`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2)
- Pinned source revision: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- TRELLIS.2-4B weights: `af44b45f2e35a493886929c6d786e563ec68364d`
- KernelGoblin license: [MIT](LICENSE)
- Third-party provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

Bring a real kernel, its real call site, and the hardware you care about. The
goblin will take it from there.
