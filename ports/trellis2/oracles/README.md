# TRELLIS.2 Development Oracles

This directory is not part of the native runtime. It contains small tools for
capturing immutable behavior from the pinned upstream Torch implementation so
the Swift + Metal port can be tested without importing Torch.

`export_slat_block_fixture.py` loads only block 0 from the authenticated
`shape_slat_flow_model_512` checkpoint, runs two deterministic BF16 tokens with
the production 3D RoPE path, and
writes:

- the final 1,536-channel BF16 output;
- BF16 values at every meaningful block boundary; and
- provenance, input formulas, source revisions, checkpoint hash, and output
  hashes.

Regeneration is deliberately explicit:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_slat_block_fixture.py \
  --checkpoint /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --output build/trellis2/slat-block0-tiny.bf16
```

Compare the generated hashes and values before replacing a committed fixture.
Do not commit the checkpoint or extracted weights. The tiny activation fixture
is derived from `microsoft/TRELLIS.2-4B` at revision
`af44b45f2e35a493886929c6d786e563ec68364d`, whose weights are MIT licensed.

The complete stage oracle uses the same deterministic two-token contract while
executing all 30 production blocks:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_slat_shape_flow_fixture.py \
  --checkpoint /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --output build/trellis2/slat-shape-flow-tiny.f32
```

This fixture covers every shape-flow weight and operation. Its deliberately
small token count makes it a conformance test, not a representative memory or
performance workload.

The sampler fixture isolates the pinned sparse Flow-Euler schedule, guidance
interval, sequential positive/negative CFG calls, population-standard-deviation
rescale, and Euler update without loading model weights:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_flow_euler_fixture.py \
  --output Tests/KernelGoblinTrellis2Tests/Fixtures/flow-euler-sparse.f32
```

The texture-flow stage uses the matching exporter and authenticated texture
checkpoint:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_slat_texture_flow_fixture.py \
  --checkpoint /path/to/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
  --output Tests/KernelGoblinTrellis2Tests/Fixtures/slat-texture-flow-tiny.f32
```

The real-checkpoint sampler integration fixture intentionally uses two Euler
steps so it verifies sequential CFG and repeated stage execution without making
fixture regeneration unnecessarily expensive:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_slat_shape_sampler_fixture.py \
  --checkpoint /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --output Tests/KernelGoblinTrellis2Tests/Fixtures/slat-shape-sampler-2step.f32
```

Texture sampling is positive-only at the pinned guidance strength of one. Its
matching fixture verifies repeated noise-plus-normalized-shape concatenation,
the complete texture flow, Euler state updates, and final texture
denormalization:

```sh
build/trellis2/.venv/bin/python \
  ports/trellis2/oracles/export_slat_texture_sampler_fixture.py \
  --checkpoint /path/to/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
  --output Tests/KernelGoblinTrellis2Tests/Fixtures/slat-texture-sampler-2step.f32
```

Both integration traces store every model prediction and Euler state. The
first model call is the strict graph-parity check. Later values also expose the
deterministic trajectory drift caused when backend-specific floating-point
reduction differences are fed into the next denoising step.
