# TRELLIS.2 Development Oracles

This directory is not part of the native runtime. It contains small tools for
capturing immutable behavior from the pinned upstream Torch implementation so
the Swift + Metal port can be tested without importing Torch.

`export_slat_block_fixture.py` loads only block 0 from the authenticated
`shape_slat_flow_model_512` checkpoint, runs one deterministic BF16 token, and
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

