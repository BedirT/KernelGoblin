# Kernel Manifest Contract

Required top-level fields:

- `id`: `<model>/<operation>`, matching its directory.
- `description`: concise operation description.
- `model`: canonical upstream owner/repository.
- `operation`: stable machine-readable operation name.
- `port`: `cuda-to-metal`, `cuda-to-cuda`, or another explicit route.
- `backends`: reference first, then accelerator backends.
- `cmake_preset`: selective root preset.
- `benchmark`: executable path relative to its preset build directory.

Required `[upstream]` fields:

- `repository`: canonical primary repository URL.
- `revision`: full immutable commit SHA, not a branch or tag alone.
- `sources`: exact upstream source paths that define semantics.
- `license`: SPDX-style license identifier when known.

Required `[domain]` fields depend on the operation but must capture relevant
dtype, rank/layout, size, range, alignment, and unsupported inputs.

Required `[verification]` fields must state comparison policy, deterministic
seed where random cases exist, case count, and floating-point tolerances when
applicable.

Update the manifest whenever source revision, contract, build target, or
verification policy changes. A performance-only change does not silently change
the semantic contract.
