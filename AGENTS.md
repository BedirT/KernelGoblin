# KernelGoblin Agent Guide

## Mission

Build trustworthy, reusable ports and optimizations of GPU kernels used by real
model repositories. Preserve upstream behavior first; optimize only after a
correct accelerator implementation is proven.

## Start Here

1. Read `README.md` and the target kernel's `kernel.toml` and `README.md`.
2. Run `./kg doctor`, `./kg list`, and `git status --short --branch`.
3. For a port or optimization, use the repo skill `$port-gpu-kernel` from
   `.agents/skills/port-gpu-kernel/SKILL.md`.
4. Keep upstream research pinned to an exact revision and primary sources.

## Repository Map

- `kg`, `tools/kg.py`: dependency-free kernel selection and validation CLI.
- `kernels/<model>/<operation>/`: one independently buildable kernel port.
- `ports/<model>/`: isolated full-model compatibility runtime and call-site tests.
- `.agents/skills/`: reusable Codex workflows checked into this repository.
- `.codex/agents/`: optional read-only specialists for explicitly requested
  parallel research or review.
- `docs/`: toolchain and agent-harness decisions.

## Commands

```sh
./kg doctor
./kg list
./kg validate
./kg setup trellis2/z_order
./kg test trellis2/z_order
./kg benchmark trellis2/z_order
./kg model setup trellis2
./kg model test trellis2
./kg model run trellis2 --input image.png --output build/trellis2/output
./kg model run trellis2 --pipeline-type 1024_cascade \
  --input image.png --output build/trellis2/output-1024
```

Manual equivalents are documented in `README.md`. Build only the requested
kernel; do not install every model's dependencies globally.

## Kernel Contract

Every kernel must include:

- A `kernel.toml` with stable ID, operation, backends, license, exact upstream
  repository/revision/source paths, input domain, and comparison policy.
- A known-good CPU reference or immutable golden fixtures derived from upstream.
- Accelerator tests covering edge cases, representative shapes, and randomized
  differential comparisons.
- An integration test that dispatches the real backend, not an emulation or
  compile-only proxy.
- A benchmark that checks correctness before timing and reports hardware,
  workload, warmup, iteration count, synchronization, and timing boundaries.
- A kernel README stating what was and was not verified.

For floating-point ports, derive tolerances from the algorithm and dtype. Never
loosen a tolerance merely to make a failing port pass. Integer kernels should
normally be bit-exact.

## Engineering Rules

- Preserve the original algorithm until conformance passes; tune in a separate
  change so correctness and performance regressions are attributable.
- Keep the TRELLIS.2 production runtime in `Sources/KernelGoblinTrellis2*`
  Swift + Metal only. Python, PyTorch, LibTorch, and MPS framework code belong
  under `ports/trellis2/` as optional reference/oracle tooling and must never be
  imported or launched by a native setup, test, or run command.
- Use native backend toolchains and check every host API or launch error.
- Separate backend-independent interfaces from Metal, CUDA, and CPU sources.
- Reuse the root CMake/CTest and `kg` patterns. Add dependencies only to the
  selected kernel and explain why native or existing tools were insufficient.
- Keep generated files under `build/`; never commit model weights or build
  products.
- Preserve unrelated worktree changes. Do not commit, push, download weights,
  accept licenses, or spend cloud GPU money unless the user asks.

## Verification Ladder

Run the strongest applicable checks in order:

1. `./kg validate`
2. Focused CPU/reference unit tests
3. Real accelerator differential and round-trip tests
4. Debug/runtime diagnostics (`compute-sanitizer` for CUDA when available;
   Metal validation or Xcode GPU tools for Metal when applicable)
5. Representative benchmark with explicit synchronization
6. Model call-site integration, then full model smoke test only when hardware,
   weights, licenses, and dependencies are available

Report the exact highest completed rung. Kernel success is not model success.

## Agent Collaboration

Use subagents only when the user explicitly requests delegation or parallel
work. Prefer the project roles in `.codex/agents/` for bounded read-only jobs:

- `upstream_researcher`: source, license, call-site, and semantic inventory.
- `kernel_reviewer`: correctness, memory safety, and conformance review.
- `benchmark_reviewer`: timing methodology and performance-claim review.

The primary agent owns edits, integration, final verification, and closeout.
Do not let multiple agents edit the same kernel concurrently.

## Code Review Rules

Flag as high priority:

- Tests that never execute the claimed accelerator backend.
- Missing synchronization, warmup, correctness checks, or timing-boundary
  disclosure in benchmarks.
- Behavior drift from pinned upstream semantics, including shape, dtype,
  overflow, layout, or error handling.
- Out-of-bounds access, unchecked launch/API failures, races, or backend lifetime
  bugs.
- Claims that generalize one device, workload, or isolated kernel result to a
  full model or other hardware.

## Definition of Done

The selected kernel builds from a clean configuration, all relevant tests pass
on the named physical backend, benchmark output is reproducible, provenance and
scope are documented, `./kg validate` passes, and the final report distinguishes
verified facts from unavailable hardware or model-level work.
