---
name: port-gpu-kernel
description: Port, optimize, integrate, or verify custom GPU kernels in KernelGoblin across CUDA, Metal, and CPU references. Use for new model-kernel intake, CUDA-to-Metal ports, CUDA-to-CUDA optimizations, backend correctness debugging, conformance test design, or GPU benchmark work. Do not use for general model inference changes that do not touch a custom kernel.
---

# Port A GPU Kernel

## Workflow

1. Read root `AGENTS.md`, `README.md`, and `docs/TOOLS.md`.
2. Inspect the upstream repository and pin the exact revision before translating
   code. Record license and source paths in `kernel.toml`.
3. Trace real call sites and capture the operation contract: shapes, strides,
   layouts, dtypes, devices, value ranges, errors, and edge cases. Do not infer
   missing behavior from the kernel name.
4. Select one bounded operation. Keep CPU/reference, accelerator host wrapper,
   device code, tests, and benchmark inside `kernels/<model>/<operation>/`.
5. Implement or preserve a known-good reference before the new backend. For an
   optimization, record current correctness and baseline timing first.
6. Translate for behavior before tuning. Check every backend API result and
   dispatch error. Preserve arithmetic and overflow semantics intentionally.
7. Build a differential suite covering empty/minimal inputs, boundaries,
   representative model shapes, deterministic randomized inputs, and invalid
   host inputs. Use bit-exact comparison for integers and justified tolerances
   for floating point.
8. Dispatch the physical backend in tests. A compiled shader, CPU fallback, or
   mocked device is not accelerator verification.
9. Benchmark only after correctness passes. Warm up, synchronize, disclose
   whether allocation/transfers are timed, report device and workload, and
   avoid cross-device speed claims from one machine.
10. Run `./kg validate`, focused tests, runtime diagnostics when available, and
    the benchmark. Attempt model integration only when the required hardware,
    dependencies, weights, and licenses are genuinely available.
11. Report exactly what passed and what remains. Never turn “kernel verified”
    into “model verified.”

## References

- Read `references/manifest-contract.md` when adding or changing a kernel.
- Read `references/backend-checklists.md` for Metal/CUDA implementation and
  profiling gates.

## Delegation

When the user explicitly asks for parallel agents, use the project specialists
for independent read-only research or review. Keep edits and final integration
with the primary agent.
