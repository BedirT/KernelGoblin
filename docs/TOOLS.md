# Toolchain

## Core

- Git for source provenance and patch review.
- CMake 3.25+ and Ninja for selective native builds.
- A C++17 compiler.
- Python 3.11+ standard library for `kg` and TOML validation. No virtual
  environment is needed for the current kernel.

Run `./kg doctor` to locate required executables. Kernel-specific dependencies
must remain optional and be installed only when that kernel is selected.

## Metal kernels

- macOS and Xcode or Xcode Command Line Tools.
- `xcrun metal` and `xcrun metallib` for ahead-of-time shader compilation.
- Metal and Foundation frameworks for host dispatch.
- Optional diagnostics: Xcode Metal API validation/GPU capture, Instruments,
  and `xctrace`.

Apple's compute workflow is device -> command queue -> command buffer -> compute
encoder -> pipeline/resources -> dispatch -> completion. Host wrappers must
check creation and completion failures rather than treating compilation as
proof of execution.

## CUDA kernels

Install per CUDA-capable machine or container, not on Metal-only hosts:

- NVIDIA driver and a project-pinned CUDA Toolkit (`nvcc`).
- CMake/Ninja and the framework version required by the target model.
- `compute-sanitizer` for memory and race checks.
- Nsight Compute for kernel profiling and Nsight Systems for end-to-end traces.

Add PyTorch, Triton, CUTLASS, or model packages only when a kernel manifest and
call-site integration actually require them. Prefer isolated environments or
containers and pin versions; do not make them global harness requirements.

## Source guidance

- [Apple Metal compute encoder documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [OpenAI Codex skills catalog](https://github.com/openai/skills)
