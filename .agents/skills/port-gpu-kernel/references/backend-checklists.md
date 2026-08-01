# Backend Checklists

## Metal

- Compile `.metal` sources ahead of time to `.air` and `.metallib` for tests.
- Fail clearly when no `MTLDevice`, library function, pipeline, queue, encoder,
  buffer, or command object is available.
- Bind resources at indices identical to the shader signature.
- Derive threadgroup size from pipeline/device limits; guard tail threads.
- Wait for completion before reading results or stopping a host timer.
- Check command-buffer error status after completion.
- State whether shared-buffer allocation and host copies are in the benchmark.
- Use Xcode Metal validation, GPU capture, Instruments, or `xctrace` when a bug
  or optimization warrants profiling.

## CUDA

- Record CUDA toolkit, driver, GPU architecture, framework, and compiler flags.
- Check all runtime API results and `cudaGetLastError()` after launches.
- Synchronize before consuming output and at benchmark timing boundaries.
- Run `compute-sanitizer` for memory/race issues when available.
- Use CUDA events for kernel-only timing; label host end-to-end timers clearly.
- Profile representative shapes with Nsight Compute or Nsight Systems before
  optimizing. Do not optimize a guessed bottleneck.
- Test supported architectures or constrain the manifest/build explicitly.

## CUDA-to-CUDA Optimization

- Preserve the original implementation as a selectable baseline until the new
  kernel passes conformance and performance gates.
- Compare the same inputs, dtype, layout, compiler mode, warmup, and device.
- Measure several representative model shapes, not only the best case.
- Report regressions as well as wins; do not select only favorable samples.
- Re-run model call-site integration because launch configuration and workspace
  changes can be correct in isolation but incompatible in context.
