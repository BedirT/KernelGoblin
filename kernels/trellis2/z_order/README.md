# TRELLIS.2 O-Voxel Z-order

This is a direct CUDA-to-Metal translation of the O-Voxel serialization
kernel pinned in `kernel.toml`. It interleaves three 10-bit unsigned coordinates
into a 30-bit Morton code and performs the inverse operation.

The public C++ API exposes a CPU reference and a Metal implementation. The Metal
host wrapper compiles no shader at runtime: CMake invokes Apple's `metal` and
`metallib` tools, then the executable loads the pinned build artifact.

Run from the repository root:

```sh
./kg test trellis2/z_order
./kg benchmark trellis2/z_order
```

Inputs outside `[0, 1023]` are outside the upstream kernel's documented bit
domain; the bit-expansion operation discards higher coordinate bits.
