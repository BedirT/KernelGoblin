# TRELLIS.2 UV Raster

This kernel replaces the UV-space `nvdiffrast` raster/interpolation step used
by TRELLIS.2's PBR texture baker. It uses Metal's fixed-function triangle
rasterizer and emits interpolated object-space positions plus one-based face
IDs into offscreen textures.

The implementation is independent Metal code based on TRELLIS.2's public
input/output contract. It does not contain or redistribute nvdiffrast source.

```sh
./kg test trellis2/uv_raster
./kg benchmark trellis2/uv_raster
```

Tests execute the physical Metal render pipeline and compare it with a CPU
pixel-center oracle on analytic triangles, shared edges, winding reversal,
degenerate triangles, and invalid inputs. The benchmark is correctness-gated,
then includes host allocation, upload, render-target allocation, draw,
synchronized readback, and result copies in each timed iteration.

This proves the independent Metal implementation against its analytic
contract. Exact parity with pinned nvdiffrast v0.4.0 coverage and overlap rules
still needs CUDA-captured golden fixtures, so it is not claimed yet.

Verified scope is UV-space position interpolation and face coverage on the
named Apple GPU. UV chart generation, mesh simplification, closest-surface
projection, sparse PBR sampling, and GLB material assembly are separate stages.
