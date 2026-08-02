# TRELLIS.2 Native Geometry Quality Gap

This is a failure report, not an acceptance record. The native runtime creates
a reloadable GLB, but the current 512 mesh path does not yet reproduce the
topology quality of the pinned upstream CUDA export.

## Reproduced Artifact

The user generated the following geometry-only artifact from `test.png`:

```sh
./kg generate trellis2 test.png --geometry-only --steps 30
```

The ignored evidence file records commit-era native Swift + Metal execution,
seed 42, 30 steps, 2,247 pooled sparse coordinates, 726,637 decoded vertices,
and 1,455,272 faces. Its seven stage timers sum to 509.097 seconds. This is not
a controlled benchmark: process launch, RSS, thermals, and repeated samples
were not recorded.

Running more than the checkpoint default of 12 Euler steps did not close the
visible holes. That result is expected once the defects are separated by
stage: sampling steps alter the denoised latent, but do not increase the 512
spatial tier and do not run missing topology reconstruction.

## Measured Topology

A read-only topology audit of the generated GLB found:

| Metric | Result |
| --- | ---: |
| Faces | 1,455,272 |
| Vertices | 726,637 |
| Boundary edges | 45,856 |
| Boundary-connected components | 9,290 |
| Clean closed boundary loops | 2,071 |
| Six-edge closed loops | 1,902 |
| Over-connected non-manifold edges | 39,759 |
| Maximum edge incidence | 4 |
| Vertex-connected components | 1,028 |
| Watertight | No |
| Winding consistent | No |

An independent Blender import corroborated the result after its own GLB
welding: 45,876 boundary edges, 9,291 boundary components, and 85,630 edges
reported as non-manifold under Blender's broader definition.

The existing GLB reload check validates buffers, indices, counts, and material
structure. It does not measure boundaries, manifoldness, winding, connected
components, or silhouette quality. Consequently, reload success was real but
insufficient as a model-quality gate.

## Why The Current Port Has Holes

### 1. Native cleanup is not the upstream CUDA cleanup

Pinned upstream first calls `Mesh.fill_holes(max_hole_perimeter=3e-2)`, which
discovers CuMesh boundary loops and fills them by physical perimeter. The
official example then calls `o_voxel.postprocess.to_glb(..., remesh=True)`,
which reconstructs topology with narrow-band dual contouring before
simplification and export.

The native path calls `MeshHoleFiller.fillTriangleAndQuadHoles` once and writes
the result directly. It intentionally ignores every simple boundary with five
or more vertices and abandons branched boundaries. The existing oracle fixture
uses Trimesh, not the CuMesh operation reached by the pinned upstream call
site, so it proved only triangle/quad compatibility. This is the clearest
direct cause of the thousands of holes that remain in the GLB.

### 2. Coarse geometry support already drifts before mesh extraction

The committed production sparse-structure fixture records 67,776 native
occupied voxels against 91,584 upstream voxels, a 0.7393 count ratio and 0.7189
IoU for the selected mixed-precision trajectory. Thresholding, coordinate
order, and 2x max pooling match upstream, which points to accumulated flow
trajectory drift rather than an indexing transcription error.

Missing occupancy becomes missing shape-flow coordinates. No amount of later
mesh cleanup can recover semantic surface support that never reaches the shape
model.

### 3. The friendly command runs 512, while upstream defaults to 1024 cascade

The pinned Microsoft checkpoint config names `1024_cascade` as its default.
That path runs the 512 shape stage, upsamples its support, and performs another
1024 shape-flow pass with as many as 49,152 tokens. The native friendly command
currently runs only `generate512`.

Thirty sampler steps at 512 are therefore not a higher-quality substitute for
the upstream default pipeline. They cost roughly 2.5 times as many Euler
iterations as 12 steps while leaving the spatial tier unchanged.

### 4. Production-scale shape geometry is not yet oracle-verified

The strongest committed shape-flow and full-decoder fixtures are deliberately
tiny. They prove graph wiring and local numerical behavior, but there is no
same-input production-token shape trajectory, raw decoded-field comparison,
or native/upstream mesh differential. The current evidence cannot rule out
additional topology drift in the free-running shape flow or large decoder.

### 5. Foreground preprocessing is a separate quality variable

This run used the Apple Vision person fallback, while pinned upstream uses
BiRefNet for opaque inputs. The native mask was visually checked, but a pinned
alpha matte remains the correct control when comparing model kernels. Mask
differences can change silhouette support; they do not explain why the native
export leaves thousands of measurable boundary loops after decoding.

## Corrective Order

1. Add pre/post-export topology evidence and make quality acceptance reject
   pathological boundary, manifold, winding, and component counts.
2. Pin the exact CuMesh revision reached by upstream setup and capture raw
   decoded-mesh plus repaired/remeshed topology fixtures.
3. Port perimeter-bounded hole filling, duplicate/non-manifold repair, small
   component removal, face orientation, simplification, and narrow-band remesh
   to Swift + Metal.
4. Capture a same-input production-token shape trajectory and decoded-head to
   mesh differential. Fix free-running drift rather than widening tolerances.
5. Implement the native 1024 cascade before comparing the friendly command to
   Microsoft's default CUDA output.
6. Re-benchmark only after the topology and semantic quality gates pass. Use 12
   steps for checkpoint-default comparisons; extra steps are an experiment,
   not a repair mechanism.

Until those gates pass, the honest claim is: the 512 native graph executes and
exports a reloadable artifact, but upstream-default geometry quality is not yet
verified.
