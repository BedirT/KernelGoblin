#include <metal_stdlib>
using namespace metal;

struct RasterOut {
  float4 clip_position [[position]];
  float3 object_position;
};

struct FragmentOut {
  float4 position [[color(0)]];
  uint face_id [[color(1)]];
};

vertex RasterOut uv_raster_vertex(
    uint vertex_id [[vertex_id]],
    const device float* positions [[buffer(0)]],
    const device float* uvs [[buffer(1)]],
    const device uint* faces [[buffer(2)]]) {
  const uint index = faces[vertex_id];
  const float2 uv = float2(uvs[index * 2], uvs[index * 2 + 1]);
  RasterOut out;
  // Metal readback row zero is the top row. Flip clip Y so row zero is V=0,
  // matching the tensor layout used by the upstream baker.
  out.clip_position = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
  out.object_position = float3(
      positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]);
  return out;
}

fragment FragmentOut uv_raster_fragment(
    RasterOut in [[stage_in]], uint primitive_id [[primitive_id]]) {
  FragmentOut out;
  out.position = float4(in.object_position, 1.0f);
  out.face_id = primitive_id + 1;
  return out;
}
