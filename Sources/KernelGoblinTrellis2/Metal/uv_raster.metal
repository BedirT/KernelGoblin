#include <metal_stdlib>
using namespace metal;

struct UVRasterVertexOut {
  float4 clip_position [[position]];
  float3 object_position;
};

struct UVRasterFragmentOut {
  float4 position [[color(0)]];
  uint face_id [[color(1)]];
};

vertex UVRasterVertexOut kg_uv_raster_vertex(
    uint vertex_id [[vertex_id]],
    const device float* positions [[buffer(0)]],
    const device float* uvs [[buffer(1)]],
    const device uint* faces [[buffer(2)]]) {
  const uint index = faces[vertex_id];
  const float2 uv = float2(uvs[ulong(index) * 2], uvs[ulong(index) * 2 + 1]);
  UVRasterVertexOut out;
  out.clip_position = float4(
      uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
  out.object_position = float3(
      positions[ulong(index) * 3], positions[ulong(index) * 3 + 1],
      positions[ulong(index) * 3 + 2]);
  return out;
}

fragment UVRasterFragmentOut kg_uv_raster_fragment(
    UVRasterVertexOut in [[stage_in]], uint primitive_id [[primitive_id]]) {
  UVRasterFragmentOut out;
  out.position = float4(in.object_position, 1.0f);
  out.face_id = primitive_id + 1;
  return out;
}

kernel void kg_uv_raster_compact(
    texture2d<float, access::read> position_texture [[texture(0)]],
    texture2d<uint, access::read> id_texture [[texture(1)]],
    device float* positions [[buffer(0)]],
    device uint* face_ids [[buffer(1)]],
    device uchar* mask [[buffer(2)]],
    uint2 pixel [[thread_position_in_grid]]) {
  if (pixel.x >= position_texture.get_width() ||
      pixel.y >= position_texture.get_height()) return;
  const ulong index = ulong(pixel.y) * position_texture.get_width() + pixel.x;
  const float4 position = position_texture.read(pixel);
  const uint face_id = id_texture.read(pixel).x;
  positions[index * 3] = position.x;
  positions[index * 3 + 1] = position.y;
  positions[index * 3 + 2] = position.z;
  face_ids[index] = face_id;
  mask[index] = face_id == 0 ? 0 : 1;
}
