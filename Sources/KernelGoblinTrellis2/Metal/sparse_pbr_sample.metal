#include <metal_stdlib>
using namespace metal;

struct SparsePBRLookupEntry {
  ulong key;
  uint row;
  uint padding;
};

struct SparsePBRSampleParams {
  uint query_count;
  uint channels;
  uint width;
  uint height;
  uint depth;
  uint entry_count;
  float origin_x;
  float origin_y;
  float origin_z;
  float inverse_voxel_x;
  float inverse_voxel_y;
  float inverse_voxel_z;
  uint has_mask;
};

static long kg_sparse_pbr_lookup(
    const device SparsePBRLookupEntry* entries,
    uint count,
    ulong key) {
  uint lower = 0;
  uint upper = count;
  while (lower < upper) {
    const uint middle = lower + (upper - lower) / 2;
    if (entries[middle].key < key) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower < count && entries[lower].key == key
      ? long(entries[lower].row) : -1;
}

kernel void kg_sparse_pbr_trilinear_f32(
    const device float* features [[buffer(0)]],
    const device SparsePBRLookupEntry* entries [[buffer(1)]],
    const device packed_float3* positions [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SparsePBRSampleParams& params [[buffer(4)]],
    const device uchar* mask [[buffer(5)]],
    uint query_index [[thread_position_in_grid]]) {
  if (query_index >= params.query_count) return;
  const ulong output_base = ulong(query_index) * params.channels;
  if (params.has_mask != 0 && mask[query_index] == 0) {
    for (uint channel = 0; channel < params.channels; ++channel) {
      output[output_base + channel] = 0.0f;
    }
    return;
  }
  const float3 object_position = float3(positions[query_index]);
  const float3 point = (object_position - float3(
      params.origin_x, params.origin_y, params.origin_z)) * float3(
      params.inverse_voxel_x, params.inverse_voxel_y, params.inverse_voxel_z);
  if (!all(isfinite(point))) {
    const float nonfinite = as_type<float>(0x7fc00000u);
    for (uint channel = 0; channel < params.channels; ++channel) {
      output[output_base + channel] = nonfinite;
    }
    return;
  }
  if (point.x <= -0.5f || point.y <= -0.5f || point.z <= -0.5f ||
      point.x >= float(params.width) + 0.5f ||
      point.y >= float(params.height) + 0.5f ||
      point.z >= float(params.depth) + 0.5f) {
    for (uint channel = 0; channel < params.channels; ++channel) {
      output[output_base + channel] = 0.0f;
    }
    return;
  }
  const int3 base = int3(floor(point - 0.5f));
  float denominator = 0.0f;
  float accumulators[16];
  for (uint channel = 0; channel < params.channels; ++channel) {
    accumulators[channel] = 0.0f;
  }
  for (int dx = 0; dx <= 1; ++dx) {
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dz = 0; dz <= 1; ++dz) {
        const int3 coordinate = base + int3(dx, dy, dz);
        if (any(coordinate < int3(0)) ||
            coordinate.x >= int(params.width) ||
            coordinate.y >= int(params.height) ||
            coordinate.z >= int(params.depth)) continue;
        const ulong key = (ulong(coordinate.x) * params.height
            + ulong(coordinate.y)) * params.depth + ulong(coordinate.z);
        const long row = kg_sparse_pbr_lookup(entries, params.entry_count, key);
        if (row < 0) continue;
        const float3 delta = abs(point - float3(coordinate) - 0.5f);
        const float weight = (1.0f - delta.x) *
            (1.0f - delta.y) * (1.0f - delta.z);
        denominator += weight;
        const ulong feature_base = ulong(row) * params.channels;
        for (uint channel = 0; channel < params.channels; ++channel) {
          accumulators[channel] += features[feature_base + channel] * weight;
        }
      }
    }
  }
  denominator = max(denominator, 1.0e-12f);
  for (uint channel = 0; channel < params.channels; ++channel) {
    output[output_base + channel] = accumulators[channel] / denominator;
  }
}
