#include <metal_stdlib>
using namespace metal;

struct SparseSpatialToChannelParams {
  uint coarse_count;
  uint input_channels;
  uint output_channels;
};

kernel void kg_spatial_to_channel_pack_f32(
    const device float* input [[buffer(0)]],
    const device int* source_indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SparseSpatialToChannelParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(params.coarse_count) * params.output_channels;
  if (ulong(index) >= count) return;
  const ulong coarse = ulong(index) / params.output_channels;
  const ulong packed_channel = ulong(index) % params.output_channels;
  const ulong child = packed_channel / params.input_channels;
  const ulong channel = packed_channel % params.input_channels;
  const int source = source_indices[coarse * 8 + child];
  output[index] = source < 0 ? 0.0f
      : input[ulong(source) * params.input_channels + channel];
}

kernel void kg_spatial_to_channel_skip_f32(
    const device float* input [[buffer(0)]],
    const device int* source_indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SparseSpatialToChannelParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(params.coarse_count) * params.output_channels;
  if (ulong(index) >= count) return;
  const ulong coarse = ulong(index) / params.output_channels;
  const ulong output_channel = ulong(index) % params.output_channels;
  const ulong packed_channels = ulong(params.input_channels) * 8;
  const ulong reduction = packed_channels / params.output_channels;
  const ulong packed_start = output_channel * reduction;
  float sum = 0.0f;
  for (ulong offset = 0; offset < reduction; ++offset) {
    const ulong packed_channel = packed_start + offset;
    const ulong child = packed_channel / params.input_channels;
    const ulong channel = packed_channel % params.input_channels;
    const int source = source_indices[coarse * 8 + child];
    if (source >= 0) {
      sum += input[ulong(source) * params.input_channels + channel];
    }
  }
  output[index] = sum / float(reduction);
}

kernel void kg_sparse_posterior_mean_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(rows) * 32;
  if (ulong(index) >= count) return;
  const ulong row = ulong(index) / 32;
  const ulong channel = ulong(index) % 32;
  output[index] = input[row * 64 + channel];
}
