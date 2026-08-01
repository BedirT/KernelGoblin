#include <metal_stdlib>
using namespace metal;

struct SparseSubdivisionParams {
  uint child_count;
  uint input_channels;
  uint output_channels;
};

kernel void kg_channel_to_spatial_f32(
    const device float* input [[buffer(0)]],
    const device uint* parents [[buffer(1)]],
    const device uint* children [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SparseSubdivisionParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(params.child_count) * params.output_channels;
  if (ulong(index) >= count) return;
  const ulong child = ulong(index) / params.output_channels;
  const ulong channel = ulong(index) % params.output_channels;
  const ulong source_row = ulong(parents[child]) * 8 + children[child];
  output[index] = input[source_row * params.output_channels + channel];
}

kernel void kg_channel_to_spatial_skip_f32(
    const device float* input [[buffer(0)]],
    const device uint* parents [[buffer(1)]],
    const device uint* children [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SparseSubdivisionParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(params.child_count) * params.output_channels;
  if (ulong(index) >= count) return;
  const ulong child = ulong(index) / params.output_channels;
  const ulong output_channel = ulong(index) % params.output_channels;
  const ulong child_channels = params.input_channels / 8;
  const ulong repeat = params.output_channels / child_channels;
  const ulong source_channel = output_channel / repeat;
  const ulong source_row = ulong(parents[child]) * 8 + children[child];
  output[index] = input[source_row * child_channels + source_channel];
}
