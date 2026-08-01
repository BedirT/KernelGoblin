#include <metal_stdlib>
using namespace metal;

struct SubmConv3x3Params {
  ulong weight_offset;
  ulong bias_offset;
  uint token_count;
  uint input_channels;
  uint output_channels;
  uint has_bias;
};

kernel void kg_subm_conv3x3_f16_simdgroup_f32(
    const device float* input [[buffer(0)]],
    const device int* neighbors [[buffer(1)]],
    const device uchar* checkpoint [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SubmConv3x3Params& params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float input_tile[64];
  threadgroup float weight_tile[64];
  threadgroup float contribution_tile[64];
  threadgroup float accumulated_tile[64];
  const uint token_base = group.y * 8;
  const uint output_base = group.x * 8;
  const device half* weights = reinterpret_cast<const device half*>(
      checkpoint + params.weight_offset);
  const device half* bias = reinterpret_cast<const device half*>(
      checkpoint + params.bias_offset);
  for (uint index = lane; index < 64; index += 32) accumulated_tile[index] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = 0; offset < 27; ++offset) {
    simdgroup_float8x8 contribution =
        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint channel_base = 0; channel_base < params.input_channels;
         channel_base += 8) {
      for (uint index = lane; index < 64; index += 32) {
        const uint tile_row = index / 8;
        const uint tile_column = index % 8;
        const uint token = token_base + tile_row;
        const uint channel = channel_base + tile_column;
        const int source = token < params.token_count
            ? neighbors[token * 27 + offset] : -1;
        input_tile[index] = source >= 0 && uint(source) < params.token_count &&
                channel < params.input_channels
            ? input[ulong(source) * ulong(params.input_channels) + ulong(channel)]
            : 0.0f;
        const uint output_channel = output_base + tile_column;
        const uint weight_channel = channel_base + tile_row;
        weight_tile[index] = output_channel < params.output_channels &&
                weight_channel < params.input_channels
            ? float(weights[
                (ulong(output_channel) * 27ul + ulong(offset))
                    * ulong(params.input_channels) + ulong(weight_channel)])
            : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 input_matrix;
      simdgroup_float8x8 weight_matrix;
      simdgroup_load(input_matrix, input_tile, 8);
      simdgroup_load(weight_matrix, weight_tile, 8);
      simdgroup_multiply_accumulate(
          contribution, input_matrix, weight_matrix, contribution);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(contribution, contribution_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < 64; index += 32) {
      accumulated_tile[index] = float(half(
          accumulated_tile[index] + contribution_tile[index]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint index = lane; index < 64; index += 32) {
    const uint token = token_base + index / 8;
    const uint output_channel = output_base + index % 8;
    if (token < params.token_count && output_channel < params.output_channels) {
      const float value = accumulated_tile[index]
          + (params.has_bias != 0 ? float(bias[output_channel]) : 0.0f);
      output[token * params.output_channels + output_channel] = float(half(value));
    }
  }
}
