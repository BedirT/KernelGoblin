#include <metal_stdlib>
using namespace metal;

struct LinearF32Params {
  ulong weight_offset;
  ulong bias_offset;
  uint rows;
  uint input_channels;
  uint output_channels;
  uint has_bias;
};

inline float kg_bf16_to_f32(ushort value) {
  return as_type<float>(uint(value) << 16);
}

constant uint kg_simdgroup_matrix_tile = 8;

kernel void kg_linear_simdgroup_bf16_weights_f32_output(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float input_tile[8 * 8];
  threadgroup float weight_tile[8 * 8];
  threadgroup float output_tile[8 * 8];
  const uint row_base = group.y * kg_simdgroup_matrix_tile;
  const uint output_base = group.x * kg_simdgroup_matrix_tile;
  const device ushort* weight =
      reinterpret_cast<const device ushort*>(checkpoint + params.weight_offset);
  const device ushort* bias =
      reinterpret_cast<const device ushort*>(checkpoint + params.bias_offset);
  simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint channel_base = 0; channel_base < params.input_channels;
       channel_base += kg_simdgroup_matrix_tile) {
    for (uint index = lane; index < 64; index += 32) {
      const uint tile_row = index / kg_simdgroup_matrix_tile;
      const uint tile_column = index % kg_simdgroup_matrix_tile;
      const uint row = row_base + tile_row;
      const uint channel = channel_base + tile_column;
      input_tile[index] = row < params.rows && channel < params.input_channels
          ? input[row * params.input_channels + channel]
          : 0.0f;

      const uint output_channel = output_base + tile_column;
      const uint weight_channel = channel_base + tile_row;
      weight_tile[index] =
          output_channel < params.output_channels &&
                  weight_channel < params.input_channels
              ? kg_bf16_to_f32(
                    weight[output_channel * params.input_channels + weight_channel])
              : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 input_matrix;
    simdgroup_float8x8 weight_matrix;
    simdgroup_load(input_matrix, input_tile, kg_simdgroup_matrix_tile);
    simdgroup_load(weight_matrix, weight_tile, kg_simdgroup_matrix_tile);
    simdgroup_multiply_accumulate(
        accumulator, input_matrix, weight_matrix, accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(accumulator, output_tile, kg_simdgroup_matrix_tile);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < 64; index += 32) {
    const uint tile_row = index / kg_simdgroup_matrix_tile;
    const uint tile_column = index % kg_simdgroup_matrix_tile;
    const uint row = row_base + tile_row;
    const uint output_channel = output_base + tile_column;
    if (row < params.rows && output_channel < params.output_channels) {
      const float bias_value = params.has_bias
          ? kg_bf16_to_f32(bias[output_channel])
          : 0.0f;
      output[row * params.output_channels + output_channel] =
          output_tile[index] + bias_value;
    }
  }
}
