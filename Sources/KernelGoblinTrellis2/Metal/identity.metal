#include <metal_stdlib>
using namespace metal;

kernel void kg_identity_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = input[index];
}

struct LinearF32Params {
  ulong weight_offset;
  ulong bias_offset;
  uint rows;
  uint input_channels;
  uint output_channels;
  uint has_bias;
};

kernel void kg_linear_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.output_channels;
  if (index >= count) return;
  const uint row = index / params.output_channels;
  const uint output_channel = index % params.output_channels;
  const device float* weight =
      reinterpret_cast<const device float*>(checkpoint + params.weight_offset);
  const device float* bias =
      reinterpret_cast<const device float*>(checkpoint + params.bias_offset);
  float value = params.has_bias ? bias[output_channel] : 0.0f;
  const uint input_base = row * params.input_channels;
  const uint weight_base = output_channel * params.input_channels;
  for (uint channel = 0; channel < params.input_channels; ++channel) {
    value = fma(input[input_base + channel], weight[weight_base + channel], value);
  }
  output[index] = value;
}

inline float kg_bf16_to_f32(ushort value) {
  return as_type<float>(uint(value) << 16);
}

kernel void kg_linear_bf16_weights_f32_output(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.output_channels;
  if (index >= count) return;
  const uint row = index / params.output_channels;
  const uint output_channel = index % params.output_channels;
  const device ushort* weight =
      reinterpret_cast<const device ushort*>(checkpoint + params.weight_offset);
  const device ushort* bias =
      reinterpret_cast<const device ushort*>(checkpoint + params.bias_offset);
  float value = params.has_bias ? kg_bf16_to_f32(bias[output_channel]) : 0.0f;
  const uint input_base = row * params.input_channels;
  const uint weight_base = output_channel * params.input_channels;
  for (uint channel = 0; channel < params.input_channels; ++channel) {
    value = fma(
        input[input_base + channel],
        kg_bf16_to_f32(weight[weight_base + channel]),
        value);
  }
  output[index] = value;
}

struct TimestepEmbeddingParams {
  uint rows;
  uint dimensions;
  float log_max_period;
};

kernel void kg_timestep_embedding_f32(
    const device float* timesteps [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TimestepEmbeddingParams& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.dimensions;
  if (index >= count) return;
  const uint row = index / params.dimensions;
  const uint column = index % params.dimensions;
  const uint half_dimensions = params.dimensions / 2;
  if (column >= half_dimensions * 2) {
    output[index] = 0.0f;
    return;
  }
  const uint frequency_index =
      column < half_dimensions ? column : column - half_dimensions;
  const float frequency = exp(
      -params.log_max_period * float(frequency_index) / float(half_dimensions));
  const float phase = timesteps[row] * frequency;
  output[index] = column < half_dimensions ? cos(phase) : sin(phase);
}

kernel void kg_silu_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = input[index] / (1.0f + exp(-input[index]));
}
