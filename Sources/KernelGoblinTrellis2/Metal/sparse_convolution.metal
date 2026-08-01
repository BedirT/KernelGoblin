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

kernel void kg_subm_conv3x3_f16_scalar_f32(
    const device float* input [[buffer(0)]],
    const device int* neighbors [[buffer(1)]],
    const device uchar* checkpoint [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SubmConv3x3Params& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const ulong count = ulong(params.token_count) * ulong(params.output_channels);
  if (ulong(index) >= count) return;
  const uint token = index / params.output_channels;
  const uint output_channel = index % params.output_channels;
  const device half* weights = reinterpret_cast<const device half*>(
      checkpoint + params.weight_offset);
  const device half* bias = reinterpret_cast<const device half*>(
      checkpoint + params.bias_offset);
  half accumulated = half(0.0f);
  for (uint offset = 0; offset < 27; ++offset) {
    const int source = neighbors[token * 27 + offset];
    if (source < 0 || uint(source) >= params.token_count) continue;
    const ulong weight_base =
        (ulong(output_channel) * 27ul + ulong(offset)) * ulong(params.input_channels);
    const ulong input_base = ulong(source) * ulong(params.input_channels);
    float contribution = 0.0f;
    for (uint channel = 0; channel < params.input_channels; ++channel) {
      contribution = fma(input[input_base + ulong(channel)],
                         float(weights[weight_base + ulong(channel)]), contribution);
    }
    accumulated = half(float(accumulated) + contribution);
  }
  const float value = float(accumulated)
      + (params.has_bias != 0 ? float(bias[output_channel]) : 0.0f);
  output[index] = float(half(value));
}
