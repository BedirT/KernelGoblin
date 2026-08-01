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

struct LayerNormParams {
  ulong weight_offset;
  ulong bias_offset;
  uint rows;
  uint channels;
  uint has_affine;
  float epsilon;
};

kernel void kg_layer_norm_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LayerNormParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows) return;
  const uint base = row * params.channels;
  float mean = 0.0f;
  for (uint channel = 0; channel < params.channels; ++channel) {
    mean += input[base + channel];
  }
  mean /= float(params.channels);
  float variance = 0.0f;
  for (uint channel = 0; channel < params.channels; ++channel) {
    const float centered = input[base + channel] - mean;
    variance = fma(centered, centered, variance);
  }
  variance /= float(params.channels);
  const float inverse_std = rsqrt(variance + params.epsilon);
  const device ushort* weight =
      reinterpret_cast<const device ushort*>(checkpoint + params.weight_offset);
  const device ushort* bias =
      reinterpret_cast<const device ushort*>(checkpoint + params.bias_offset);
  for (uint channel = 0; channel < params.channels; ++channel) {
    float value = (input[base + channel] - mean) * inverse_std;
    if (params.has_affine) {
      value = fma(value, kg_bf16_to_f32(weight[channel]), kg_bf16_to_f32(bias[channel]));
    }
    output[base + channel] = value;
  }
}

struct RMSNormParams {
  ulong gamma_offset;
  uint rows;
  uint heads;
  uint dimensions;
  float epsilon;
};

kernel void kg_multihead_rms_norm_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint group_count = params.rows * params.heads;
  if (index >= group_count) return;
  const uint head = index % params.heads;
  const uint base = index * params.dimensions;
  float squared_sum = 0.0f;
  for (uint dimension = 0; dimension < params.dimensions; ++dimension) {
    squared_sum = fma(input[base + dimension], input[base + dimension], squared_sum);
  }
  const float inverse_norm = 1.0f / max(sqrt(squared_sum), params.epsilon);
  const float scale = sqrt(float(params.dimensions));
  const device ushort* gamma =
      reinterpret_cast<const device ushort*>(checkpoint + params.gamma_offset);
  const uint gamma_base = head * params.dimensions;
  for (uint dimension = 0; dimension < params.dimensions; ++dimension) {
    output[base + dimension] = input[base + dimension] * inverse_norm *
        kg_bf16_to_f32(gamma[gamma_base + dimension]) * scale;
  }
}

struct RoPE3DParams {
  uint tokens;
  uint heads;
  uint dimensions;
  uint frequency_dimensions;
  float minimum_frequency;
  float maximum_frequency;
};

kernel void kg_rope3d_f32(
    const device float* input [[buffer(0)]],
    const device int* coordinates [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RoPE3DParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.heads * params.dimensions;
  if (index >= count) return;
  const uint dimension = index % params.dimensions;
  const uint pair = dimension / 2;
  const uint spatial_pairs = 3 * params.frequency_dimensions;
  if (pair >= spatial_pairs) {
    output[index] = input[index];
    return;
  }
  const uint token = index / (params.heads * params.dimensions);
  const uint axis = pair / params.frequency_dimensions;
  const uint frequency_index = pair % params.frequency_dimensions;
  const float frequency = params.minimum_frequency / pow(
      params.maximum_frequency,
      float(frequency_index) / float(params.frequency_dimensions));
  const float angle = float(coordinates[token * 4 + axis + 1]) * frequency;
  const float cosine = cos(angle);
  const float sine = sin(angle);
  const uint pair_base = index - dimension + pair * 2;
  const float real = input[pair_base];
  const float imaginary = input[pair_base + 1];
  output[index] = (dimension & 1u) == 0u
      ? fma(-imaginary, sine, real * cosine)
      : fma(real, sine, imaginary * cosine);
}

struct AttentionParams {
  uint query_count;
  uint key_count;
  uint heads;
  uint dimensions;
  float scale;
};

kernel void kg_fused_attention_f32(
    const device float* queries [[buffer(0)]],
    const device float* keys [[buffer(1)]],
    const device float* values [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint dimension [[thread_index_in_threadgroup]]) {
  const uint query = group / params.heads;
  const uint head = group % params.heads;
  if (query >= params.query_count || dimension >= params.dimensions) return;

  threadgroup float reduction[256];
  threadgroup float running_max;
  threadgroup float running_sum;
  threadgroup float previous_scale;
  threadgroup float current_scale;
  if (dimension == 0) {
    running_max = -INFINITY;
    running_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint query_base = (query * params.heads + head) * params.dimensions;
  float accumulator = 0.0f;
  for (uint key = 0; key < params.key_count; ++key) {
    const uint key_base = (key * params.heads + head) * params.dimensions;
    reduction[dimension] = queries[query_base + dimension] * keys[key_base + dimension];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = params.dimensions / 2; stride > 0; stride >>= 1) {
      if (dimension < stride) reduction[dimension] += reduction[dimension + stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (dimension == 0) {
      const float score = reduction[0] * params.scale;
      const float next_max = max(running_max, score);
      previous_scale = exp(running_max - next_max);
      current_scale = exp(score - next_max);
      running_sum = running_sum * previous_scale + current_scale;
      running_max = next_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator = accumulator * previous_scale +
        current_scale * values[key_base + dimension];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  output[query_base + dimension] = accumulator / running_sum;
}

inline float kg_round_f32_to_bf16_value(float value) {
  uint bits = as_type<uint>(value);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return as_type<float>(bits & 0xFFFF0000u);
}

kernel void kg_round_bf16_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = kg_round_f32_to_bf16_value(input[index]);
}

struct SplitQKVParams {
  uint rows;
  uint channels;
};

kernel void kg_split_qkv_f32(
    const device float* input [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* key [[buffer(2)]],
    device float* value [[buffer(3)]],
    constant SplitQKVParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.channels;
  if (index >= count) return;
  const uint row = index / params.channels;
  const uint channel = index % params.channels;
  const uint input_base = row * params.channels * 3 + channel;
  query[index] = input[input_base];
  key[index] = input[input_base + params.channels];
  value[index] = input[input_base + params.channels * 2];
}

kernel void kg_split_kv_f32(
    const device float* input [[buffer(0)]],
    device float* key [[buffer(1)]],
    device float* value [[buffer(2)]],
    constant SplitQKVParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.channels;
  if (index >= count) return;
  const uint row = index / params.channels;
  const uint channel = index % params.channels;
  const uint input_base = row * params.channels * 2 + channel;
  key[index] = input[input_base];
  value[index] = input[input_base + params.channels];
}

kernel void kg_gelu_tanh_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const float x = input[index];
  if (x >= 10.0f) {
    output[index] = x;
    return;
  }
  if (x <= -10.0f) {
    output[index] = 0.0f;
    return;
  }
  constexpr float coefficient = 0.7978845608028654f;
  output[index] = 0.5f * x *
      (1.0f + tanh(coefficient * (x + 0.044715f * x * x * x)));
}

struct ModulateParams {
  uint rows;
  uint channels;
  uint shift_offset;
  uint scale_offset;
};

kernel void kg_modulate_f32(
    const device float* input [[buffer(0)]],
    const device float* modulation [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ModulateParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.channels;
  if (index >= count) return;
  const uint channel = index % params.channels;
  output[index] = fma(
      input[index], 1.0f + modulation[params.scale_offset + channel],
      modulation[params.shift_offset + channel]);
}

kernel void kg_modulate_bf16_f32(
    const device float* input [[buffer(0)]],
    const device float* modulation [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ModulateParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.rows * params.channels;
  if (index >= count) return;
  const uint channel = index % params.channels;
  const float scale = kg_round_f32_to_bf16_value(
      1.0f + modulation[params.scale_offset + channel]);
  const float product = kg_round_f32_to_bf16_value(input[index] * scale);
  output[index] = kg_round_f32_to_bf16_value(
      product + modulation[params.shift_offset + channel]);
}

struct ResidualParams {
  uint count;
  uint channels;
  uint gate_offset;
  uint has_gate;
};

kernel void kg_residual_f32(
    const device float* residual [[buffer(0)]],
    const device float* branch [[buffer(1)]],
    const device float* modulation [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ResidualParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.count) return;
  const float gate = params.has_gate
      ? modulation[params.gate_offset + index % params.channels] : 1.0f;
  output[index] = fma(branch[index], gate, residual[index]);
}

kernel void kg_residual_bf16_f32(
    const device float* residual [[buffer(0)]],
    const device float* branch [[buffer(1)]],
    const device float* modulation [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ResidualParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.count) return;
  float value = branch[index];
  if (params.has_gate) {
    const float gate = modulation[params.gate_offset + index % params.channels];
    value = kg_round_f32_to_bf16_value(value * gate);
  }
  output[index] = kg_round_f32_to_bf16_value(residual[index] + value);
}

kernel void kg_add_f32(
    const device float* lhs [[buffer(0)]],
    const device float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = lhs[index] + rhs[index];
}

struct AddCheckpointBF16Params {
  ulong offset;
  uint count;
};

kernel void kg_add_checkpoint_bf16_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant AddCheckpointBF16Params& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.count) return;
  const device ushort* values =
      reinterpret_cast<const device ushort*>(checkpoint + params.offset);
  output[index] = input[index] + kg_bf16_to_f32(values[index]);
}
