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

constant uint kg_linear_tile = 16;

kernel void kg_linear_tiled_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]) {
  threadgroup float input_tile[16 * 16];
  threadgroup float weight_tile[16 * 16];
  const uint row = group.y * kg_linear_tile + local.y;
  const uint output_channel = group.x * kg_linear_tile + local.x;
  const device float* weight =
      reinterpret_cast<const device float*>(checkpoint + params.weight_offset);
  const device float* bias =
      reinterpret_cast<const device float*>(checkpoint + params.bias_offset);
  float value = output_channel < params.output_channels && params.has_bias
      ? bias[output_channel]
      : 0.0f;

  for (uint channel_base = 0; channel_base < params.input_channels;
       channel_base += kg_linear_tile) {
    const uint channel = channel_base + local.x;
    input_tile[local.y * kg_linear_tile + local.x] =
        row < params.rows && channel < params.input_channels
            ? input[row * params.input_channels + channel]
            : 0.0f;
    const uint tile_output_channel = group.x * kg_linear_tile + local.y;
    weight_tile[local.y * kg_linear_tile + local.x] =
        tile_output_channel < params.output_channels && channel < params.input_channels
            ? weight[tile_output_channel * params.input_channels + channel]
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < params.rows && output_channel < params.output_channels) {
      for (uint tile_channel = 0; tile_channel < kg_linear_tile; ++tile_channel) {
        value = fma(
            input_tile[local.y * kg_linear_tile + tile_channel],
            weight_tile[local.x * kg_linear_tile + tile_channel],
            value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < params.rows && output_channel < params.output_channels) {
    output[row * params.output_channels + output_channel] = value;
  }
}

kernel void kg_linear_tiled_bf16_weights_f32_output(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]) {
  threadgroup float input_tile[16 * 16];
  threadgroup float weight_tile[16 * 16];
  const uint row = group.y * kg_linear_tile + local.y;
  const uint output_channel = group.x * kg_linear_tile + local.x;
  const device ushort* weight =
      reinterpret_cast<const device ushort*>(checkpoint + params.weight_offset);
  const device ushort* bias =
      reinterpret_cast<const device ushort*>(checkpoint + params.bias_offset);
  float value = output_channel < params.output_channels && params.has_bias
      ? kg_bf16_to_f32(bias[output_channel])
      : 0.0f;

  for (uint channel_base = 0; channel_base < params.input_channels;
       channel_base += kg_linear_tile) {
    const uint channel = channel_base + local.x;
    input_tile[local.y * kg_linear_tile + local.x] =
        row < params.rows && channel < params.input_channels
            ? input[row * params.input_channels + channel]
            : 0.0f;
    const uint tile_output_channel = group.x * kg_linear_tile + local.y;
    weight_tile[local.y * kg_linear_tile + local.x] =
        tile_output_channel < params.output_channels && channel < params.input_channels
            ? kg_bf16_to_f32(
                  weight[tile_output_channel * params.input_channels + channel])
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < params.rows && output_channel < params.output_channels) {
      for (uint tile_channel = 0; tile_channel < kg_linear_tile; ++tile_channel) {
        value = fma(
            input_tile[local.y * kg_linear_tile + tile_channel],
            weight_tile[local.x * kg_linear_tile + tile_channel],
            value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < params.rows && output_channel < params.output_channels) {
    output[row * params.output_channels + output_channel] = value;
  }
}

kernel void kg_linear_tiled_f16_weights_f32_output(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LinearF32Params& params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]) {
  threadgroup float input_tile[16 * 16];
  threadgroup float weight_tile[16 * 16];
  const uint row = group.y * kg_linear_tile + local.y;
  const uint output_channel = group.x * kg_linear_tile + local.x;
  const device half* weight =
      reinterpret_cast<const device half*>(checkpoint + params.weight_offset);
  const device half* bias =
      reinterpret_cast<const device half*>(checkpoint + params.bias_offset);
  float value = output_channel < params.output_channels && params.has_bias
      ? float(bias[output_channel]) : 0.0f;
  for (uint channel_base = 0; channel_base < params.input_channels;
       channel_base += kg_linear_tile) {
    const uint channel = channel_base + local.x;
    input_tile[local.y * kg_linear_tile + local.x] =
        row < params.rows && channel < params.input_channels
            ? input[row * params.input_channels + channel] : 0.0f;
    const uint tile_output_channel = group.x * kg_linear_tile + local.y;
    weight_tile[local.y * kg_linear_tile + local.x] =
        tile_output_channel < params.output_channels && channel < params.input_channels
            ? float(weight[tile_output_channel * params.input_channels + channel]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row < params.rows && output_channel < params.output_channels) {
      for (uint tile_channel = 0; tile_channel < kg_linear_tile; ++tile_channel) {
        value = fma(input_tile[local.y * kg_linear_tile + tile_channel],
                    weight_tile[local.x * kg_linear_tile + tile_channel], value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && output_channel < params.output_channels) {
    output[row * params.output_channels + output_channel] = value;
  }
}

struct PatchEmbedF32Params {
  ulong weight_offset;
  ulong bias_offset;
  uint image_height;
  uint image_width;
  uint patch_size;
  uint input_channels;
  uint output_channels;
  uint output_token_offset;
};

kernel void kg_dino_patch_embed_f32(
    const device float* image [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant PatchEmbedF32Params& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const ulong patches_h = ulong(params.image_height / params.patch_size);
  const ulong patches_w = ulong(params.image_width / params.patch_size);
  const ulong patch_count = patches_h * patches_w;
  const ulong count = patch_count * ulong(params.output_channels);
  if (ulong(index) >= count) return;
  const ulong patch = ulong(index) / ulong(params.output_channels);
  const ulong output_channel = ulong(index) % ulong(params.output_channels);
  const ulong patch_y = patch / patches_w;
  const ulong patch_x = patch % patches_w;
  const device float* weight =
      reinterpret_cast<const device float*>(checkpoint + params.weight_offset);
  const device float* bias =
      reinterpret_cast<const device float*>(checkpoint + params.bias_offset);
  const ulong kernel_area = ulong(params.patch_size) * ulong(params.patch_size);
  float value = bias[output_channel];
  for (ulong channel = 0; channel < ulong(params.input_channels); ++channel) {
    const ulong image_channel_base =
        channel * ulong(params.image_height) * ulong(params.image_width);
    const ulong weight_channel_base =
        (output_channel * ulong(params.input_channels) + channel) * kernel_area;
    for (ulong y = 0; y < ulong(params.patch_size); ++y) {
      const ulong image_y = patch_y * ulong(params.patch_size) + y;
      for (ulong x = 0; x < ulong(params.patch_size); ++x) {
        const ulong image_x = patch_x * ulong(params.patch_size) + x;
        value = fma(
            image[image_channel_base + image_y * ulong(params.image_width) + image_x],
            weight[weight_channel_base + y * ulong(params.patch_size) + x],
            value);
      }
    }
  }
  output[(patch + ulong(params.output_token_offset)) * ulong(params.output_channels)
         + output_channel] = value;
}

struct DinoRopeF32Params {
  uint token_count;
  uint prefix_tokens;
  uint patches_h;
  uint patches_w;
  uint heads;
  uint dimensions;
  float theta;
};

kernel void kg_dino_rope_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant DinoRopeF32Params& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint count = params.token_count * params.heads * params.dimensions;
  if (index >= count) return;
  const uint token_stride = params.heads * params.dimensions;
  const uint token = index / token_stride;
  if (token < params.prefix_tokens) {
    output[index] = input[index];
    return;
  }
  const uint patch = token - params.prefix_tokens;
  const uint patch_y = patch / params.patches_w;
  const uint patch_x = patch % params.patches_w;
  const uint dimension = index % params.dimensions;
  const uint half_dimensions = params.dimensions / 2;
  const uint quarter = params.dimensions / 4;
  const uint angle_dimension = dimension % half_dimensions;
  const uint axis = angle_dimension / quarter;
  const uint frequency = angle_dimension % quarter;
  const float coordinate = axis == 0
      ? 2.0f * ((float(patch_y) + 0.5f) / float(params.patches_h)) - 1.0f
      : 2.0f * ((float(patch_x) + 0.5f) / float(params.patches_w)) - 1.0f;
  const float inverse_frequency = pow(
      params.theta, -4.0f * float(frequency) / float(params.dimensions));
  const float angle = 2.0f * M_PI_F * coordinate * inverse_frequency;
  const uint vector_base = index - dimension;
  const float rotated = dimension < half_dimensions
      ? -input[vector_base + dimension + half_dimensions]
      : input[vector_base + dimension - half_dimensions];
  output[index] = fma(input[index], cos(angle), rotated * sin(angle));
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

kernel void kg_layer_norm_f32_affine_f32(
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
  const device float* weight =
      reinterpret_cast<const device float*>(checkpoint + params.weight_offset);
  const device float* bias =
      reinterpret_cast<const device float*>(checkpoint + params.bias_offset);
  for (uint channel = 0; channel < params.channels; ++channel) {
    const float normalized = (input[base + channel] - mean) * inverse_std;
    output[base + channel] = params.has_affine
        ? fma(normalized, weight[channel], bias[channel])
        : normalized;
  }
}

kernel void kg_layer_norm_f32_affine_f16(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LayerNormParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows) return;
  const uint base = row * params.channels;
  float mean = 0.0f;
  for (uint channel = 0; channel < params.channels; ++channel) mean += input[base + channel];
  mean /= float(params.channels);
  float variance = 0.0f;
  for (uint channel = 0; channel < params.channels; ++channel) {
    const float centered = input[base + channel] - mean;
    variance = fma(centered, centered, variance);
  }
  variance /= float(params.channels);
  const float inverse_std = rsqrt(variance + params.epsilon);
  const device half* weight =
      reinterpret_cast<const device half*>(checkpoint + params.weight_offset);
  const device half* bias =
      reinterpret_cast<const device half*>(checkpoint + params.bias_offset);
  for (uint channel = 0; channel < params.channels; ++channel) {
    const float normalized = (input[base + channel] - mean) * inverse_std;
    output[base + channel] = params.has_affine
        ? fma(normalized, float(weight[channel]), float(bias[channel])) : normalized;
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

constant uint kg_attention_queries_per_group = 8;

kernel void kg_simdgroup_attention_f32(
    const device float* queries [[buffer(0)]],
    const device float* keys [[buffer(1)]],
    const device float* values [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint query = group.x * kg_attention_queries_per_group + simdgroup_index;
  const uint head = group.y;
  if (query >= params.query_count || head >= params.heads) return;

  const uint query_base = (query * params.heads + head) * params.dimensions;
  float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  const uint dimensions_per_lane = params.dimensions / 32;
  for (uint key = 0; key < params.key_count; ++key) {
    const uint key_base = (key * params.heads + head) * params.dimensions;
    float partial_score = 0.0f;
    for (uint slot = 0; slot < dimensions_per_lane; ++slot) {
      const uint dimension = lane * dimensions_per_lane + slot;
      partial_score = fma(
          queries[query_base + dimension], keys[key_base + dimension], partial_score);
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_max = max(running_max, score);
    const float previous_scale = exp(running_max - next_max);
    const float current_scale = exp(score - next_max);
    running_sum = running_sum * previous_scale + current_scale;
    running_max = next_max;
    for (uint slot = 0; slot < dimensions_per_lane; ++slot) {
      const uint dimension = lane * dimensions_per_lane + slot;
      if (dimension < params.dimensions) {
        accumulators[slot] = accumulators[slot] * previous_scale
            + current_scale * values[key_base + dimension];
      }
    }
  }
  for (uint slot = 0; slot < dimensions_per_lane; ++slot) {
    const uint dimension = lane * dimensions_per_lane + slot;
    if (dimension < params.dimensions) {
      output[query_base + dimension] = accumulators[slot] / running_sum;
    }
  }
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

inline float kg_erf_f32(float value) {
  const float sign = value < 0.0f ? -1.0f : 1.0f;
  const float magnitude = abs(value);
  const float t = 1.0f / (1.0f + 0.3275911f * magnitude);
  const float polynomial =
      (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
          - 0.284496736f) * t + 0.254829592f) * t;
  return sign * (1.0f - polynomial * exp(-magnitude * magnitude));
}

kernel void kg_gelu_erf_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const float value = input[index];
  output[index] = 0.5f * value
      * (1.0f + kg_erf_f32(value * 0.7071067811865475f));
}

struct LayerScaleResidualF32Params {
  ulong scale_offset;
  uint count;
  uint channels;
};

kernel void kg_layer_scale_residual_f32(
    const device float* residual [[buffer(0)]],
    const device float* branch [[buffer(1)]],
    const device uchar* checkpoint [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant LayerScaleResidualF32Params& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.count) return;
  const device float* scale =
      reinterpret_cast<const device float*>(checkpoint + params.scale_offset);
  output[index] = fma(branch[index], scale[index % params.channels], residual[index]);
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
