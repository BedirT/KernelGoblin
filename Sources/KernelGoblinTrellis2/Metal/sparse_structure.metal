#include <metal_stdlib>
using namespace metal;

struct Conv3DParams {
  ulong weight_offset;
  ulong bias_offset;
  uint input_resolution;
  uint output_resolution;
  uint input_channels;
  uint output_channels;
  uint kernel_size;
  uint stride;
  uint padding;
  uint weights_are_f16;
};

kernel void kg_conv3d_voxel_major_f32(
    const device float* input [[buffer(0)]],
    const device uchar* checkpoint [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Conv3DParams& params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const ulong output_voxels = ulong(params.output_resolution)
      * ulong(params.output_resolution) * ulong(params.output_resolution);
  const ulong count = output_voxels * ulong(params.output_channels);
  if (ulong(index) >= count) return;

  const ulong output_voxel = ulong(index) / ulong(params.output_channels);
  const uint output_channel = uint(ulong(index) % ulong(params.output_channels));
  const uint oz = uint(output_voxel % ulong(params.output_resolution));
  const ulong output_xy = output_voxel / ulong(params.output_resolution);
  const uint oy = uint(output_xy % ulong(params.output_resolution));
  const uint ox = uint(output_xy / ulong(params.output_resolution));
  const device float* weight_f32 =
      reinterpret_cast<const device float*>(checkpoint + params.weight_offset);
  const device half* weight_f16 =
      reinterpret_cast<const device half*>(checkpoint + params.weight_offset);
  const device float* bias_f32 =
      reinterpret_cast<const device float*>(checkpoint + params.bias_offset);
  const device half* bias_f16 =
      reinterpret_cast<const device half*>(checkpoint + params.bias_offset);
  float value = params.weights_are_f16 != 0
      ? float(bias_f16[output_channel]) : bias_f32[output_channel];

  for (uint input_channel = 0; input_channel < params.input_channels; ++input_channel) {
    for (uint kx = 0; kx < params.kernel_size; ++kx) {
      const int ix = int(ox * params.stride + kx) - int(params.padding);
      if (ix < 0 || ix >= int(params.input_resolution)) continue;
      for (uint ky = 0; ky < params.kernel_size; ++ky) {
        const int iy = int(oy * params.stride + ky) - int(params.padding);
        if (iy < 0 || iy >= int(params.input_resolution)) continue;
        for (uint kz = 0; kz < params.kernel_size; ++kz) {
          const int iz = int(oz * params.stride + kz) - int(params.padding);
          if (iz < 0 || iz >= int(params.input_resolution)) continue;
          const ulong input_voxel =
              (ulong(ix) * ulong(params.input_resolution) + ulong(iy))
                  * ulong(params.input_resolution) + ulong(iz);
          const ulong input_index =
              input_voxel * ulong(params.input_channels) + ulong(input_channel);
          const ulong weight_index =
              ((((ulong(output_channel) * ulong(params.input_channels)
                  + ulong(input_channel)) * ulong(params.kernel_size) + ulong(kx))
                  * ulong(params.kernel_size) + ulong(ky))
                  * ulong(params.kernel_size) + ulong(kz));
          const float weight = params.weights_are_f16 != 0
              ? float(weight_f16[weight_index]) : weight_f32[weight_index];
          value = fma(input[input_index], weight, value);
        }
      }
    }
  }
  output[index] = value;
}

kernel void kg_round_f16_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = float(half(input[index]));
}

struct PixelShuffle3DParams {
  uint input_resolution;
  uint output_channels;
  uint factor;
};

kernel void kg_pixel_shuffle_3d_voxel_major_f32(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant PixelShuffle3DParams& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  const uint output_resolution = params.input_resolution * params.factor;
  const ulong output_voxels = ulong(output_resolution) * ulong(output_resolution)
      * ulong(output_resolution);
  const ulong count = output_voxels * ulong(params.output_channels);
  if (ulong(index) >= count) return;
  const ulong output_voxel = ulong(index) / ulong(params.output_channels);
  const uint channel = uint(ulong(index) % ulong(params.output_channels));
  const uint oz = uint(output_voxel % ulong(output_resolution));
  const ulong output_xy = output_voxel / ulong(output_resolution);
  const uint oy = uint(output_xy % ulong(output_resolution));
  const uint ox = uint(output_xy / ulong(output_resolution));
  const uint ix = ox / params.factor;
  const uint iy = oy / params.factor;
  const uint iz = oz / params.factor;
  const uint rx = ox % params.factor;
  const uint ry = oy % params.factor;
  const uint rz = oz % params.factor;
  const uint factor_cubed = params.factor * params.factor * params.factor;
  const uint input_channels = params.output_channels * factor_cubed;
  const uint shuffled_channel = channel * factor_cubed
      + (rx * params.factor + ry) * params.factor + rz;
  const ulong input_voxel =
      (ulong(ix) * ulong(params.input_resolution) + ulong(iy))
          * ulong(params.input_resolution) + ulong(iz);
  output[index] = input[input_voxel * ulong(input_channels) + ulong(shuffled_channel)];
}
