#include <metal_stdlib>
using namespace metal;

kernel void kg_trellis_texture_to_pbr_f32(
    const device float* raw [[buffer(0)]],
    device float* pbr [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  pbr[index] = fma(raw[index], 0.5f, 0.5f);
}
