#include <metal_stdlib>
using namespace metal;

kernel void kg_flexible_dual_grid_head_f32(
    const device float* raw [[buffer(0)]],
    device float* dual [[buffer(1)]],
    device uchar* intersections [[buffer(2)]],
    device float* split [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  const ulong raw_base = ulong(index) * 7;
  const ulong vector_base = ulong(index) * 3;
  for (uint axis = 0; axis < 3; ++axis) {
    dual[vector_base + axis] = 2.0f / (1.0f + exp(-raw[raw_base + axis])) - 0.5f;
    intersections[vector_base + axis] = raw[raw_base + 3 + axis] > 0.0f ? 1 : 0;
  }
  const float value = raw[raw_base + 6];
  split[index] = value > 20.0f ? value
      : (value < -20.0f ? exp(value) : log(1.0f + exp(value)));
}
