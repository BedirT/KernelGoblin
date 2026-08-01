#include <metal_stdlib>
using namespace metal;

static uint expand_bits(uint value) {
  value = (value * 0x00010001u) & 0xFF0000FFu;
  value = (value * 0x00000101u) & 0x0F00F00Fu;
  value = (value * 0x00000011u) & 0xC30C30C3u;
  value = (value * 0x00000005u) & 0x49249249u;
  return value;
}

static uint extract_bits(uint value) {
  value &= 0x49249249u;
  value = (value ^ (value >> 2u)) & 0x030C30C3u;
  value = (value ^ (value >> 4u)) & 0x0300F00Fu;
  value = (value ^ (value >> 8u)) & 0x030000FFu;
  value = (value ^ (value >> 16u)) & 0x000003FFu;
  return value;
}

kernel void z_order_encode(
    device const uint* x [[buffer(0)]],
    device const uint* y [[buffer(1)]],
    device const uint* z [[buffer(2)]],
    device uint* codes [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  codes[index] = expand_bits(x[index]) * 4u +
                 expand_bits(y[index]) * 2u + expand_bits(z[index]);
}

kernel void z_order_decode(
    device const uint* codes [[buffer(0)]],
    device uint* x [[buffer(1)]],
    device uint* y [[buffer(2)]],
    device uint* z [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  x[index] = extract_bits(codes[index] >> 2u);
  y[index] = extract_bits(codes[index] >> 1u);
  z[index] = extract_bits(codes[index]);
}
