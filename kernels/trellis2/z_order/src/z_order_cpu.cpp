#include "kernel_goblin/trellis2/z_order.hpp"

#include <stdexcept>

namespace kernel_goblin::trellis2 {
namespace {

std::uint32_t expand_bits(std::uint32_t value) {
  value = (value * 0x00010001u) & 0xFF0000FFu;
  value = (value * 0x00000101u) & 0x0F00F00Fu;
  value = (value * 0x00000011u) & 0xC30C30C3u;
  value = (value * 0x00000005u) & 0x49249249u;
  return value;
}

std::uint32_t extract_bits(std::uint32_t value) {
  value &= 0x49249249u;
  value = (value ^ (value >> 2u)) & 0x030C30C3u;
  value = (value ^ (value >> 4u)) & 0x0300F00Fu;
  value = (value ^ (value >> 8u)) & 0x030000FFu;
  value = (value ^ (value >> 16u)) & 0x000003FFu;
  return value;
}

}  // namespace

std::vector<std::uint32_t> z_order_encode_cpu(
    const std::vector<std::uint32_t>& x,
    const std::vector<std::uint32_t>& y,
    const std::vector<std::uint32_t>& z) {
  if (x.size() != y.size() || x.size() != z.size()) {
    throw std::invalid_argument("x, y, and z must have identical lengths");
  }
  std::vector<std::uint32_t> codes(x.size());
  for (std::size_t index = 0; index < x.size(); ++index) {
    codes[index] = expand_bits(x[index]) * 4u +
                   expand_bits(y[index]) * 2u + expand_bits(z[index]);
  }
  return codes;
}

Coordinates z_order_decode_cpu(const std::vector<std::uint32_t>& codes) {
  Coordinates result{
      std::vector<std::uint32_t>(codes.size()),
      std::vector<std::uint32_t>(codes.size()),
      std::vector<std::uint32_t>(codes.size()),
  };
  for (std::size_t index = 0; index < codes.size(); ++index) {
    result.x[index] = extract_bits(codes[index] >> 2u);
    result.y[index] = extract_bits(codes[index] >> 1u);
    result.z[index] = extract_bits(codes[index]);
  }
  return result;
}

}  // namespace kernel_goblin::trellis2
