#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kernel_goblin::trellis2 {

struct Coordinates {
  std::vector<std::uint32_t> x;
  std::vector<std::uint32_t> y;
  std::vector<std::uint32_t> z;
};

std::vector<std::uint32_t> z_order_encode_cpu(
    const std::vector<std::uint32_t>& x,
    const std::vector<std::uint32_t>& y,
    const std::vector<std::uint32_t>& z);

Coordinates z_order_decode_cpu(const std::vector<std::uint32_t>& codes);

class MetalZOrder {
 public:
  MetalZOrder();
  ~MetalZOrder();
  MetalZOrder(MetalZOrder&&) noexcept;
  MetalZOrder& operator=(MetalZOrder&&) noexcept;
  MetalZOrder(const MetalZOrder&) = delete;
  MetalZOrder& operator=(const MetalZOrder&) = delete;

  [[nodiscard]] std::string device_name() const;

  std::vector<std::uint32_t> encode(
      const std::vector<std::uint32_t>& x,
      const std::vector<std::uint32_t>& y,
      const std::vector<std::uint32_t>& z) const;

  Coordinates decode(const std::vector<std::uint32_t>& codes) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernel_goblin::trellis2
