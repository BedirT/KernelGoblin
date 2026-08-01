#include "kernel_goblin/trellis2/z_order.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace kg = kernel_goblin::trellis2;

namespace {

template <typename T>
void expect_equal(const T& actual, const T& expected, const std::string& label) {
  if (actual != expected) throw std::runtime_error(label + " mismatch");
}

void expect_invalid_lengths(const kg::MetalZOrder& metal) {
  bool cpu_threw = false;
  bool metal_threw = false;
  try {
    static_cast<void>(kg::z_order_encode_cpu({1}, {}, {}));
  } catch (const std::invalid_argument&) {
    cpu_threw = true;
  }
  try {
    static_cast<void>(metal.encode({1}, {}, {}));
  } catch (const std::invalid_argument&) {
    metal_threw = true;
  }
  if (!cpu_threw || !metal_threw) throw std::runtime_error("length validation mismatch");
}

void verify_case(const kg::MetalZOrder& metal,
                 const std::vector<std::uint32_t>& x,
                 const std::vector<std::uint32_t>& y,
                 const std::vector<std::uint32_t>& z,
                 const std::string& label) {
  const auto reference_codes = kg::z_order_encode_cpu(x, y, z);
  const auto metal_codes = metal.encode(x, y, z);
  expect_equal(metal_codes, reference_codes, label + " encode");

  const auto reference_decoded = kg::z_order_decode_cpu(reference_codes);
  const auto metal_decoded = metal.decode(metal_codes);
  expect_equal(metal_decoded.x, reference_decoded.x, label + " decode x");
  expect_equal(metal_decoded.y, reference_decoded.y, label + " decode y");
  expect_equal(metal_decoded.z, reference_decoded.z, label + " decode z");
  expect_equal(metal_decoded.x, x, label + " round trip x");
  expect_equal(metal_decoded.y, y, label + " round trip y");
  expect_equal(metal_decoded.z, z, label + " round trip z");
}

}  // namespace

int main() {
  try {
    kg::MetalZOrder metal;
    std::cout << "Metal device: " << metal.device_name() << '\n';

    verify_case(metal, {}, {}, {}, "empty");
    expect_invalid_lengths(metal);

    const std::vector<std::uint32_t> known_x{0, 1, 0, 0, 1, 1023};
    const std::vector<std::uint32_t> known_y{0, 0, 1, 0, 1, 1023};
    const std::vector<std::uint32_t> known_z{0, 0, 0, 1, 1, 1023};
    const std::vector<std::uint32_t> known_codes{0, 4, 2, 1, 7, 0x3FFFFFFF};
    expect_equal(kg::z_order_encode_cpu(known_x, known_y, known_z), known_codes,
                 "known Morton codes");
    verify_case(metal, known_x, known_y, known_z, "known");

    std::vector<std::uint32_t> axis(1024);
    std::vector<std::uint32_t> zero(1024, 0);
    for (std::uint32_t value = 0; value < 1024; ++value) axis[value] = value;
    verify_case(metal, axis, zero, zero, "x axis exhaustive");
    verify_case(metal, zero, axis, zero, "y axis exhaustive");
    verify_case(metal, zero, zero, axis, "z axis exhaustive");

    constexpr std::size_t count = 65536;
    std::mt19937 random(0xC0FFEEu);
    std::uniform_int_distribution<std::uint32_t> coordinate(0, 1023);
    std::vector<std::uint32_t> x(count), y(count), z(count);
    for (std::size_t index = 0; index < count; ++index) {
      x[index] = coordinate(random);
      y[index] = coordinate(random);
      z[index] = coordinate(random);
    }
    verify_case(metal, x, y, z, "randomized");

    std::cout << "PASS: CPU and Metal are bit-exact across all conformance cases\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
