#include "kernel_goblin/trellis2/z_order.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace kg = kernel_goblin::trellis2;
using Clock = std::chrono::steady_clock;

template <typename Operation>
double measure_ms(Operation operation, int iterations) {
  operation();
  const auto start = Clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) operation();
  const auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
  try {
    constexpr std::size_t count = 1u << 20u;
    constexpr int iterations = 20;
    std::mt19937 random(0xC0FFEEu);
    std::uniform_int_distribution<std::uint32_t> coordinate(0, 1023);
    std::vector<std::uint32_t> x(count), y(count), z(count);
    for (std::size_t index = 0; index < count; ++index) {
      x[index] = coordinate(random);
      y[index] = coordinate(random);
      z[index] = coordinate(random);
    }

    kg::MetalZOrder metal;
    const auto expected = kg::z_order_encode_cpu(x, y, z);
    if (metal.encode(x, y, z) != expected) throw std::runtime_error("pre-benchmark conformance failed");

    const double cpu_ms = measure_ms([&] { static_cast<void>(kg::z_order_encode_cpu(x, y, z)); }, iterations);
    const double metal_ms = measure_ms([&] { static_cast<void>(metal.encode(x, y, z)); }, iterations);
    const double items = static_cast<double>(count);

    std::cout << std::fixed << std::setprecision(3)
              << "kernel: trellis2/z_order encode\n"
              << "device: " << metal.device_name() << "\n"
              << "elements: " << count << "\n"
              << "iterations: " << iterations << "\n"
              << "cpu_ms: " << cpu_ms << "\n"
              << "metal_ms: " << metal_ms << "\n"
              << "cpu_mitems_per_s: " << items / (cpu_ms * 1000.0) << "\n"
              << "metal_mitems_per_s: " << items / (metal_ms * 1000.0) << "\n"
              << "note: Metal timing includes shared-buffer allocation, copies, dispatch, and synchronization\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
