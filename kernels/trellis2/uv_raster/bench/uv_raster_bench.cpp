#include "kernel_goblin/trellis2/uv_raster.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  using clock = std::chrono::steady_clock;
  using kernel_goblin::trellis2::MetalUVRaster;

  const std::vector<float> positions = {0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1};
  const std::vector<float> uvs = {0, 0, 1, 0, 1, 1, 0, 1};
  const std::vector<std::uint32_t> faces = {0, 1, 2, 0, 2, 3};
  constexpr std::uint32_t size = 1024;
  constexpr int warmup = 3;
  constexpr int iterations = 20;
  MetalUVRaster raster;

  const auto check = raster.rasterize(positions, uvs, faces, size, size);
  for (std::uint32_t y = 0; y < size; ++y) {
    for (std::uint32_t x = 0; x < size; ++x) {
      const auto pixel = static_cast<std::size_t>(y) * size + x;
      const float expected_x = (static_cast<float>(x) + 0.5f) / size;
      const float expected_y = (static_cast<float>(y) + 0.5f) / size;
      if ((check.face_ids[pixel] != 1 && check.face_ids[pixel] != 2) ||
          std::abs(check.positions[pixel * 4] - expected_x) > 2e-6f ||
          std::abs(check.positions[pixel * 4 + 1] - expected_y) > 2e-6f ||
          std::abs(check.positions[pixel * 4 + 2] - expected_y) > 2e-6f ||
          check.positions[pixel * 4 + 3] != 1.0f) {
        std::cerr << "FAIL: analytic correctness gate failed at pixel " << pixel << '\n';
        return 1;
      }
    }
  }
  for (int i = 0; i < warmup; ++i) (void)raster.rasterize(positions, uvs, faces, size, size);

  std::vector<double> milliseconds;
  for (int i = 0; i < iterations; ++i) {
    const auto start = clock::now();
    (void)raster.rasterize(positions, uvs, faces, size, size);
    milliseconds.push_back(std::chrono::duration<double, std::milli>(clock::now() - start).count());
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  std::cout << "backend=Metal device=\"" << raster.device_name() << "\"\n"
            << "workload=1024x1024 faces=2 warmup=" << warmup
            << " iterations=" << iterations << '\n'
            << "timing=host allocation + upload + draw + synchronized readback\n"
            << "median_ms=" << milliseconds[milliseconds.size() / 2]
            << " min_ms=" << milliseconds.front()
            << " max_ms=" << milliseconds.back() << '\n';
  return 0;
}
