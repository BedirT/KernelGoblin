#include "kernel_goblin/trellis2/uv_raster.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using kernel_goblin::trellis2::MetalUVRaster;
using kernel_goblin::trellis2::UVRasterResult;
using kernel_goblin::trellis2::uv_raster_cpu;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void compare_single_triangle(const UVRasterResult& expected,
                             const UVRasterResult& actual) {
  if (expected.face_ids != actual.face_ids) {
    std::size_t differences = 0;
    std::string detail;
    for (std::size_t i = 0; i < expected.face_ids.size(); ++i) {
      if (expected.face_ids[i] != actual.face_ids[i]) {
        ++differences;
        if (differences <= 8) {
          detail += " " + std::to_string(i) + ":" +
                    std::to_string(expected.face_ids[i]) + "->" +
                    std::to_string(actual.face_ids[i]);
        }
      }
    }
    throw std::runtime_error("Metal coverage or face IDs differ from CPU at " +
                             std::to_string(differences) + " pixels:" + detail);
  }
  for (std::size_t pixel = 0; pixel < expected.face_ids.size(); ++pixel) {
    if (expected.face_ids[pixel] == 0) continue;
    for (std::size_t component = 0; component < 4; ++component) {
      const auto offset = pixel * 4 + component;
      require(std::abs(expected.positions[offset] - actual.positions[offset]) <= 2e-6f,
              "Metal interpolated position differs from CPU");
    }
  }
}

void test_single_triangle(MetalUVRaster& metal) {
  const std::vector<float> positions = {0, 0, 0, 1, 0, 0, 0, 1, 1};
  const std::vector<float> uvs = {0.125f, 0.125f, 0.875f, 0.125f, 0.125f, 0.875f};
  const std::vector<std::uint32_t> faces = {0, 1, 2};
  compare_single_triangle(uv_raster_cpu(positions, uvs, faces, 8, 8),
                          metal.rasterize(positions, uvs, faces, 8, 8));

  const std::vector<std::uint32_t> reversed = {2, 1, 0};
  compare_single_triangle(uv_raster_cpu(positions, uvs, reversed, 8, 8),
                          metal.rasterize(positions, uvs, reversed, 8, 8));
}

void test_shared_edge(MetalUVRaster& metal) {
  const std::vector<float> positions = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  const std::vector<float> uvs = {0, 0, 1, 0, 1, 1, 0, 1};
  const std::vector<std::uint32_t> faces = {0, 1, 2, 0, 2, 3};
  const auto actual = metal.rasterize(positions, uvs, faces, 16, 16);
  for (auto id : actual.face_ids) require(id != 0, "shared-edge quad contains a coverage crack");
}

void test_degenerate(MetalUVRaster& metal) {
  const std::vector<float> positions = {0, 0, 0, 1, 0, 0, 2, 0, 0};
  const std::vector<float> uvs = {0.1f, 0.1f, 0.5f, 0.5f, 0.9f, 0.9f};
  const std::vector<std::uint32_t> faces = {0, 1, 2};
  const auto result = metal.rasterize(positions, uvs, faces, 8, 8);
  for (auto id : result.face_ids) require(id == 0, "degenerate triangle covered pixels");
}

void test_invalid_inputs(MetalUVRaster& metal) {
  bool threw = false;
  try {
    (void)metal.rasterize({0, 0, 0}, {0, 0}, {1, 1, 1}, 8, 8);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "out-of-range faces were accepted");
  threw = false;
  try {
    (void)metal.rasterize({0, 0, 0}, {0, 0}, {}, 0, 8);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "zero-size target was accepted");
  threw = false;
  try {
    (void)metal.rasterize({0, 0, 0, 1, 0, 0, 0, 1, 0},
                           {-0.1f, 0, 1, 0, 0, 1}, {0, 1, 2}, 8, 8);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "out-of-range normalized UVs were accepted");
}

}  // namespace

int main() {
  try {
    MetalUVRaster metal;
    test_single_triangle(metal);
    test_shared_edge(metal);
    test_degenerate(metal);
    test_invalid_inputs(metal);
    std::cout << "PASS: physical Metal UV raster on " << metal.device_name() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
