#include "kernel_goblin/trellis2/uv_raster.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kernel_goblin::trellis2 {
namespace {

void validate(const std::vector<float>& positions,
              const std::vector<float>& uvs,
              const std::vector<std::uint32_t>& faces,
              std::uint32_t width,
              std::uint32_t height) {
  if (positions.size() % 3 != 0 || uvs.size() % 2 != 0 || faces.size() % 3 != 0) {
    throw std::invalid_argument("positions, UVs, and faces must have packed tuple shapes");
  }
  if (positions.size() / 3 != uvs.size() / 2) {
    throw std::invalid_argument("positions and UVs must have identical vertex counts");
  }
  if (width == 0 || height == 0) {
    throw std::invalid_argument("raster dimensions must be nonzero");
  }
  for (float value : positions) {
    if (!std::isfinite(value)) throw std::invalid_argument("positions must be finite");
  }
  for (float value : uvs) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
      throw std::invalid_argument("UVs must be finite and normalized to [0, 1]");
    }
  }
  const auto vertex_count = positions.size() / 3;
  for (std::uint32_t index : faces) {
    if (index >= vertex_count) throw std::out_of_range("face index exceeds vertex count");
  }
  const auto pixels = static_cast<std::uint64_t>(width) * height;
  if (pixels > std::numeric_limits<std::size_t>::max() / (4 * sizeof(float))) {
    throw std::length_error("raster dimensions overflow host address space");
  }
}

float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

}  // namespace

UVRasterResult uv_raster_cpu(const std::vector<float>& positions,
                             const std::vector<float>& uvs,
                             const std::vector<std::uint32_t>& faces,
                             std::uint32_t width,
                             std::uint32_t height) {
  validate(positions, uvs, faces, width, height);
  UVRasterResult result;
  result.width = width;
  result.height = height;
  const auto pixel_count = static_cast<std::size_t>(width) * height;
  result.positions.assign(pixel_count * 4, 0.0f);
  result.face_ids.assign(pixel_count, 0);

  for (std::size_t face = 0; face < faces.size() / 3; ++face) {
    const auto i0 = faces[face * 3];
    const auto i1 = faces[face * 3 + 1];
    const auto i2 = faces[face * 3 + 2];
    const float x0 = uvs[i0 * 2], y0 = uvs[i0 * 2 + 1];
    const float x1 = uvs[i1 * 2], y1 = uvs[i1 * 2 + 1];
    const float x2 = uvs[i2 * 2], y2 = uvs[i2 * 2 + 1];
    const float area = edge(x0, y0, x1, y1, x2, y2);
    if (std::abs(area) <= std::numeric_limits<float>::epsilon()) continue;

    for (std::uint32_t y = 0; y < height; ++y) {
      const float py = (static_cast<float>(y) + 0.5f) / height;
      for (std::uint32_t x = 0; x < width; ++x) {
        const float px = (static_cast<float>(x) + 0.5f) / width;
        const float w0 = edge(x1, y1, x2, y2, px, py) / area;
        const float w1 = edge(x2, y2, x0, y0, px, py) / area;
        const float w2 = 1.0f - w0 - w1;
        // Atlas charts do not overlap, so an edge belongs to one adjacent
        // triangle. Excluding exact edge centers matches Metal's top-left fill
        // ownership for this per-face reference without double coverage.
        constexpr float edge_tolerance = 1e-6f;
        if (w0 <= edge_tolerance || w1 <= edge_tolerance || w2 <= edge_tolerance) continue;
        const auto pixel = static_cast<std::size_t>(y) * width + x;
        for (std::size_t component = 0; component < 3; ++component) {
          result.positions[pixel * 4 + component] =
              w0 * positions[i0 * 3 + component] +
              w1 * positions[i1 * 3 + component] +
              w2 * positions[i2 * 3 + component];
        }
        result.positions[pixel * 4 + 3] = 1.0f;
        result.face_ids[pixel] = static_cast<std::uint32_t>(face + 1);
      }
    }
  }
  return result;
}

}  // namespace kernel_goblin::trellis2
