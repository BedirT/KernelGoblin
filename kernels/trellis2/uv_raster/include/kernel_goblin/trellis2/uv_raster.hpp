#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kernel_goblin::trellis2 {

struct UVRasterResult {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<float> positions;
  std::vector<std::uint32_t> face_ids;
};

UVRasterResult uv_raster_cpu(const std::vector<float>& positions,
                             const std::vector<float>& uvs,
                             const std::vector<std::uint32_t>& faces,
                             std::uint32_t width,
                             std::uint32_t height);

class MetalUVRaster {
 public:
  MetalUVRaster();
  ~MetalUVRaster();
  MetalUVRaster(MetalUVRaster&&) noexcept;
  MetalUVRaster& operator=(MetalUVRaster&&) noexcept;
  MetalUVRaster(const MetalUVRaster&) = delete;
  MetalUVRaster& operator=(const MetalUVRaster&) = delete;

  [[nodiscard]] std::string device_name() const;
  UVRasterResult rasterize(const std::vector<float>& positions,
                           const std::vector<float>& uvs,
                           const std::vector<std::uint32_t>& faces,
                           std::uint32_t width,
                           std::uint32_t height) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kernel_goblin::trellis2
