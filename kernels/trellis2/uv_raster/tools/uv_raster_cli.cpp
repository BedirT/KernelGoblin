#include "kernel_goblin/trellis2/uv_raster.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Header {
  std::array<char, 4> magic;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t vertex_count;
  std::uint32_t face_count;
};

template <typename T>
std::vector<T> read_vector(std::ifstream& stream, std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::length_error("input array size overflow");
  }
  std::vector<T> values(count);
  stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
  if (!stream) throw std::runtime_error("truncated UV raster input");
  return values;
}

template <typename T>
void write_vector(std::ofstream& stream, const std::vector<T>& values) {
  stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  if (!stream) throw std::runtime_error("could not write UV raster output");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: uv_raster_cli INPUT OUTPUT");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("could not open UV raster input");
    Header header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.magic != std::array<char, 4>{'K', 'G', 'U', 'V'}) {
      throw std::runtime_error("invalid UV raster input header");
    }
    const auto positions = read_vector<float>(input, static_cast<std::size_t>(header.vertex_count) * 3);
    const auto uvs = read_vector<float>(input, static_cast<std::size_t>(header.vertex_count) * 2);
    const auto faces = read_vector<std::uint32_t>(input, static_cast<std::size_t>(header.face_count) * 3);
    if (input.peek() != std::ifstream::traits_type::eof()) {
      throw std::runtime_error("UV raster input contains trailing data");
    }

    kernel_goblin::trellis2::MetalUVRaster raster;
    const auto result = raster.rasterize(
        positions, uvs, faces, header.width, header.height);
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not open UV raster output");
    const Header output_header{{'K', 'G', 'U', 'R'}, result.width, result.height, 0, 0};
    output.write(reinterpret_cast<const char*>(&output_header), sizeof(output_header));
    write_vector(output, result.positions);
    write_vector(output, result.face_ids);
    std::cout << "backend=Metal device=\"" << raster.device_name() << "\"\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
