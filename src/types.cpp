// stainkit/src/types.cpp
//
// Translation unit for non-trivial value-type helpers declared in
// include/stainkit/types.h. Keeping them in a .cpp rather than the header
// means the inline definitions do not bloat every translation unit that
// pulls in the header.

#include "stainkit/types.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace stainkit {

namespace {

// Hard upper bound on image width/height. 64k * 64k * 4 channels is enough
// to address a whole-slide image that is roughly 0.25 µm/pixel over a 16mm
// tissue section, and small enough to avoid surprising allocations.
constexpr std::size_t kMaxDimension = 1u << 16;

}  // namespace

std::size_t ComputeStride(std::size_t width, std::size_t channels) {
  if (width == 0 || channels == 0) {
    return 0;
  }
  // Round up to the nearest 4-byte boundary for vectorised loads/stores.
  constexpr std::size_t kAlignment = 4;
  const std::size_t     raw        = width * channels;
  return (raw + (kAlignment - 1)) & ~(kAlignment - 1);
}

Image MakeImage(std::size_t width, std::size_t height, PixelLayout layout) {
  if (width == 0 || height == 0) {
    throw std::invalid_argument("MakeImage: zero-sized dimension");
  }
  if (width > kMaxDimension || height > kMaxDimension) {
    throw std::invalid_argument("MakeImage: dimension exceeds kMaxDimension");
  }
  const std::size_t channels = static_cast<std::size_t>(layout);
  const std::size_t stride   = ComputeStride(width, channels);
  Image img;
  img.width  = width;
  img.height = height;
  img.stride = stride;
  img.layout = layout;
  img.pixels.resize(stride * height, 0);
  return img;
}

float3 ClampUnit(const float3& v) noexcept {
  return {std::clamp(v[0], 0.0f, 1.0f),
          std::clamp(v[1], 0.0f, 1.0f),
          std::clamp(v[2], 0.0f, 1.0f)};
}

float Luma(const float3& v) noexcept {
  return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2];
}

}  // namespace stainkit
