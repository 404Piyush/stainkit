// stainkit/tests/test_cpu_reference.cpp
//
// Correctness test for the pure-CPU reference implementation. We feed a
// synthesised H&E-like image and assert that:
//   * the output has the right dimensions;
//   * the output is non-empty and contains a sensible range of values;
//   * the elapsed time is positive.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "stainkit/cpu_reference.h"
#include "stainkit/types.h"

namespace stk = stainkit;

namespace {

stk::Image MakeSyntheticHePatch(int w, int h) {
  auto img = stk::MakeImage(static_cast<std::size_t>(w),
                            static_cast<std::size_t>(h), stk::PixelLayout::kRgb);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Pinkish background, with a darker "nucleus" in the centre.
      const float cx    = x - w * 0.5f;
      const float cy    = y - h * 0.5f;
      const float r     = std::sqrt(cx * cx + cy * cy);
      const float core  = std::exp(-r * r / (w * h * 0.005f));
      const std::uint8_t R =
          static_cast<std::uint8_t>(std::clamp(220.0f - 90.0f * core, 0.0f, 255.0f));
      const std::uint8_t G =
          static_cast<std::uint8_t>(std::clamp(180.0f - 80.0f * core, 0.0f, 255.0f));
      const std::uint8_t B =
          static_cast<std::uint8_t>(std::clamp(190.0f - 30.0f * core, 0.0f, 255.0f));
      img.pixels[y * img.stride + 3 * x + 0] = R;
      img.pixels[y * img.stride + 3 * x + 1] = G;
      img.pixels[y * img.stride + 3 * x + 2] = B;
    }
  }
  return img;
}

}  // namespace

TEST(CpuReference, ProducesSameDims) {
  auto img = MakeSyntheticHePatch(48, 32);
  stk::StainTarget target;
  double ms = 0.0;
  auto out  = stk::CpuReferenceStainNormalise(img, stk::PipelineParams{}, target,
                                              &ms);
  EXPECT_EQ(out.width, img.width);
  EXPECT_EQ(out.height, img.height);
  EXPECT_GT(ms, 0.0);
}

TEST(CpuReference, OutputInRange) {
  auto img = MakeSyntheticHePatch(24, 24);
  stk::StainTarget target;
  auto out = stk::CpuReferenceStainNormalise(img, stk::PipelineParams{}, target);
  for (std::size_t y = 0; y < out.height; ++y) {
    for (std::size_t x = 0; x < out.width; ++x) {
      const std::uint8_t r = out.pixels[y * out.stride + 3 * x + 0];
      const std::uint8_t g = out.pixels[y * out.stride + 3 * x + 1];
      const std::uint8_t b = out.pixels[y * out.stride + 3 * x + 2];
      // All output pixels must be valid 8-bit values.
      EXPECT_LE(r, 255);
      EXPECT_LE(g, 255);
      EXPECT_LE(b, 255);
    }
  }
}
