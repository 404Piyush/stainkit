// stainkit/tests/test_pipeline.cpp
//
// Integration test for the full Pipeline. Mirrors the CPU reference
// tests but adds an end-to-end check.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "stainkit/cpu_reference.h"
#include "stainkit/pipeline.h"
#include "stainkit/types.h"

namespace stk = stainkit;

namespace {

stk::Image MakeSyntheticHePatch(int w, int h) {
  auto img = stk::MakeImage(static_cast<std::size_t>(w),
                            static_cast<std::size_t>(h), stk::PixelLayout::kRgb);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float cx   = x - w * 0.5f;
      const float cy   = y - h * 0.5f;
      const float r    = std::sqrt(cx * cx + cy * cy);
      const float core = std::exp(-r * r / (w * h * 0.005f));
      img.pixels[y * img.stride + 3 * x + 0] =
          static_cast<std::uint8_t>(std::clamp(220.0f - 90.0f * core, 0.0f, 255.0f));
      img.pixels[y * img.stride + 3 * x + 1] =
          static_cast<std::uint8_t>(std::clamp(180.0f - 80.0f * core, 0.0f, 255.0f));
      img.pixels[y * img.stride + 3 * x + 2] =
          static_cast<std::uint8_t>(std::clamp(190.0f - 30.0f * core, 0.0f, 255.0f));
    }
  }
  return img;
}

}  // namespace

TEST(Pipeline, CpuBaselineRun) {
  auto img = MakeSyntheticHePatch(48, 32);
  stk::StainTarget target;
  double ms = 0.0;
  auto out  = stk::CpuReferenceStainNormalise(img, stk::PipelineParams{}, target,
                                              &ms);
  EXPECT_EQ(out.width, img.width);
  EXPECT_EQ(out.height, img.height);
  EXPECT_GT(ms, 0.0);
}

TEST(Pipeline, GpuRunIfAvailable) {
  if (!stk::Pipeline::IsCudaAvailable()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  auto img    = MakeSyntheticHePatch(64, 64);
  auto pipe   = stk::Pipeline::Make();
  ASSERT_NE(pipe, nullptr);
  stk::StainTarget target;
  stk::PipelineParams params;
  params.compute_tissue_mask = true;
  auto result = pipe->RunWithCpuBaseline(img, params, target);
  EXPECT_EQ(result.normalised.width, img.width);
  EXPECT_EQ(result.normalised.height, img.height);
  EXPECT_EQ(result.tissue_mask.width, img.width);
  EXPECT_EQ(result.tissue_mask.height, img.height);
  EXPECT_GT(result.timing.total_ms, 0.0);
  EXPECT_GT(result.timing.cpu_baseline_ms, 0.0);
}
