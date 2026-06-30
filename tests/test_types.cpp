// stainkit/tests/test_types.cpp
//
// Smoke tests for the host-side type helpers.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "stainkit/types.h"

namespace stk = stainkit;

TEST(Types, MakeImagePadsToRow) {
  auto img = stk::MakeImage(7, 5, stk::PixelLayout::kRgb);
  EXPECT_EQ(img.width, 7u);
  EXPECT_EQ(img.height, 5u);
  // 7 * 3 = 21 bytes, rounded up to 24.
  EXPECT_EQ(img.stride, 24u);
  EXPECT_EQ(img.pixels.size(), img.stride * img.height);
}

TEST(Types, MakeImageRgbaStride) {
  auto img = stk::MakeImage(3, 3, stk::PixelLayout::kRgba);
  EXPECT_EQ(img.stride, 12u);  // 3 * 4 = 12, already a multiple of 4
  EXPECT_EQ(img.channels(), 4u);
}

TEST(Types, ClampUnit) {
  auto c = stk::ClampUnit({-1.0f, 2.0f, 0.5f});
  EXPECT_FLOAT_EQ(c[0], 0.0f);
  EXPECT_FLOAT_EQ(c[1], 1.0f);
  EXPECT_FLOAT_EQ(c[2], 0.5f);
}

TEST(Types, LumaIsBt709) {
  // Pure red, green, blue luminances per BT.709.
  EXPECT_NEAR(stk::Luma({1.0f, 0.0f, 0.0f}), 0.2126f, 1e-4);
  EXPECT_NEAR(stk::Luma({0.0f, 1.0f, 0.0f}), 0.7152f, 1e-4);
  EXPECT_NEAR(stk::Luma({0.0f, 0.0f, 1.0f}), 0.0722f, 1e-4);
}

TEST(Types, StainMatrixIdentity) {
  auto m = stk::StainMatrix::Identity();
  auto h = m.hematoxylin();
  auto e = m.eosin();
  EXPECT_GT(h[0], 0.0f);
  EXPECT_GT(e[1], 0.0f);
}

TEST(Types, ComputeStrideZero) {
  EXPECT_EQ(stk::ComputeStride(0, 3), 0u);
  EXPECT_EQ(stk::ComputeStride(10, 0), 0u);
}
