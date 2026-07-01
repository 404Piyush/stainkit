// stainkit/tests/test_kernels.cu
//
// Smoke tests for the CUDA kernels. Each test is marked as skipped if
// no CUDA device is available so the suite still passes on a CPU-only
// build agent.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "stainkit/kernels/color_deconvolution.h"
#include "stainkit/kernels/od_conversion.h"
#include "stainkit/kernels/stain_normalization.h"
#include "stainkit/kernels/tissue_mask.h"

namespace stk = stainkit;
namespace sk = stk::kernels;

namespace {

bool CudaAvailable() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  return (e == cudaSuccess && count > 0);
}

template <typename T>
T* Upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

}  // namespace

#define SKIP_IF_NO_CUDA()                         \
  do {                                            \
    if (!CudaAvailable()) {                       \
      GTEST_SKIP() << "No CUDA device available"; \
    }                                             \
  } while (0)

TEST(Kernels, RgbToOdRoundTrip) {
  SKIP_IF_NO_CUDA();
  const std::size_t w = 16, h = 16;
  std::vector<float> in(w * h * 3);
  for (std::size_t i = 0; i < w * h; ++i) {
    in[3 * i + 0] = 0.5f;
    in[3 * i + 1] = 0.25f;
    in[3 * i + 2] = 0.75f;
  }
  float* d = Upload(in);
  sk::RgbToOd(d, w, h);
  std::vector<float> out(w * h * 3);
  cudaMemcpy(out.data(), d, w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  sk::OdToRgb(d, d, w, h);
  std::vector<float> round_trip(w * h * 3);
  cudaMemcpy(round_trip.data(), d, w * h * 3 * sizeof(float),
             cudaMemcpyDeviceToHost);
  // OD should be the negative log of 0.5, 0.25, 0.75 respectively.
  for (std::size_t i = 0; i < w * h; ++i) {
    EXPECT_NEAR(out[3 * i + 0], -std::log(0.5f), 1e-3);
    EXPECT_NEAR(out[3 * i + 1], -std::log(0.25f), 1e-3);
    EXPECT_NEAR(out[3 * i + 2], -std::log(0.75f), 1e-3);
    EXPECT_NEAR(round_trip[3 * i + 0], 0.5f, 1e-3);
    EXPECT_NEAR(round_trip[3 * i + 1], 0.25f, 1e-3);
    EXPECT_NEAR(round_trip[3 * i + 2], 0.75f, 1e-3);
  }
  cudaFree(d);
}

TEST(Kernels, OtsuThresholdDiscriminatesClasses) {
  SKIP_IF_NO_CUDA();
  // Construct a synthetic image with two clear intensity classes.
  const std::size_t w = 32, h = 32;
  std::vector<float> lum(w * h, 0.1f);
  for (std::size_t y = h / 4; y < 3 * h / 4; ++y) {
    for (std::size_t x = w / 4; x < 3 * w / 4; ++x) {
      lum[y * w + x] = 0.9f;
    }
  }
  float* d = Upload(lum);
  const float t = sk::OtsuThresholdDevice(d, w, h);
  EXPECT_GT(t, 0.4f);
  EXPECT_LT(t, 0.6f);
  cudaFree(d);
}
