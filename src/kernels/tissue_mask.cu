// stainkit/src/kernels/tissue_mask.cu
//
// Otsu-based tissue masking. We expose two implementations:
//
//   * `OtsuThresholdHost` - copies the luminance image back to the host
//     and runs Otsu's method on the CPU. Easy to debug, fine for small
//     images.
//   * `OtsuThresholdDevice` - histogram-and-threshold entirely on the
//     GPU, using shared-memory privatisation and atomic adds. Faster
//     for large images.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "stainkit/kernels/morphology.h"
#include "stainkit/kernels/tissue_mask.h"

namespace stainkit {
namespace kernels {
namespace {

constexpr int kBins = 256;

__global__ void LumaKernel(const float* __restrict__ d_in,
                           float* __restrict__ d_out, std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const std::size_t base = 3 * idx;
  const float r = d_in[base + 0];
  const float g = d_in[base + 1];
  const float b = d_in[base + 2];
  d_out[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__global__ void HistogramKernel(const float* __restrict__ d_lum,
                                int* __restrict__ d_hist, std::size_t npix) {
  extern __shared__ unsigned int s_hist[];
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    s_hist[i] = 0u;
  }
  __syncthreads();
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < npix) {
    // Manual clamp: std::clamp is __host__ constexpr in C++17 and cannot
    // be called from a __global__ function without --expt-relaxed-constexpr.
    float v = d_lum[idx] * 255.0f;
    if (v < 0.0f)
      v = 0.0f;
    if (v > 255.0f)
      v = 255.0f;
    const int bin = static_cast<int>(v);
    atomicAdd(&s_hist[bin], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    atomicAdd(&d_hist[i], static_cast<int>(s_hist[i]));
  }
}

__global__ void ThresholdKernel(const float* __restrict__ d_lum,
                                const int* __restrict__ d_hist, float total,
                                float* __restrict__ d_threshold) {
  extern __shared__ unsigned int s_hist[];
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    s_hist[i] = static_cast<unsigned int>(d_hist[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum_all = 0.0f;
    float sum_back = 0.0f;
    float weight_back = 0.0f;
    for (int i = 0; i < kBins; ++i) {
      sum_all += static_cast<float>(i) * static_cast<float>(s_hist[i]);
    }
    float var_max = 0.0f;
    int best = 0;
    for (int t = 0; t < kBins; ++t) {
      weight_back += static_cast<float>(s_hist[t]);
      if (weight_back <= 0.0f)
        continue;
      const float weight_fore = total - weight_back;
      if (weight_fore <= 0.0f)
        break;
      sum_back += static_cast<float>(t) * static_cast<float>(s_hist[t]);
      const float mean_back = sum_back / weight_back;
      const float mean_fore = (sum_all - sum_back) / weight_fore;
      const float between = weight_back * weight_fore *
                            (mean_back - mean_fore) * (mean_back - mean_fore);
      if (between > var_max) {
        var_max = between;
        best = t;
      }
    }
    *d_threshold = static_cast<float>(best) / 255.0f;
  }
}

__global__ void BinariseKernel(const float* __restrict__ d_lum, float threshold,
                               std::uint8_t* __restrict__ d_out,
                               std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  d_out[idx] = (d_lum[idx] > threshold) ? 255 : 0;
}

}  // namespace

void RgbToLuminance(const float* d_in_rgb, std::size_t width,
                    std::size_t height, float* d_out_lum, cudaStream_t stream) {
  if (d_in_rgb == nullptr || d_out_lum == nullptr)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  LumaKernel<<<grid, block, 0, stream>>>(d_in_rgb, d_out_lum, npix);
}

float OtsuThresholdHost(const float* d_in_lum, std::size_t width,
                        std::size_t height, cudaStream_t stream) {
  if (d_in_lum == nullptr) {
    throw std::invalid_argument("OtsuThresholdHost: null device pointer");
  }
  const std::size_t npix = width * height;
  std::vector<float> h_lum(npix);
  cudaMemcpyAsync(h_lum.data(), d_in_lum, npix * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::vector<int> hist(kBins, 0);
  for (float v : h_lum) {
    int bin = static_cast<int>(std::clamp(v * 255.0f, 0.0f, 255.0f));
    hist[bin]++;
  }
  const float total = static_cast<float>(npix);
  float sum_all = 0.0f, sum_back = 0.0f;
  for (int i = 0; i < kBins; ++i) {
    sum_all += static_cast<float>(i) * static_cast<float>(hist[i]);
  }
  float var_max = 0.0f;
  int best = 0;
  float weight_b = 0.0f;
  for (int t = 0; t < kBins; ++t) {
    weight_b += static_cast<float>(hist[t]);
    if (weight_b <= 0.0f)
      continue;
    const float weight_f = total - weight_b;
    if (weight_f <= 0.0f)
      break;
    sum_back += static_cast<float>(t) * static_cast<float>(hist[t]);
    const float mean_b = sum_back / weight_b;
    const float mean_f = (sum_all - sum_back) / weight_f;
    const float between =
        weight_b * weight_f * (mean_b - mean_f) * (mean_b - mean_f);
    if (between > var_max) {
      var_max = between;
      best = t;
    }
  }
  return static_cast<float>(best) / 255.0f;
}

float OtsuThresholdDevice(const float* d_in_lum, std::size_t width,
                          std::size_t height, cudaStream_t stream) {
  if (d_in_lum == nullptr) {
    throw std::invalid_argument("OtsuThresholdDevice: null device pointer");
  }
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);

  int* d_hist = nullptr;
  float* d_thresh = nullptr;
  cudaMalloc(&d_hist, kBins * sizeof(int));
  cudaMalloc(&d_thresh, sizeof(float));
  cudaMemsetAsync(d_hist, 0, kBins * sizeof(int), stream);

  HistogramKernel<<<grid, block, kBins * sizeof(unsigned int), stream>>>(
      d_in_lum, d_hist, npix);
  ThresholdKernel<<<1, block, kBins * sizeof(unsigned int), stream>>>(
      d_in_lum, d_hist, static_cast<float>(npix), d_thresh);

  float threshold = 0.0f;
  cudaMemcpyAsync(&threshold, d_thresh, sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_hist);
  cudaFree(d_thresh);
  return threshold;
}

void ThresholdToMask(const float* d_in_lum, std::size_t width,
                     std::size_t height, float threshold, std::size_t radius,
                     std::uint8_t* d_out_mask, cudaStream_t stream) {
  if (d_in_lum == nullptr || d_out_mask == nullptr)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  BinariseKernel<<<grid, block, 0, stream>>>(d_in_lum, threshold, d_out_mask,
                                             npix);
  if (radius > 0) {
    // Open then close -> removes small isolated regions and fills small
    // holes inside the tissue. Cheap and effective for histopathology.
    Open(d_out_mask, width, height, radius, stream);
    Close(d_out_mask, width, height, radius, stream);
  }
}

}  // namespace kernels
}  // namespace stainkit
