// stainkit/src/kernels/od_conversion.cu
//
// RGB <-> optical density conversions. These are tiny, vectorisable
// per-pixel kernels that we expose both as in-place and out-of-place
// launchers.

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

#include "stainkit/kernels/od_conversion.h"

namespace stainkit {
namespace kernels {
namespace {

constexpr float kEps = 1e-6f;

__global__ void RgbToOdKernel(const float* __restrict__ in,
                              float* __restrict__ out, std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const std::size_t base = 3 * idx;
  const float r = fmaxf(in[base + 0], kEps);
  const float g = fmaxf(in[base + 1], kEps);
  const float b = fmaxf(in[base + 2], kEps);
  out[base + 0] = -logf(r);
  out[base + 1] = -logf(g);
  out[base + 2] = -logf(b);
}

__global__ void OdToRgbKernel(const float* __restrict__ in,
                              float* __restrict__ out, std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const std::size_t base = 3 * idx;
  out[base + 0] = expf(-in[base + 0]);
  out[base + 1] = expf(-in[base + 1]);
  out[base + 2] = expf(-in[base + 2]);
}

__global__ void RgbToLumaKernel(const float* __restrict__ in,
                                float* __restrict__ out, std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const std::size_t base = 3 * idx;
  const float r = in[base + 0];
  const float g = in[base + 1];
  const float b = in[base + 2];
  out[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

}  // namespace

void RgbToOd(float* d_io, std::size_t width, std::size_t height,
             cudaStream_t stream) {
  if (d_io == nullptr || width == 0 || height == 0)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  RgbToOdKernel<<<grid, block, 0, stream>>>(d_io, d_io, npix);
}

void RgbToOd(const float* d_in, float* d_out, std::size_t width,
             std::size_t height, cudaStream_t stream) {
  if (d_in == nullptr || d_out == nullptr || width == 0 || height == 0)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  RgbToOdKernel<<<grid, block, 0, stream>>>(d_in, d_out, npix);
}

void OdToRgb(const float* d_in, float* d_out, std::size_t width,
             std::size_t height, cudaStream_t stream) {
  if (d_in == nullptr || d_out == nullptr || width == 0 || height == 0)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  OdToRgbKernel<<<grid, block, 0, stream>>>(d_in, d_out, npix);
}

void RgbToLuma(const float* d_in, float* d_out, std::size_t width,
               std::size_t height, cudaStream_t stream) {
  if (d_in == nullptr || d_out == nullptr || width == 0 || height == 0)
    return;
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  RgbToLumaKernel<<<grid, block, 0, stream>>>(d_in, d_out, npix);
}

}  // namespace kernels
}  // namespace stainkit