// stainkit/src/kernels/color_deconvolution.cu
//
// Ruifrok & Johnston (2001) color deconvolution kernel.
//   - Input:  per-pixel RGB in [0, 1].
//   - Output: per-pixel OD for each stain channel.
//
// The math reduces to a 3x3 matrix multiply per pixel. We pass the
// inverse stain matrix as a kernel argument (in device memory) rather
// than using __constant__ memory: cudaMemcpyToSymbolAsync has caused
// segfaults on Colab's CUDA 12.8 runtime, and using regular device
// memory is more portable anyway.

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "stainkit/kernels/color_deconvolution.h"
#include "stainkit/types.h"

namespace stainkit {
namespace kernels {
namespace {

// 9 floats = the inverse stain matrix (column-major). Kept in device
// memory and passed by pointer to the kernel.
__global__ void DeconvolveKernel(const float* __restrict__ d_in,
                                 const float* __restrict__ d_stain_inv,
                                 float* __restrict__ d_out, std::size_t npix) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (idx >= npix) return;
  const std::size_t base = 3 * idx;

  // Convert RGB to OD inline (saves a full pass over global memory).
  const float eps   = 1e-6f;
  const float od_r  = -logf(fmaxf(d_in[base + 0], eps));
  const float od_g  = -logf(fmaxf(d_in[base + 1], eps));
  const float od_b  = -logf(fmaxf(d_in[base + 2], eps));

  // Apply the inverse stain matrix. We use the complete 3x3 form even
  // when only H and E are wanted so a future residual channel can be
  // exposed without a code change.
  const float c0 = d_stain_inv[0] * od_r + d_stain_inv[1] * od_g + d_stain_inv[2] * od_b;
  const float c1 = d_stain_inv[3] * od_r + d_stain_inv[4] * od_g + d_stain_inv[5] * od_b;
  const float c2 = d_stain_inv[6] * od_r + d_stain_inv[7] * od_g + d_stain_inv[8] * od_b;

  d_out[2 * idx + 0] = c0;
  d_out[2 * idx + 1] = c1;
  // c2 is computed for completeness (residual channel); not written out.
  (void)c2;
}

inline cudaStream_t AsStream(void* s) {
  return (s == nullptr) ? 0 : *static_cast<cudaStream_t*>(s);
}

}  // namespace

float ColorDeconvolveRgb(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const float* h_matrix_values_6,
                         float* d_out_stain_od, int num_stains,
                         int num_streams, void* stream) {
  std::fprintf(stderr, "[CDR] enter w=%zu h=%zu\n", width, height); std::fflush(stderr);
  if (d_in_rgb == nullptr || d_out_stain_od == nullptr) {
    throw std::invalid_argument("ColorDeconvolveRgb: null device pointer");
  }
  if (h_matrix_values_6 == nullptr) {
    throw std::invalid_argument("ColorDeconvolveRgb: null host matrix");
  }
  if (width == 0 || height == 0) {
    throw std::invalid_argument("ColorDeconvolveRgb: zero-sized image");
  }
  std::fprintf(stderr, "[CDR] past null check\n"); std::fflush(stderr);

  // Build a 3x3 stain matrix on the host. We use the (R, G, B)-column-major
  // form. The third column is taken as the cross product of H and E.
  std::array<float, 9> m{};
  m[0] = h_matrix_values_6[0];
  m[1] = h_matrix_values_6[1];
  m[2] = h_matrix_values_6[2];
  m[3] = h_matrix_values_6[3];
  m[4] = h_matrix_values_6[4];
  m[5] = h_matrix_values_6[5];
  const float hx = m[0], hy = m[1], hz = m[2];
  const float ex = m[3], ey = m[4], ez = m[5];
  m[6] = hy * ez - hz * ey;
  m[7] = hz * ex - hx * ez;
  m[8] = hx * ey - hy * ex;
  const float nrm = std::sqrt(m[6] * m[6] + m[7] * m[7] + m[8] * m[8]);
  if (nrm > 1e-6f) {
    m[6] /= nrm;
    m[7] /= nrm;
    m[8] /= nrm;
  }

  // Invert the 3x3 stain matrix on the host.
  const float det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                    m[1] * (m[3] * m[8] - m[5] * m[6]) +
                    m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (std::abs(det) < 1e-12f) {
    throw std::runtime_error(
        "ColorDeconvolveRgb: stain matrix is singular, cannot invert");
  }
  const float inv_det = 1.0f / det;
  std::array<float, 9> inv{};
  inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
  inv[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
  inv[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
  std::fprintf(stderr, "[CDR] inverted matrix\n"); std::fflush(stderr);

  // Upload the inverse to a small device buffer. This is safer than
  // cudaMemcpyToSymbolAsync (which segfaulted on Colab's runtime).
  std::fprintf(stderr, "[CDR] about to AsStream\n"); std::fflush(stderr);
  cudaStream_t s = AsStream(stream);
  std::fprintf(stderr, "[CDR] AsStream returned stream=%p\n", (void*)s); std::fflush(stderr);
  float* d_stain_inv = nullptr;
  std::fprintf(stderr, "[CDR] about to cudaMalloc\n"); std::fflush(stderr);
  cudaError_t e1 = cudaMalloc(&d_stain_inv, sizeof(float) * 9);
  std::fprintf(stderr, "[CDR] cudaMalloc returned e1=%d (%s)\n", (int)e1, cudaGetErrorString(e1)); std::fflush(stderr);
  if (e1 != cudaSuccess) {
    throw std::runtime_error(
        std::string("ColorDeconvolveRgb: cudaMalloc failed: ") +
        cudaGetErrorString(e1));
  }
  cudaError_t e2 = cudaMemcpyAsync(d_stain_inv, inv.data(), sizeof(float) * 9,
                                   cudaMemcpyHostToDevice, s);
  if (e2 != cudaSuccess) {
    cudaFree(d_stain_inv);
    throw std::runtime_error(
        std::string("ColorDeconvolveRgb: cudaMemcpyAsync failed: ") +
        cudaGetErrorString(e2));
  }

  const std::size_t npix  = width * height;
  const int         block = 256;
  const int         grid  = static_cast<int>((npix + block - 1) / block);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);
  DeconvolveKernel<<<grid, block, 0, s>>>(d_in_rgb, d_stain_inv,
                                          d_out_stain_od, npix);
  cudaError_t e3 = cudaGetLastError();
  if (e3 != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_stain_inv);
    throw std::runtime_error(
        std::string("ColorDeconvolveRgb: kernel launch failed: ") +
        cudaGetErrorString(e3));
  }
  cudaEventRecord(stop, s);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_stain_inv);

  (void)num_stains;  // reserved for future use
  (void)num_streams;
  return elapsed_ms;
}

}  // namespace kernels
}  // namespace stainkit