// stainkit/src/kernels/color_deconvolution.cu
//
// Ruifrok & Johnston (2001) color deconvolution kernel.
//   - Input:  per-pixel RGB in [0, 1].
//   - Output: per-pixel OD for each stain channel.
//
// The math reduces to a single 3x3 matrix multiply per pixel. We keep
// the stain matrix in constant memory so the entire deconvolution fits
// in a few hundred bytes.

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "stainkit/kernels/color_deconvolution.h"
#include "stainkit/types.h"

namespace stainkit {
namespace kernels {
namespace {

// Two copies: one for the (R,G,B)-column-major basis, one for the
// pixel-format stain matrix. Kept in __constant__ memory so the SM can
// broadcast it.
__constant__ float cStain[9];

__global__ void DeconvolveKernel(const float* __restrict__ d_in,
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

  // Apply the pseudo-inverse stain matrix. We use the "complete" 3x3 form
  // even when only H and E are wanted so a future residual channel can be
  // exposed without a code change.
  const float c0 = cStain[0] * od_r + cStain[1] * od_g + cStain[2] * od_b;
  const float c1 = cStain[3] * od_r + cStain[4] * od_g + cStain[5] * od_b;
  const float c2 = cStain[6] * od_r + cStain[7] * od_g + cStain[8] * od_b;

  d_out[2 * idx + 0] = c0;
  d_out[2 * idx + 1] = c1;
  if (idx == 0) {
    // (the residual channel is dropped on output; keep slot for the future)
    (void)c2;
  }
}

inline cudaStream_t AsStream(void* s) {
  return (s == nullptr) ? 0 : *reinterpret_cast<cudaStream_t*>(&s);
}

}  // namespace

float ColorDeconvolveRgb(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const StainMatrix& matrix,
                         float* d_out_stain_od, int num_stains,
                         int num_streams, void* stream) {
  if (d_in_rgb == nullptr || d_out_stain_od == nullptr) {
    throw std::invalid_argument("ColorDeconvolveRgb: null device pointer");
  }
  if (width == 0 || height == 0) {
    throw std::invalid_argument("ColorDeconvolveRgb: zero-sized image");
  }

  // Build a 3x3 stain matrix on the host. We use the (R, G, B)-column-major
  // form so that the kernel's `cStain` matches a row-major constant array.
  // The third column is taken as the cross product of H and E to give a
  // orthonormal "residual" basis.
  std::array<float, 9> m{};
  m[0] = matrix.values[0];
  m[1] = matrix.values[1];
  m[2] = matrix.values[2];
  m[3] = matrix.values[3];
  m[4] = matrix.values[4];
  m[5] = matrix.values[5];
  // Cross product (H x E) for the third column.
  const float hx = m[0], hy = m[1], hz = m[2];
  const float ex = m[3], ey = m[4], ez = m[5];
  m[6] = hy * ez - hz * ey;
  m[7] = hz * ex - hx * ez;
  m[8] = hx * ey - hy * ex;
  // Normalise the residual.
  const float n = std::sqrt(m[6] * m[6] + m[7] * m[7] + m[8] * m[8]);
  if (n > 1e-6f) {
    m[6] /= n;
    m[7] /= n;
    m[8] /= n;
  }

  // Invert the 3x3 stain matrix on the host (we only need this once per
  // pipeline invocation).
  // (We compute it via the same cofactor expansion as the CPU reference.)
  auto det3 = [](const std::array<float, 9>& a) {
    return a[0] * (a[4] * a[8] - a[5] * a[7]) -
           a[1] * (a[3] * a[8] - a[5] * a[6]) +
           a[2] * (a[3] * a[7] - a[4] * a[6]);
  };
  const float det = det3(m);
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

  // Upload the inverse to constant memory.
  cudaStream_t s = AsStream(stream);
  cudaMemcpyToSymbolAsync(cStain, inv.data(), sizeof(float) * 9, 0,
                          cudaMemcpyHostToDevice, s);

  // Run the kernel.
  const std::size_t npix  = width * height;
  const int         block = 256;
  const int         grid  = static_cast<int>((npix + block - 1) / block);

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);
  DeconvolveKernel<<<grid, block, 0, s>>>(d_in_rgb, d_out_stain_od, npix);
  cudaEventRecord(stop, s);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  (void)num_stains;  // reserved for future use
  (void)num_streams;
  return elapsed_ms;
}

}  // namespace kernels
}  // namespace stainkit
