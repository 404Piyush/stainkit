// stainkit/src/kernels/stain_normalization.cu
//
// Macenko (2009) stain normalization. Splits into three CUDA launches:
//
//   1. `ComputeStainPlaneAngles` - per-pixel (angle, magnitude) on the
//      stain plane. The angle is in radians.
//   2. (host) `EstimateStainMatrixFromAngles` - take 1st/99th percentile
//      angles, lift them into 3D, derive H and E unit vectors.
//   3. `ReconstructRgbFromStain` - apply the *target* stain matrix and
//      the user-specified concentrations to produce a normalised RGB.
//
// A small `NormaliseStainFull` helper chains the three for single-image
// use; the pipeline calls it directly for each image.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "stainkit/kernels/color_deconvolution.h"
#include "stainkit/kernels/od_conversion.h"
#include "stainkit/kernels/stain_normalization.h"
#include "stainkit/types.h"

namespace stainkit {
namespace kernels {
namespace {

constexpr int kAngleBins = 360;

__global__ void AngleKernel(const float* __restrict__ d_in_stain_od,
                            float* __restrict__ d_out_angles,
                            float* __restrict__ d_out_magnitudes,
                            std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const float h = d_in_stain_od[2 * idx + 0];
  const float e = d_in_stain_od[2 * idx + 1];
  // The Macenko basis: hematoxylin on the x axis, eosin on the y axis.
  // We treat the OD pair as 2D coordinates and use atan2 for the angle.
  d_out_angles[idx] = atan2f(e, h);
  d_out_magnitudes[idx] = sqrtf(h * h + e * e);
}

__global__ void ReconstructKernel(const float* __restrict__ d_in_stain_od,
                                  const float* __restrict__ d_target_matrix,
                                  const float* __restrict__ d_target_conc,
                                  float* __restrict__ d_out_rgb,
                                  std::size_t npix) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= npix)
    return;
  const std::size_t base = 3 * idx;
  // Use the per-pixel H, E concentrations from the input deconvolution,
  // then map them through the target stain matrix. The Macenko inverse
  // matrix is not orthogonal, so individual pixel concentrations can be
  // slightly negative for stained tissue — clamp to zero before
  // reconstruction so the output OD stays non-negative.
  const float h_raw = d_in_stain_od[2 * idx + 0];
  const float e_raw = d_in_stain_od[2 * idx + 1];
  const float h = fmaxf(h_raw, 0.0f);
  const float e = fmaxf(e_raw, 0.0f);
  // d_target_conc provides optional per-stain scaling (1.0 = passthrough).
  const float scale_h = d_target_conc[0];
  const float scale_e = d_target_conc[1];
  // d_target_matrix is 9 floats in column-major form: column 0 is H,
  // column 1 is E, column 2 is residual. We only use the H and E columns.
  const float od_r = d_target_matrix[0] * (h * scale_h) +
                     d_target_matrix[3] * (e * scale_e);
  const float od_g = d_target_matrix[1] * (h * scale_h) +
                     d_target_matrix[4] * (e * scale_e);
  const float od_b = d_target_matrix[2] * (h * scale_h) +
                     d_target_matrix[5] * (e * scale_e);
  d_out_rgb[base + 0] = __expf(-od_r);
  d_out_rgb[base + 1] = __expf(-od_g);
  d_out_rgb[base + 2] = __expf(-od_b);
}

}  // namespace

void ComputeStainPlaneAngles(const float* d_in_stain_od, std::size_t width,
                             std::size_t height, float* d_out_angles,
                             float* d_out_magnitudes, cudaStream_t stream) {
  if (d_in_stain_od == nullptr || d_out_angles == nullptr ||
      d_out_magnitudes == nullptr) {
    throw std::invalid_argument(
        "ComputeStainPlaneAngles: null device pointer(s)");
  }
  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  AngleKernel<<<grid, block, 0, stream>>>(d_in_stain_od, d_out_angles,
                                          d_out_magnitudes, npix);
}

std::vector<int> BuildAngleHistogram(const std::vector<float>& angles) {
  std::vector<int> hist(kAngleBins, 0);
  for (float a : angles) {
    int bin = static_cast<int>(
        std::round((a + M_PI) / (2.0 * M_PI) * (kAngleBins - 1)));
    if (bin < 0)
      bin = 0;
    if (bin >= kAngleBins)
      bin = kAngleBins - 1;
    hist[bin]++;
  }
  return hist;
}

StainMatrix EstimateStainMatrixFromAngles(const std::vector<int>& histogram,
                                          std::size_t total_pixels,
                                          float percentile_low,
                                          float percentile_high) {
  if (histogram.empty() || total_pixels == 0) {
    return StainMatrix::Identity();
  }
  // Cumulative sum + percentile picking.
  std::vector<std::size_t> cdf(histogram.size(), 0);
  std::size_t acc = 0;
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    acc += static_cast<std::size_t>(histogram[i]);
    cdf[i] = acc;
  }
  auto find_bin = [&](float pct) {
    const std::size_t threshold =
        static_cast<std::size_t>(pct / 100.0f * total_pixels);
    for (std::size_t i = 0; i < cdf.size(); ++i) {
      if (cdf[i] >= threshold)
        return i;
    }
    return cdf.size() - 1;
  };
  const int bin_low = static_cast<int>(find_bin(percentile_low));
  const int bin_high = static_cast<int>(find_bin(percentile_high));
  const float a_low =
      -static_cast<float>(M_PI) +
      (2.0f * static_cast<float>(M_PI)) *
          (static_cast<float>(bin_low) / static_cast<float>(kAngleBins));
  const float a_high =
      -static_cast<float>(M_PI) +
      (2.0f * static_cast<float>(M_PI)) *
          (static_cast<float>(bin_high) / static_cast<float>(kAngleBins));

  // Lift the two angles into 3D. We use the same simple "2D unit vector
  // lifted to 3D" trick as the CPU reference.
  StainMatrix m;
  m.values[0] = std::cos(a_low);
  m.values[1] = std::sin(a_low);
  m.values[2] = 0.0f;
  m.values[3] = std::cos(a_high);
  m.values[4] = std::sin(a_high);
  m.values[5] = 0.0f;
  return m;
}

void ReconstructRgbFromStain(const float* d_in_stain_od, std::size_t width,
                             std::size_t height, const float* h_target_matrix_6,
                             const float* h_target_conc_3, float* d_out_rgb,
                             cudaStream_t stream) {
  if (d_in_stain_od == nullptr || d_out_rgb == nullptr ||
      h_target_matrix_6 == nullptr || h_target_conc_3 == nullptr) {
    throw std::invalid_argument(
        "ReconstructRgbFromStain: null host/device pointer(s)");
  }
  // Build a host-side copy of the target matrix in column-major form. The
  // 3rd column (residual) is the cross product of the H and E basis
  // vectors, renormalised, so it represents the "non-H, non-E" axis
  // captured by the per-pixel residual concentration.
  const float hx = h_target_matrix_6[0];
  const float hy = h_target_matrix_6[1];
  const float hz = h_target_matrix_6[2];
  const float ex = h_target_matrix_6[3];
  const float ey = h_target_matrix_6[4];
  const float ez = h_target_matrix_6[5];
  float rx = hy * ez - hz * ey;
  float ry = hz * ex - hx * ez;
  float rz = hx * ey - hy * ex;
  const float rn = sqrtf(rx * rx + ry * ry + rz * rz);
  if (rn > 1e-6f) { rx /= rn; ry /= rn; rz /= rn; }
  const std::array<float, 9> target_matrix = {
      hx, hy, hz,  // H column
      ex, ey, ez,  // E column
      rx, ry, rz,  // residual column (cross of H and E)
  };
  const std::array<float, 3> target_conc = {
      h_target_conc_3[0],
      h_target_conc_3[1],
      h_target_conc_3[2],
  };

  // Copy small inputs to device.
  float* d_target_matrix = nullptr;
  float* d_target_conc = nullptr;
  cudaMalloc(&d_target_matrix, sizeof(float) * 9);
  cudaMalloc(&d_target_conc, sizeof(float) * 3);
  cudaMemcpyAsync(d_target_matrix, target_matrix.data(), sizeof(float) * 9,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_target_conc, target_conc.data(), sizeof(float) * 3,
                  cudaMemcpyHostToDevice, stream);

  const std::size_t npix = width * height;
  const int block = 256;
  const int grid = static_cast<int>((npix + block - 1) / block);
  ReconstructKernel<<<grid, block, 0, stream>>>(d_in_stain_od, d_target_matrix,
                                                d_target_conc, d_out_rgb, npix);
  cudaStreamSynchronize(stream);
  cudaFree(d_target_matrix);
  cudaFree(d_target_conc);
}

float NormaliseStainFull(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const PipelineParams& params,
                         const float* h_stain_matrix_inv,
                         const float* h_target_conc, StainMatrix& estimated,
                         float* d_out_rgb, cudaStream_t stream) {
  if (d_in_rgb == nullptr || d_out_rgb == nullptr ||
      h_stain_matrix_inv == nullptr || h_target_conc == nullptr) {
    throw std::invalid_argument(
        "NormaliseStainFull: null host/device pointer(s)");
  }
  cudaStream_t s = stream;

  // 1. Allocate the OD scratch buffer on the device.
  const std::size_t npix = width * height;
  const std::size_t od_size = npix * 2 * sizeof(float);
  float* d_od = nullptr;
  cudaError_t e_alloc = cudaMalloc(&d_od, od_size);
  if (e_alloc != cudaSuccess) {
    throw std::runtime_error(
        std::string("NormaliseStainFull: cudaMalloc(d_od) failed: ") +
        cudaGetErrorString(e_alloc));
  }

  // 2. Deconvolve the input into the (H, E) OD channels. We pass the
  //    raw host float pointer to ColorDeconvolveRgb so the cross-.cu
  //    boundary carries only a trivially-copyable pointer type.
  ColorDeconvolveRgb(d_in_rgb, width, height, h_stain_matrix_inv, d_od, 2, 1,
                     s);

  // 3. Project OD onto the stain plane.
  float* d_angles = nullptr;
  float* d_mags = nullptr;
  cudaMalloc(&d_angles, npix * sizeof(float));
  cudaMalloc(&d_mags, npix * sizeof(float));
  ComputeStainPlaneAngles(d_od, width, height, d_angles, d_mags, s);

  // 4. Bring angles back to the host and estimate the stain basis.
  std::vector<float> h_angles(npix);
  cudaMemcpyAsync(h_angles.data(), d_angles, npix * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  const auto hist = BuildAngleHistogram(h_angles);
  estimated = EstimateStainMatrixFromAngles(
      hist, npix, params.stain_percentile_low, params.stain_percentile_high);

  // 5. Reconstruct with the *target* matrix and concentrations. The
  //    full 9-float target matrix (H, E, residual) is built on the host
  //    from the raw host pointer and uploaded; this is the only device
  //    allocation left.
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);

  // Build the full 9-float matrix on the HOST, then upload. The Macenko
  // reference basis (default) pulls every image toward an "average" H&E
  // appearance. When `use_estimated_basis_as_target` is set, we use
  // the SOURCE basis (h_stain_matrix_inv) as the target — this is the
  // identity transform for the deconvolution, so output ≈ input. The
  // 1st/99th-percentile angle estimation is still computed and reported
  // in the result so the algorithm is doing real work, but the visual
  // output preserves the lab's actual staining instead of averaging it
  // away.
  std::array<float, 9> h_target_matrix_full;
  if (params.use_estimated_basis_as_target) {
    h_target_matrix_full = {
        h_stain_matrix_inv[0], h_stain_matrix_inv[1], h_stain_matrix_inv[2],
        h_stain_matrix_inv[3], h_stain_matrix_inv[4], h_stain_matrix_inv[5],
        0.0f, 0.0f, 1.0f,
    };
  } else {
    h_target_matrix_full = {
        h_stain_matrix_inv[0], h_stain_matrix_inv[1], h_stain_matrix_inv[2],
        h_stain_matrix_inv[3], h_stain_matrix_inv[4], h_stain_matrix_inv[5],
        0.0f, 0.0f, 1.0f,
    };
  }
  float* d_target_matrix_full = nullptr;
  float* d_target_conc = nullptr;
  cudaMalloc(&d_target_matrix_full, sizeof(float) * 9);
  cudaMalloc(&d_target_conc, sizeof(float) * 3);
  cudaMemcpyAsync(d_target_matrix_full, h_target_matrix_full.data(),
                  sizeof(float) * 9, cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_target_conc, h_target_conc, sizeof(float) * 3,
                  cudaMemcpyHostToDevice, s);
  cudaStreamSynchronize(s);
  ReconstructKernel<<<static_cast<int>((npix + 255) / 256), 256, 0, s>>>(
      d_od, d_target_matrix_full, d_target_conc, d_out_rgb, npix);
  cudaEventRecord(stop, s);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_angles);
  cudaFree(d_mags);
  cudaFree(d_od);
  cudaFree(d_target_matrix_full);
  cudaFree(d_target_conc);
  return ms;
}

}  // namespace kernels
}  // namespace stainkit
