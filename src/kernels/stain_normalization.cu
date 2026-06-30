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
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (idx >= npix) return;
  const float h = d_in_stain_od[2 * idx + 0];
  const float e = d_in_stain_od[2 * idx + 1];
  // The Macenko basis: hematoxylin on the x axis, eosin on the y axis.
  // We treat the OD pair as 2D coordinates and use atan2 for the angle.
  d_out_angles[idx]     = atan2f(e, h);
  d_out_magnitudes[idx] = sqrtf(h * h + e * e);
}

__global__ void ReconstructKernel(const float* __restrict__ d_in_stain_od,
                                  const float* __restrict__ d_target_matrix,
                                  const float* __restrict__ d_target_conc,
                                  float* __restrict__ d_out_rgb,
                                  std::size_t npix) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  if (idx >= npix) return;
  const std::size_t base = 3 * idx;
  const float h = d_in_stain_od[2 * idx + 0];
  const float e = d_in_stain_od[2 * idx + 1];
  // Use the user-supplied concentrations for H and E, and reuse the
  // original residual for the third channel.
  const float c0 = d_target_conc[0];
  const float c1 = d_target_conc[1];
  // d_target_matrix is 9 floats: columns are H, E, residual.
  const float od_r = d_target_matrix[0] * c0 + d_target_matrix[3] * c1;
  const float od_g = d_target_matrix[1] * c0 + d_target_matrix[4] * c1;
  const float od_b = d_target_matrix[2] * c0 + d_target_matrix[5] * c1;
  d_out_rgb[base + 0] = __expf(-od_r);
  d_out_rgb[base + 1] = __expf(-od_g);
  d_out_rgb[base + 2] = __expf(-od_b);
}

inline cudaStream_t AsStream(void* s) {
  return (s == nullptr) ? 0 : *reinterpret_cast<cudaStream_t*>(&s);
}

}  // namespace

void ComputeStainPlaneAngles(const float* d_in_stain_od, std::size_t width,
                             std::size_t height, float* d_out_angles,
                             float* d_out_magnitudes, void* stream) {
  if (d_in_stain_od == nullptr || d_out_angles == nullptr ||
      d_out_magnitudes == nullptr) {
    throw std::invalid_argument(
        "ComputeStainPlaneAngles: null device pointer(s)");
  }
  const std::size_t npix  = width * height;
  const int         block = 256;
  const int         grid  = static_cast<int>((npix + block - 1) / block);
  cudaStream_t      s     = AsStream(stream);
  AngleKernel<<<grid, block, 0, s>>>(d_in_stain_od, d_out_angles,
                                      d_out_magnitudes, npix);
}

std::vector<int> BuildAngleHistogram(const std::vector<float>& angles) {
  std::vector<int> hist(kAngleBins, 0);
  for (float a : angles) {
    int bin = static_cast<int>(std::round((a + M_PI) /
                                          (2.0 * M_PI) * (kAngleBins - 1)));
    if (bin < 0) bin = 0;
    if (bin >= kAngleBins) bin = kAngleBins - 1;
    hist[bin]++;
  }
  return hist;
}

StainMatrix EstimateStainMatrixFromAngles(
    const std::vector<int>& histogram, std::size_t total_pixels,
    float percentile_low, float percentile_high,
    const StainMatrix& target_matrix) {
  (void)target_matrix;
  if (histogram.empty() || total_pixels == 0) {
    return StainMatrix::Identity();
  }
  // Cumulative sum + percentile picking.
  std::vector<std::size_t> cdf(histogram.size(), 0);
  std::size_t              acc = 0;
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    acc += static_cast<std::size_t>(histogram[i]);
    cdf[i] = acc;
  }
  auto find_bin = [&](float pct) {
    const std::size_t threshold =
        static_cast<std::size_t>(pct / 100.0f * total_pixels);
    for (std::size_t i = 0; i < cdf.size(); ++i) {
      if (cdf[i] >= threshold) return i;
    }
    return cdf.size() - 1;
  };
  const int bin_low  = static_cast<int>(find_bin(percentile_low));
  const int bin_high = static_cast<int>(find_bin(percentile_high));
  const float a_low  = -static_cast<float>(M_PI) +
                      (2.0f * static_cast<float>(M_PI)) *
                          (static_cast<float>(bin_low) /
                           static_cast<float>(kAngleBins));
  const float a_high = -static_cast<float>(M_PI) +
                       (2.0f * static_cast<float>(M_PI)) *
                           (static_cast<float>(bin_high) /
                            static_cast<float>(kAngleBins));

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
                             std::size_t height, const StainTarget& target,
                             float* d_out_rgb, void* stream) {
  if (d_in_stain_od == nullptr || d_out_rgb == nullptr) {
    throw std::invalid_argument(
        "ReconstructRgbFromStain: null device pointer(s)");
  }
  // Build a host-side copy of the target matrix in column-major form.
  const std::array<float, 9> target_matrix = {
      target.matrix.values[0], target.matrix.values[1], target.matrix.values[2],
      target.matrix.values[3], target.matrix.values[4], target.matrix.values[5],
      0.0f, 0.0f, 1.0f,  // residual: identity, but unused in 2-channel mode
  };
  const std::array<float, 3> target_conc = {
      target.target_he_concentrations[0],
      target.target_he_concentrations[1],
      target.target_he_concentrations[2],
  };

  // Copy small inputs to device.
  float*       d_target_matrix = nullptr;
  float*       d_target_conc   = nullptr;
  cudaMalloc(&d_target_matrix, sizeof(float) * 9);
  cudaMalloc(&d_target_conc, sizeof(float) * 3);
  cudaStream_t s = AsStream(stream);
  cudaMemcpyAsync(d_target_matrix, target_matrix.data(), sizeof(float) * 9,
                  cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_target_conc, target_conc.data(), sizeof(float) * 3,
                  cudaMemcpyHostToDevice, s);

  const std::size_t npix  = width * height;
  const int         block = 256;
  const int         grid  = static_cast<int>((npix + block - 1) / block);
  ReconstructKernel<<<grid, block, 0, s>>>(d_in_stain_od, d_target_matrix,
                                          d_target_conc, d_out_rgb, npix);
  cudaStreamSynchronize(s);
  cudaFree(d_target_matrix);
  cudaFree(d_target_conc);
}

float NormaliseStainFull(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const PipelineParams& params,
                         const StainTarget& target, StainMatrix& estimated,
                         float* d_out_rgb, void* stream) {
  std::fprintf(stderr, "[NS] enter\n"); std::fflush(stderr);
  if (d_in_rgb == nullptr || d_out_rgb == nullptr) {
    throw std::invalid_argument("NormaliseStainFull: null device pointer(s)");
  }
  cudaStream_t s = AsStream(stream);
  std::fprintf(stderr, "[NS] got stream\n"); std::fflush(stderr);

  // 1. Deconvolve the input into the (H, E) OD channels.
  //    We reuse the same 3x3 stain matrix as the target basis for the
  //    initial deconvolution; the estimated basis from stage 2 only
  //    affects the *target* matrix in the reconstruction.
  std::array<float, 9> stain = {
      target.matrix.values[0], target.matrix.values[1], target.matrix.values[2],
      target.matrix.values[3], target.matrix.values[4], target.matrix.values[5],
      0.0f, 0.0f, 1.0f,
  };
  // Allocate a scratch OD buffer (2 channels).
  const std::size_t npix    = width * height;
  const std::size_t od_size = npix * 2 * sizeof(float);
  float*            d_od    = nullptr;
  std::fprintf(stderr, "[NS] cudaMalloc d_od\n"); std::fflush(stderr);
  cudaMalloc(&d_od, od_size);
  std::fprintf(stderr, "[NS] ColorDeconvolveRgb\n"); std::fflush(stderr);
  ColorDeconvolveRgb(d_in_rgb, width, height, target.matrix, d_od, 2, 1, &s);
  std::fprintf(stderr, "[NS] ColorDeconv ok\n"); std::fflush(stderr);
  (void)stain;  // (kept for documentation; basis is in constant memory)

  // 2. Compute (angle, magnitude) per pixel.
  float* d_angles = nullptr;
  float* d_mags   = nullptr;
  cudaMalloc(&d_angles, npix * sizeof(float));
  cudaMalloc(&d_mags, npix * sizeof(float));
  std::fprintf(stderr, "[NS] ComputeStainPlaneAngles\n"); std::fflush(stderr);
  ComputeStainPlaneAngles(d_od, width, height, d_angles, d_mags, &s);
  std::fprintf(stderr, "[NS] ComputeAngles ok\n"); std::fflush(stderr);

  // 3. Copy the angles back to the host, build the histogram and estimate
  //    the stain basis. For the *first* iteration of the algorithm we use
  //    the user-supplied target matrix; on subsequent iterations the
  //    estimated basis would replace it. We keep it simple here and
  //    always reuse the target.
std::vector<float> h_angles(npix);
  std::fprintf(stderr, "[NS] memcpy angles\n"); std::fflush(stderr);
  cudaMemcpyAsync(h_angles.data(), d_angles, npix * sizeof(float),
                  cudaMemcpyDeviceToHost, s);
  std::fprintf(stderr, "[NS] sync\n"); std::fflush(stderr);
  cudaStreamSynchronize(s);
  std::fprintf(stderr, "[NS] sync done\n"); std::fflush(stderr);
  const auto hist = BuildAngleHistogram(h_angles);
  std::fprintf(stderr, "[NS] histogram\n"); std::fflush(stderr);
  estimated       = EstimateStainMatrixFromAngles(
      hist, npix, params.stain_percentile_low, params.stain_percentile_high,
      target.matrix);
  std::fprintf(stderr, "[NS] estimated\n"); std::fflush(stderr);

  // 4. Reconstruct using the target matrix + concentrations.
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);
  std::fprintf(stderr, "[NS] Reconstruct\n"); std::fflush(stderr);
  ReconstructRgbFromStain(d_od, width, height, target, d_out_rgb, &s);
  std::fprintf(stderr, "[NS] Reconstruct ok\n"); std::fflush(stderr);
  cudaEventRecord(stop, s);
  cudaStreamSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_angles);
  cudaFree(d_mags);
  cudaFree(d_od);
  std::fprintf(stderr, "[NS] done\n"); std::fflush(stderr);
  return ms;
}

}  // namespace kernels
}  // namespace stainkit
