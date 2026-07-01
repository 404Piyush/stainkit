// stainkit/include/stainkit/kernels/stain_normalization.h
//
// Macenko et al. (2009) "A reference image set for H&E stain normalisation"
// GPU implementation.
//
// Pipeline:
//   1. Read the optical-density image (output of color_deconvolution).
//   2. Project OD onto the plane of unit vectors.
//   3. Estimate the H&E basis via 1st / 99th percentile angles.
//   4. Reconstruct RGB using a *target* stain matrix (and a target
//      concentration vector when the user specifies one).
//
// Step 3 is *sequential per image* because it depends on a reduction over
// all pixel angles. We expose it as a separate kernel pair
// (computeAngles + estimateStainMatrixFromAngles) so it can be run once
// per batch and amortised.

#ifndef STK_INCLUDE_STAINKIT_KERNELS_STAIN_NORMALIZATION_H_
#define STK_INCLUDE_STAINKIT_KERNELS_STAIN_NORMALIZATION_H_

#include <cuda_runtime.h>
#include <cstddef>

#include "stainkit/types.h"

namespace stainkit {
namespace kernels {

// ---------------------------------------------------------------------------
// Stage 1: project OD plane and write (angle, magnitude) per pixel.
// ---------------------------------------------------------------------------
// Output `d_angles` is in radians, range [-pi, pi].
// Output `d_magnitudes` is non-negative.
void ComputeStainPlaneAngles(const float* d_in_stain_od, std::size_t width,
                             std::size_t height, float* d_out_angles,
                             float* d_out_magnitudes,
                             cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Stage 2: estimate the 3x2 stain matrix from the angle histogram.
// ---------------------------------------------------------------------------
// Reads the *histogram* of stain plane angles (host-resident), takes
// the 1st/99th percentile angles, and produces the unit vectors for
// hematoxylin and eosin. The third column is taken as the orthogonal
// complement (Ruifrok-Johnston residual) when `target_matrix` is
// supplied; otherwise the user can leave it zeroed.
StainMatrix EstimateStainMatrixFromAngles(
    const std::vector<int>& histogram, std::size_t total_pixels,
    float percentile_low, float percentile_high);

// Helper that builds the histogram on the host side (cheap, ~360 bins).
std::vector<int> BuildAngleHistogram(const std::vector<float>& angles);

// ---------------------------------------------------------------------------
// Stage 3: reconstruct RGB with the target stain matrix.
// ---------------------------------------------------------------------------
// `d_in_stain_od` is the 2-channel OD image.
// `h_target_matrix_6` is a HOST pointer to six floats (H_r, H_g, H_b,
// E_r, E_g, E_b) — the target stain basis. Raw pointers are used (rather
// than StainTarget&) to avoid cross-TU ABI hazards.
// `h_target_conc_3` is a HOST pointer to three floats (H conc, E conc,
// residual conc).
// `d_out_rgb` must hold `width * height * 3` floats.
void ReconstructRgbFromStain(const float* d_in_stain_od, std::size_t width,
                             std::size_t height, const float* h_target_matrix_6,
                             const float* h_target_conc_3, float* d_out_rgb,
                             cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// One-shot helper that runs the full pipeline for a single image.
// ---------------------------------------------------------------------------
// Useful when the caller has already deconvolved and just wants to swap
// to a target basis. Internally synchronises on the supplied stream.
//
// `h_stain_matrix_inv` is a HOST pointer to a 6-float array
// (H_r, H_g, H_b, E_r, E_g, E_b).
// `h_target_conc` is a HOST pointer to a 3-float array (H concentration,
// E concentration, residual).
// They are uploaded to device internally so the g++/nvcc ABI boundary
// only carries trivially-copyable float pointers (StainTarget contains
// std::string and would otherwise ABI-mismatch).
float NormaliseStainFull(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const PipelineParams& params,
                         const float* h_stain_matrix_inv,
                         const float* h_target_conc,
                         StainMatrix& estimated,
                         float* d_out_rgb, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_KERNELS_STAIN_NORMALIZATION_H_
