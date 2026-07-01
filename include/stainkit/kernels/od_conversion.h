// stainkit/include/stainkit/kernels/od_conversion.h
//
// RGB <-> optical density conversions used by both the deconvolution and
// the reconstruction kernels. Kept in a dedicated header so the math is
// re-usable across the codebase.

#ifndef STK_INCLUDE_STAINKIT_KERNELS_OD_CONVERSION_H_
#define STK_INCLUDE_STAINKIT_KERNELS_OD_CONVERSION_H_

#include <cuda_runtime.h>
#include <cstddef>

namespace stainkit {
namespace kernels {

// ---------------------------------------------------------------------------
// RGB -> optical density, in-place safe (aliasing allowed only on float
// pointers of the same size).
// ---------------------------------------------------------------------------
// `d_io` must hold `width * height * 3` floats. Values below 1e-6 are
// clipped to 1e-6 to keep the logarithm bounded.
void RgbToOd(float* d_io, std::size_t width, std::size_t height,
             cudaStream_t stream = 0);

// Same as `RgbToOd` but with separate input/output buffers.
void RgbToOd(const float* d_in, float* d_out, std::size_t width,
             std::size_t height, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Optical density -> RGB.
// ---------------------------------------------------------------------------
// `d_out` must hold `width * height * 3` floats.
void OdToRgb(const float* d_in, float* d_out, std::size_t width,
             std::size_t height, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// RGB -> grayscale luminance (BT.709 coefficients).
// ---------------------------------------------------------------------------
void RgbToLuma(const float* d_in, float* d_out, std::size_t width,
               std::size_t height, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_KERNELS_OD_CONVERSION_H_