// stainkit/include/stainkit/kernels/color_deconvolution.h
//
// Ruifrok & Johnston (2001) "Color deconvolution of histochemical reagents"
// implemented as a single CUDA kernel.
//
// Given:
//   * the input RGB image (normalised to [0, 1] floats, gamma-decoded),
//   * a 3x3 stain matrix whose columns are the unit RGB vectors for
//     hematoxylin, eosin and a third (residual) stain,
//
// the kernel produces the 2 (or 3) stain-concentration channels.
// Concentration values are in *optical density* units, i.e. negative
// logarithms of the (normalised, gamma-decoded) RGB intensities.

#ifndef STK_INCLUDE_STAINKIT_KERNELS_COLOR_DECONVOLUTION_H_
#define STK_INCLUDE_STAINKIT_KERNELS_COLOR_DECONVOLUTION_H_

#include <cstddef>

#include "stainkit/types.h"

namespace stainkit {
namespace kernels {

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
// Runs the deconvolution kernel on `num_streams` CUDA streams for an RGB
// image of `width x height` pixels. The host is responsible for staging
// the input in device memory (see Pipeline). The output `stain_od` buffer
// must hold `width * height * num_stains` floats.
//
// All pointers must be device pointers. The function returns the elapsed
// time in milliseconds as measured by CUDA events on the default stream.
float ColorDeconvolveRgb(const float* d_in_rgb, std::size_t width,
                         std::size_t height, const StainMatrix& matrix,
                         float* d_out_stain_od, int num_stains = 2,
                         int num_streams = 1, void* stream = nullptr);

}  // namespace kernels
}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_KERNELS_COLOR_DECONVOLUTION_H_
