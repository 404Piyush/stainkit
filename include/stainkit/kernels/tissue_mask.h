// stainkit/include/stainkit/kernels/tissue_mask.h
//
// Otsu-based tissue mask kernel.
//
// We compute a 1-channel intensity image (luminance) and then run Otsu's
// method on a smoothed histogram. The output is a binary mask where
// 255 = tissue and 0 = background/glass.

#ifndef STK_INCLUDE_STAINKIT_KERNELS_TISSUE_MASK_H_
#define STK_INCLUDE_STAINKIT_KERNELS_TISSUE_MASK_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stainkit {
namespace kernels {

// ---------------------------------------------------------------------------
// Luminance reduction
// ---------------------------------------------------------------------------
// Converts RGB (float, 3 channels) to a 1-channel luminance image using
// the BT.709 coefficients.
void RgbToLuminance(const float* d_in_rgb, std::size_t width,
                    std::size_t height, float* d_out_lum,
                    cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Otsu's threshold on a luminance image (host side after copy-back).
// ---------------------------------------------------------------------------
// The luminance image is expected to be on the device; this launcher will
// copy it to the host, build the histogram, run Otsu's method, and copy
// the resulting threshold back. For very large images a device-side
// histogram is preferred — see the device-side variant below.
float OtsuThresholdHost(const float* d_in_lum, std::size_t width,
                        std::size_t height, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Device-side histogram and Otsu threshold (preferred for large images).
// ---------------------------------------------------------------------------
float OtsuThresholdDevice(const float* d_in_lum, std::size_t width,
                          std::size_t height, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Binarisation + morphological post-processing.
// ---------------------------------------------------------------------------
// `radius` is the morphological opening radius in pixels (0 disables the
// post-processing pass).
void ThresholdToMask(const float* d_in_lum, std::size_t width,
                     std::size_t height, float threshold, std::size_t radius,
                     std::uint8_t* d_out_mask, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_KERNELS_TISSUE_MASK_H_