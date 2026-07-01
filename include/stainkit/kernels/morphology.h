// stainkit/include/stainkit/kernels/morphology.h
//
// Morphological operations used to clean up the tissue mask: opening and
// closing with a flat disc structuring element.

#ifndef STK_INCLUDE_STAINKIT_KERNELS_MORPHOLOGY_H_
#define STK_INCLUDE_STAINKIT_KERNELS_MORPHOLOGY_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace stainkit {
namespace kernels {

// ---------------------------------------------------------------------------
// Binary erosion with a disc structuring element of given radius.
// ---------------------------------------------------------------------------
// Operates in-place on `d_io`. `radius` is in pixels.
void Erode(std::uint8_t* d_io, std::size_t width, std::size_t height,
           std::size_t radius, cudaStream_t stream = 0);

// Binary dilation with a disc structuring element of given radius.
void Dilate(std::uint8_t* d_io, std::size_t width, std::size_t height,
            std::size_t radius, cudaStream_t stream = 0);

// Opening = erode + dilate. Disables itself for radius == 0.
void Open(std::uint8_t* d_io, std::size_t width, std::size_t height,
          std::size_t radius, cudaStream_t stream = 0);

// Closing = dilate + erode.
void Close(std::uint8_t* d_io, std::size_t width, std::size_t height,
           std::size_t radius, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_KERNELS_MORPHOLOGY_H_