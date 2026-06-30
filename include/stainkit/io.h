// stainkit/include/stainkit/io.h
//
// Image I/O — read/write on the host side. The pipeline always uses these
// helpers to land pixels on disk, so the CUDA side never has to think about
// PNG / JPEG / TIFF encoding.
//
// All functions are header-only thin wrappers around the single-header
// stb_image / stb_image_write libraries vendored under third_party/stb/.

#ifndef STK_INCLUDE_STAINKIT_IO_H_
#define STK_INCLUDE_STAINKIT_IO_H_

#include <filesystem>
#include <string>
#include <vector>

#include "stainkit/types.h"

namespace stainkit {

// ---------------------------------------------------------------------------
// Detection / capability queries
// ---------------------------------------------------------------------------
// Returns true if the file extension is one stainkit can decode.
bool IsSupportedImage(const std::filesystem::path& path);

// Returns true if the file extension is one stainkit can encode.
bool IsSupportedOutputImage(const std::filesystem::path& path);

// ---------------------------------------------------------------------------
// Host-side decoding
// ---------------------------------------------------------------------------
// Reads `path` into an `Image`. If `force_channels` is non-zero the decoder
// is asked to expand the image to that many channels (e.g. 3 to force RGB).
// Throws `std::runtime_error` on any failure.
Image ReadImage(const std::filesystem::path& path, int force_channels = 0);

// ---------------------------------------------------------------------------
// Host-side encoding
// ---------------------------------------------------------------------------
// Writes `image` to `path`. Format is inferred from the extension.
// `jpeg_quality` is only used for JPEG output (1..100).
// Throws `std::runtime_error` on any failure.
void WriteImage(const Image& image, const std::filesystem::path& path,
                int jpeg_quality = 92);

// ---------------------------------------------------------------------------
// Convenience helpers used by the CLI / benchmarks
// ---------------------------------------------------------------------------
// Returns the list of image files in `dir` (non-recursive). The order is
// sorted so that downstream runs are deterministic.
std::vector<std::filesystem::path> ListImagesIn(
    const std::filesystem::path& dir);

// Writes a 3-panel visualisation:
//   [ left  ] input image
//   [ centre] post-macenko normalised image
//   [ right ] tissue mask (white = tissue)
void WriteVisualisationPanel(const Image& input, const Image& normalised,
                             const Image& mask,
                             const std::filesystem::path& path);

}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_IO_H_
