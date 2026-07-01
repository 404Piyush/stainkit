// stainkit/include/stainkit/types.h
//
// Core value types for the stainkit pipeline.
// ---------------------------------------------------------------------------
// This header defines the *device-agnostic* POD types that flow between the
// host-side pipeline, the CUDA kernels, and the (optional) Python bindings.
// They are intentionally minimal and trivially copyable so they can be
// uploaded/downloaded via cudaMemcpy in a single shot.

#ifndef STK_INCLUDE_STAINKIT_TYPES_H_
#define STK_INCLUDE_STAINKIT_TYPES_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stainkit {

// ---------------------------------------------------------------------------
// Basic numeric aliases
// ---------------------------------------------------------------------------
using byte = std::uint8_t;
using float2 = std::array<float, 2>;
using float3 = std::array<float, 3>;

// ---------------------------------------------------------------------------
// Pixel layouts
// ---------------------------------------------------------------------------
// `PixelLayout` is what the host actually carries. The CUDA kernels are
// templated on the layout so the same kernel source can deal with RGB and
// RGBA without a runtime branch.
enum class PixelLayout : std::uint8_t {
  kRgb = 3,
  kRgba = 4,
};

// ---------------------------------------------------------------------------
// `Image` — owns host memory for a single image.
// ---------------------------------------------------------------------------
// Semantics:
//   * `pixels` is row-major with `stride` bytes per row.
//   * `stride >= width * channels`. The extra tail padding exists so kernels
//     can use aligned loads even when width is not a multiple of 4.
//   * `data()` is a host pointer; device buffers are managed separately by
//     the pipeline. Callers must not assume pixel values are synchronised
//     between host and device.
struct Image {
  std::vector<byte> pixels;
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t stride = 0;
  PixelLayout layout = PixelLayout::kRgb;

  std::size_t channels() const noexcept {
    return static_cast<std::size_t>(layout);
  }

  std::size_t bytes_per_row() const noexcept { return stride; }

  std::size_t byte_size() const noexcept { return stride * height; }

  bool empty() const noexcept { return pixels.empty(); }
};

// ---------------------------------------------------------------------------
// Host-side helpers. Defined in src/types.cpp.
// ---------------------------------------------------------------------------
std::size_t ComputeStride(std::size_t width, std::size_t channels);

Image MakeImage(std::size_t width, std::size_t height, PixelLayout layout);

float3 ClampUnit(const float3& v) noexcept;

float Luma(const float3& v) noexcept;

// ---------------------------------------------------------------------------
// Stain vector — a 3x2 matrix describing the chromatic basis of H&E staining.
// ---------------------------------------------------------------------------
//   columns 0 and 1 are the unit (RGB) vectors of hematoxylin and eosin.
//   The matrix is stored row-major. Kernels expect it in device memory as
//   six contiguous `float` values.
struct StainMatrix {
  // [H_r, H_g, H_b, E_r, E_g, E_b]
  std::array<float, 6> values{};

  float3 hematoxylin() const noexcept {
    return {values[0], values[1], values[2]};
  }

  float3 eosin() const noexcept { return {values[3], values[4], values[5]}; }

  static StainMatrix Identity() noexcept {
    // Macenko's default basis if the user opts out of estimation.
    StainMatrix m;
    m.values = {0.5626f, 0.7201f, 0.4062f, 0.2159f, 0.8012f, 0.5581f};
    return m;
  }
};

// ---------------------------------------------------------------------------
// Target / reference staining — what the *output* image should look like.
// ---------------------------------------------------------------------------
struct StainTarget {
  float3 target_he_concentrations{{0.65f, 0.70f, 0.29f}};
  StainMatrix matrix = StainMatrix::Identity();
  float background_lab = 255.0f;
  std::string name = "default";
};

// ---------------------------------------------------------------------------
// Pipeline parameters — runtime-tunable knobs exposed to the CLI / Python.
// ---------------------------------------------------------------------------
struct PipelineParams {
  // Otsu tissue-mask controls.
  bool compute_tissue_mask = true;
  float otsu_smoothing_radius = 3.0f;
  float min_tissue_area_fraction = 0.05f;

  // Macenko controls.
  float stain_percentile_low = 1.0f;
  float stain_percentile_high = 99.0f;
  bool regularize_stain_matrix = true;

  // Memory / stream configuration.
  int num_streams = 4;
  bool use_pinned_memory = true;
  bool overlap_io_with_compute = true;
  std::size_t max_inflight_tiles = 8;

  // Output.
  bool write_visualisation = true;    // 3-panel before/after
  bool write_stain_od_image = false;  // the 2-channel OD map
  bool write_tissue_mask = true;
};

// ---------------------------------------------------------------------------
// Per-image benchmark record. Serialised to CSV by the runner.
// ---------------------------------------------------------------------------
struct BenchmarkRecord {
  std::string image_id;
  std::size_t width = 0;
  std::size_t height = 0;
  double load_ms = 0.0;
  double copy_h2d_ms = 0.0;
  double deconvolve_ms = 0.0;
  double normalise_ms = 0.0;
  double mask_ms = 0.0;
  double copy_d2h_ms = 0.0;
  double write_ms = 0.0;
  double total_ms = 0.0;
  double cpu_baseline_ms = 0.0;
};

}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_TYPES_H_
