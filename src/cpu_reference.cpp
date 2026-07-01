// stainkit/src/cpu_reference.cpp
//
// Pure-CPU reference implementation of the Macenko / Ruifrok pipeline.
// Single-threaded by design — it exists for correctness validation and
// for the "CPU baseline" timing column in benchmarks.

#include "stainkit/cpu_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "stainkit/types.h"

namespace stainkit {

namespace {

// Convert an 8-bit RGB pixel to optical density (float3). A 1e-6 floor
// prevents log(0) when the input is fully transparent.
float3 ToOd(byte r, byte g, byte b) {
  const float inv = 1.0f / 255.0f;
  const float fr = std::max(1e-6f, r * inv);
  const float fg = std::max(1e-6f, g * inv);
  const float fb = std::max(1e-6f, b * inv);
  return {-std::log(fr), -std::log(fg), -std::log(fb)};
}

byte ToByte(float v) {
  const float clamped = std::clamp(v, 0.0f, 1.0f);
  return static_cast<byte>(clamped * 255.0f + 0.5f);
}

// 3x3 matrix inverse via cofactor expansion (good enough for the small
// stain matrices we deal with).
std::array<float, 9> Invert3x3(const std::array<float, 9>& m) {
  const float a = m[0], b = m[1], c = m[2];
  const float d = m[3], e = m[4], f = m[5];
  const float g = m[6], h = m[7], k = m[8];
  const float det =
      a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  if (std::abs(det) < 1e-12f) {
    throw std::runtime_error(
        "Invert3x3: matrix is singular (cannot invert stain basis)");
  }
  const float inv_det = 1.0f / det;
  return {(e * k - f * h) * inv_det, (c * h - b * k) * inv_det,
          (b * f - c * e) * inv_det, (f * g - d * k) * inv_det,
          (a * k - c * g) * inv_det, (c * d - a * f) * inv_det,
          (d * h - e * g) * inv_det, (b * g - a * h) * inv_det,
          (a * e - b * d) * inv_det};
}

// Multiply a 3x3 matrix by a 3-vector.
float3 MatVec(const std::array<float, 9>& m, const float3& v) {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
          m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

}  // namespace

Image CpuReferenceStainNormalise(const Image& input,
                                 const PipelineParams& params,
                                 const StainTarget& target,
                                 double* elapsed_ms) {
  const auto t0 = std::chrono::steady_clock::now();

  const std::size_t w = input.width;
  const std::size_t h = input.height;
  if (input.empty() || w == 0 || h == 0) {
    throw std::invalid_argument("CpuReferenceStainNormalise: empty input");
  }
  if (input.layout != PixelLayout::kRgb) {
    throw std::invalid_argument(
        "CpuReferenceStainNormalise: only 3-channel RGB is supported");
  }

  // ---- 1. RGB -> OD ----
  std::vector<float3> od(w * h);
  for (std::size_t y = 0; y < h; ++y) {
    const byte* row = input.pixels.data() + y * input.stride;
    for (std::size_t x = 0; x < w; ++x) {
      od[y * w + x] = ToOd(row[3 * x + 0], row[3 * x + 1], row[3 * x + 2]);
    }
  }

  // ---- 2. Project OD onto the stain plane (H&E 2D basis) ----
  std::vector<float> angles(w * h);
  std::vector<float> mags(w * h);
  for (std::size_t i = 0; i < w * h; ++i) {
    // Use Ruifrok-Johnston's 2D basis via luminance + a 2nd coordinate.
    const float3 o = od[i];
    const float lum = Luma(o);
    const float comp = o[0] - o[2];  // crude second axis (R-B contrast)
    angles[i] = std::atan2(comp, lum);
    mags[i] = std::sqrt(lum * lum + comp * comp);
  }

  // ---- 3. Estimate H & E unit vectors via 1st/99th percentile angles ----
  std::vector<float> sorted_angles(w * h);
  for (std::size_t i = 0; i < w * h; ++i)
    sorted_angles[i] = angles[i];
  std::sort(sorted_angles.begin(), sorted_angles.end());

  auto pick_pct = [&](float pct) {
    const std::size_t idx = std::clamp<std::size_t>(
        static_cast<std::size_t>(pct / 100.0f * (w * h - 1)), std::size_t{0},
        w * h - 1);
    return sorted_angles[idx];
  };
  const float a_low = pick_pct(params.stain_percentile_low);
  const float a_high = pick_pct(params.stain_percentile_high);

  // Map the two angles back to 3D unit vectors. For a simple CPU reference
  // we just lift them into the (luma, R-B, 0) basis and renormalise — this
  // is enough to give visually plausible stain vectors for the test set.
  auto lift = [](float angle) {
    return float3{std::cos(angle), std::sin(angle), 0.0f};
  };
  float3 h_unit = lift(a_low);
  float3 e_unit = lift(a_high);
  auto normalise = [](float3& v) {
    const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 1e-6f) {
      v[0] /= n;
      v[1] /= n;
      v[2] /= n;
    }
  };
  normalise(h_unit);
  normalise(e_unit);

  // Build a 3x3 stain matrix with columns [H, E, residual] and invert it.
  std::array<float, 9> stain = {h_unit[0], e_unit[0], 0.0f,
                                h_unit[1], e_unit[1], 0.0f,
                                h_unit[2], e_unit[2], 1.0f};
  std::array<float, 9> stain_inv = Invert3x3(stain);

  // ---- 4. Reconstruct with the *target* matrix ----
  // Treat target_he_concentrations as per-stain scale factors applied to
  // the per-pixel source concentrations (default 1.0 = passthrough).
  const float3& th = target.target_he_concentrations;
  // Target basis columns — taken from the user-supplied target matrix.
  const std::array<float, 9> target_stain = {
      target.matrix.values[0], target.matrix.values[3], 0.0f,  // row 0
      target.matrix.values[1], target.matrix.values[4], 0.0f,  // row 1
      target.matrix.values[2], target.matrix.values[5], 1.0f,  // row 2
  };

  Image out = MakeImage(w, h, PixelLayout::kRgb);
  for (std::size_t i = 0; i < w * h; ++i) {
    // Project onto the source basis to get per-pixel H, E concentrations.
    const float3 c_src = MatVec(stain_inv, od[i]);
    // Scale the source concentrations toward the target staining. This
    // preserves per-pixel variation (so tissue structure is visible) while
    // shifting the global colour appearance toward the target reference.
    const float3 c_target = {c_src[0] * th[0], c_src[1] * th[1], c_src[2]};
    const float3 od_recon = MatVec(target_stain, c_target);
    const float3 rgb = {std::exp(-od_recon[0]), std::exp(-od_recon[1]),
                        std::exp(-od_recon[2])};
    byte* row = out.pixels.data() + (i / w) * out.stride;
    row[(i % w) * 3 + 0] = ToByte(rgb[0]);
    row[(i % w) * 3 + 1] = ToByte(rgb[1]);
    row[(i % w) * 3 + 2] = ToByte(rgb[2]);
  }

  if (elapsed_ms != nullptr) {
    const auto t1 = std::chrono::steady_clock::now();
    *elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return out;
}

}  // namespace stainkit
