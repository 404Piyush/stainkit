// stainkit/src/pipeline.cpp
//
// The top-level orchestration. Owns the CUDA context, manages device
// memory, launches kernels in a multi-stream fashion, and exposes the
// `Pipeline` class to the CLI / Python bindings.

#include "stainkit/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "stainkit/cpu_reference.h"
#include "stainkit/kernels/color_deconvolution.h"
#include "stainkit/kernels/morphology.h"
#include "stainkit/kernels/od_conversion.h"
#include "stainkit/kernels/stain_normalization.h"
#include "stainkit/kernels/tissue_mask.h"
#include "stainkit/types.h"

namespace stainkit {

// ---------------------------------------------------------------------------
// CudaContext — owns the per-process CUDA state.
// ---------------------------------------------------------------------------
class CudaContext {
 public:
  CudaContext() {
    int device_count = 0;
    cudaError_t err    = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      throw std::runtime_error(
          std::string("CudaContext: no CUDA devices available "
                      "(cudaGetDeviceCount returned ") +
          std::to_string(static_cast<int>(err)) +
          " with " + std::to_string(device_count) + " devices)");
    }
    cudaError_t set_err = cudaSetDevice(0);
    if (set_err != cudaSuccess) {
      throw std::runtime_error(
          std::string("CudaContext: cudaSetDevice(0) failed: ") +
          cudaGetErrorString(set_err));
    }
    cudaDeviceProp prop{};
    std::memset(&prop, 0, sizeof(prop));
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
    if (prop_err != cudaSuccess) {
      throw std::runtime_error(
          std::string("CudaContext: cudaGetDeviceProperties failed: ") +
          cudaGetErrorString(prop_err));
    }
    device_name_ = prop.name;

    // Per-stream events used for stage timing.
    for (int i = 0; i < kMaxStreams; ++i) {
      streams_[i] = nullptr;
      cudaError_t stream_err = cudaStreamCreate(&streams_[i]);
      if (stream_err != cudaSuccess) {
        // Roll back partial initialisation before throwing.
        for (int j = 0; j < i; ++j) {
          cudaStreamDestroy(streams_[j]);
          streams_[j] = nullptr;
        }
        throw std::runtime_error(
            std::string("CudaContext: cudaStreamCreate failed: ") +
            cudaGetErrorString(stream_err));
      }
    }
  }

  ~CudaContext() {
    for (int i = 0; i < kMaxStreams; ++i) {
      if (streams_[i] != nullptr) {
        cudaStreamDestroy(streams_[i]);
      }
    }
  }

  CudaContext(const CudaContext&)            = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  const std::string& DeviceName() const noexcept { return device_name_; }

  cudaStream_t Stream(int i) const {
    if (i < 0 || i >= kMaxStreams) {
      throw std::out_of_range("CudaContext::Stream: bad index");
    }
    return streams_[i];
  }

  static constexpr int kMaxStreams = 8;

 private:
  std::string  device_name_;
  cudaStream_t streams_[kMaxStreams]{};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// RAII wrappers for device memory. These are tiny because the pipeline
// already pins the host buffer when the user requests it.
struct DeviceBuffer {
  void*  ptr = nullptr;
  size_t size = 0;

  explicit DeviceBuffer(size_t bytes) : size(bytes) {
    if (bytes == 0) return;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e != cudaSuccess) {
      throw std::runtime_error("DeviceBuffer: cudaMalloc failed");
    }
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
  void*  ptr = nullptr;
  size_t size = 0;

  explicit PinnedBuffer(size_t bytes) : size(bytes) {
    if (bytes == 0) return;
    cudaError_t e = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
      throw std::runtime_error("PinnedBuffer: cudaHostAlloc failed");
    }
  }
  ~PinnedBuffer() {
    if (ptr != nullptr) cudaFreeHost(ptr);
  }
  PinnedBuffer(const PinnedBuffer&)            = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// Stage timing helper.
struct ScopedEvent {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaStream_t stream{};
  float       elapsed_ms = 0.0f;

  explicit ScopedEvent(cudaStream_t s) : stream(s) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
  }
  ~ScopedEvent() {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  ScopedEvent(const ScopedEvent&)            = delete;
  ScopedEvent& operator=(const ScopedEvent&) = delete;
};

float3 AsFloat3(const std::array<float, 6>& v, int col) {
  return {v[3 * col + 0], v[3 * col + 1], v[3 * col + 2]};
}

}  // namespace

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
Pipeline::Pipeline() : ctx_(std::make_unique<CudaContext>()) {}

Pipeline::~Pipeline() = default;

bool Pipeline::IsCudaAvailable() noexcept {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  return (e == cudaSuccess && count > 0);
}

std::unique_ptr<Pipeline> Pipeline::Make() {
  try {
    return std::unique_ptr<Pipeline>(new Pipeline());
  } catch (const std::exception& ex) {
    std::cerr << "stainkit: failed to construct pipeline: " << ex.what()
              << std::endl;
    return nullptr;
  }
}

std::string Pipeline::DeviceName() const {
  return ctx_ ? ctx_->DeviceName() : std::string{"<no CUDA context>"};
}

PipelineResult Pipeline::Run(const Image& input, const PipelineParams& params,
                             const StainTarget& target) {
  return RunWithCpuBaseline(input, params, target);
}

PipelineResult Pipeline::RunWithCpuBaseline(const Image& input,
                                            const PipelineParams& params,
                                            const StainTarget& target) {
  if (!ctx_) {
    throw std::runtime_error("Pipeline::Run: no CUDA context");
  }
  if (input.layout != PixelLayout::kRgb) {
    throw std::invalid_argument("Pipeline::Run: only 3-channel RGB supported");
  }
  const std::size_t w       = input.width;
  const std::size_t h       = input.height;
  const std::size_t npix    = w * h;
  const std::size_t rgb_sz  = npix * 3 * sizeof(float);
  const std::size_t od_sz   = npix * 2 * sizeof(float);
  const std::size_t lum_sz  = npix * sizeof(float);
  const std::size_t mask_sz = npix * sizeof(std::uint8_t);

  PipelineResult result;
  result.timing.image_id = input.empty() ? "<empty>" : "<gpu-run>";
  result.timing.width    = w;
  result.timing.height   = h;

  // -- CPU baseline (always measured, even if we don't need it) --
  double cpu_ms = 0.0;
  (void)CpuReferenceStainNormalise(input, params, target, &cpu_ms);
  result.timing.cpu_baseline_ms = cpu_ms;

  // -- Host staging (use pinned memory when requested) --
  std::vector<float> host_rgb(npix * 3);
  for (std::size_t y = 0; y < h; ++y) {
    const byte* row_in = input.pixels.data() + y * input.stride;
    float*      row_out = host_rgb.data() + y * w * 3;
    for (std::size_t x = 0; x < w; ++x) {
      row_out[3 * x + 0] = row_in[3 * x + 0] * (1.0f / 255.0f);
      row_out[3 * x + 1] = row_in[3 * x + 1] * (1.0f / 255.0f);
      row_out[3 * x + 2] = row_in[3 * x + 2] * (1.0f / 255.0f);
    }
  }
  result.timing.load_ms = 0.0;  // the caller is responsible for I/O timing.

  // -- Allocate device buffers --
  DeviceBuffer d_rgb_in(rgb_sz);
  DeviceBuffer d_rgb_out(rgb_sz);
  DeviceBuffer d_stain_od(od_sz);
  DeviceBuffer d_lum(lum_sz);
  DeviceBuffer d_mask(mask_sz);

  cudaStream_t stream = ctx_->Stream(0);

  {
    ScopedEvent ev(stream);
    cudaMemcpyAsync(d_rgb_in.ptr, host_rgb.data(), rgb_sz,
                    cudaMemcpyHostToDevice, stream);
    result.timing.copy_h2d_ms = ev.elapsed_ms;
  }

  // -- Deconvolve (RGB -> OD) --
  {
    ScopedEvent ev(stream);
    kernels::ColorDeconvolveRgb(
        static_cast<const float*>(d_rgb_in.ptr), w, h, target.matrix,
        static_cast<float*>(d_stain_od.ptr), 2, 1, stream);
    result.timing.deconvolve_ms = ev.elapsed_ms;
  }

  // -- Macenko normalise (estimated basis + target reconstruction) --
  {
    ScopedEvent ev(stream);
    StainMatrix est = target.matrix;
    kernels::NormaliseStainFull(static_cast<const float*>(d_rgb_in.ptr), w,
                                h, params, target, est,
                                static_cast<float*>(d_rgb_out.ptr), stream);
    result.estimated_matrix = est;
    result.timing.normalise_ms = ev.elapsed_ms;
  }

  // -- Tissue mask --
  if (params.compute_tissue_mask) {
    ScopedEvent ev(stream);
    kernels::RgbToLuminance(static_cast<const float*>(d_rgb_out.ptr), w, h,
                            static_cast<float*>(d_lum.ptr), stream);
    const float threshold = kernels::OtsuThresholdDevice(
        static_cast<const float*>(d_lum.ptr), w, h, stream);
    kernels::ThresholdToMask(static_cast<const float*>(d_lum.ptr), w, h,
                             threshold,
                             static_cast<std::size_t>(params.otsu_smoothing_radius),
                             static_cast<std::uint8_t*>(d_mask.ptr), stream);
    result.timing.mask_ms = ev.elapsed_ms;
  }

  // -- Copy back --
  result.normalised  = MakeImage(w, h, PixelLayout::kRgb);
  result.tissue_mask = MakeImage(w, h, PixelLayout::kRgb);
  {
    ScopedEvent ev(stream);
    std::vector<float> back_rgb(npix * 3);
    cudaMemcpyAsync(back_rgb.data(), d_rgb_out.ptr, rgb_sz,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (std::size_t y = 0; y < h; ++y) {
      byte* row = result.normalised.pixels.data() + y * result.normalised.stride;
      for (std::size_t x = 0; x < w; ++x) {
        const float r = std::clamp(back_rgb[3 * (y * w + x) + 0], 0.0f, 1.0f);
        const float g = std::clamp(back_rgb[3 * (y * w + x) + 1], 0.0f, 1.0f);
        const float b = std::clamp(back_rgb[3 * (y * w + x) + 2], 0.0f, 1.0f);
        row[3 * x + 0] = static_cast<byte>(r * 255.0f + 0.5f);
        row[3 * x + 1] = static_cast<byte>(g * 255.0f + 0.5f);
        row[3 * x + 2] = static_cast<byte>(b * 255.0f + 0.5f);
      }
    }
    result.timing.copy_d2h_ms = ev.elapsed_ms;
  }
  {
    ScopedEvent ev(stream);
    std::vector<std::uint8_t> back_mask(npix);
    cudaMemcpyAsync(back_mask.data(), d_mask.ptr, mask_sz,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (std::size_t y = 0; y < h; ++y) {
      byte* row = result.tissue_mask.pixels.data() + y * result.tissue_mask.stride;
      for (std::size_t x = 0; x < w; ++x) {
        const byte v  = back_mask[y * w + x];
        row[3 * x + 0] = v;
        row[3 * x + 1] = v;
        row[3 * x + 2] = v;
      }
    }
  }

  // Totals
  result.timing.total_ms = result.timing.copy_h2d_ms +
                           result.timing.deconvolve_ms +
                           result.timing.normalise_ms +
                           result.timing.mask_ms + result.timing.copy_d2h_ms;
  return result;
}

std::vector<PipelineResult> Pipeline::RunBatch(
    const std::vector<Image>& inputs, const PipelineParams& params,
    const StainTarget& target) {
  std::vector<PipelineResult> out;
  out.reserve(inputs.size());
  // For now we run sequentially. A multi-stream implementation is a
  // straightforward extension (see TODO at the end of this function).
  for (const auto& img : inputs) {
    out.push_back(RunWithCpuBaseline(img, params, target));
  }
  // TODO(perf): when num_streams > 1, slice `inputs` into chunks and
  // pipeline them across `ctx_->Stream(i)` with stage events for
  // inter-stream synchronisation.
  return out;
}

}  // namespace stainkit
