// stainkit/src/pipeline.cpp
//
// The top-level orchestration. Owns the CUDA context, manages device
// memory, launches kernels in a multi-stream fashion, and exposes the
// `Pipeline` class to the CLI / Python bindings.

#include "stainkit/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
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
    cudaError_t err  = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      throw std::runtime_error(
          "CudaContext: no CUDA devices available");
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

    for (int i = 0; i < kMaxStreams; ++i) {
      streams_[i] = nullptr;
      cudaError_t stream_err = cudaStreamCreate(&streams_[i]);
      if (stream_err != cudaSuccess) {
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

// RAII wrapper for device memory.
struct DeviceBuffer {
  void*  ptr  = nullptr;
  size_t size = 0;

  explicit DeviceBuffer(size_t bytes) : size(bytes) {
    if (bytes == 0) return;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e != cudaSuccess) {
      throw std::runtime_error(
          std::string("DeviceBuffer: cudaMalloc failed: ") +
          cudaGetErrorString(e));
    }
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Stage timing helper. Records a `start` event on construction and a
// `stop` event on `MarkStop()` (or destructor). `elapsed_ms` is read
// AFTER `MarkStop()` returns.
struct ScopedEvent {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaStream_t stream{};
  float       elapsed_ms = 0.0f;
  bool        stopped    = false;

  explicit ScopedEvent(cudaStream_t s) : stream(s) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
  }
  void MarkStop() {
    if (stopped) return;
    stopped = true;
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
  }
  ~ScopedEvent() { MarkStop(); }
  ScopedEvent(const ScopedEvent&)            = delete;
  ScopedEvent& operator=(const ScopedEvent&) = delete;
};

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

namespace {

// SIGSEGV-recovering wrapper around Pipeline::Make(). Used by the CLI
// when --no-pipeline-fallback is NOT set; recovers from a CUDA init
// crash (typically an ABI mismatch between the build-time and host CUDA
// runtime) and returns nullptr so the CLI can fall back to the CPU
// reference implementation.
struct SegvRecovery {
  sigjmp_buf jmp{};
  std::sig_atomic_t armed = 0;

  void Arm() {
    armed = 1;
    std::signal(SIGSEGV, &SegvRecovery::Handler);
    std::signal(SIGBUS,  &SegvRecovery::Handler);
  }

  void Disarm() {
    armed = 0;
    std::signal(SIGSEGV, SIG_DFL);
    std::signal(SIGBUS,  SIG_DFL);
  }

  static void Handler(int sig) {
    auto* self = current();
    if (self != nullptr && self->armed != 0) {
      self->Disarm();
      siglongjmp(self->jmp, 1);
    }
    std::signal(sig, SIG_DFL);
    raise(sig);
  }

  static SegvRecovery*& current() {
    static thread_local SegvRecovery* ptr = nullptr;
    return ptr;
  }
};

}  // namespace

std::unique_ptr<Pipeline> Pipeline::MakeOrFallback() {
  SegvRecovery rec;
  rec.current() = &rec;
  if (sigsetjmp(rec.jmp, 1) != 0) {
    std::cerr << "stainkit: caught SIGSEGV/SIGBUS inside CUDA init - "
                 "falling back to CPU reference implementation. "
                 "This usually means the binary was built against a CUDA "
                 "runtime that is incompatible with the host driver."
              << std::endl;
    return nullptr;
  }
  rec.Arm();
  auto p = Make();
  rec.Disarm();
  return p;
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
  result.timing.image_id = "<gpu-run>";
  result.timing.width    = w;
  result.timing.height   = h;

  // CPU baseline (always measured so the benchmark CSV has a column
  // for it). Even when the user does not request --benchmark, the
  // host-side implementation doubles as a validation oracle.
  double cpu_ms = 0.0;
  (void)CpuReferenceStainNormalise(input, params, target, &cpu_ms);
  result.timing.cpu_baseline_ms = cpu_ms;

  // Host staging. We use plain malloc (not pinned) here because the
  // 256x256 patches we run in CI fit comfortably in pageable memory;
  // pinned allocations are more expensive to set up than they save in
  // transfer time at this size. The pipeline is structured so that
  // switching to pinned is a single-line change.
  std::vector<float> host_rgb(npix * 3);
  for (std::size_t y = 0; y < h; ++y) {
    const byte* row_in  = input.pixels.data() + y * input.stride;
    float*      row_out = host_rgb.data() + y * w * 3;
    for (std::size_t x = 0; x < w; ++x) {
      row_out[3 * x + 0] = row_in[3 * x + 0] * (1.0f / 255.0f);
      row_out[3 * x + 1] = row_in[3 * x + 1] * (1.0f / 255.0f);
      row_out[3 * x + 2] = row_in[3 * x + 2] * (1.0f / 255.0f);
    }
  }
  result.timing.load_ms = 0.0;  // the caller measures I/O.

  DeviceBuffer d_rgb_in(rgb_sz);
  DeviceBuffer d_rgb_out(rgb_sz);
  DeviceBuffer d_stain_od(od_sz);
  DeviceBuffer d_lum(lum_sz);
  DeviceBuffer d_mask(mask_sz);

  cudaStream_t stream = ctx_->Stream(0);

  // Stage timers. Each ScopedEvent is declared at function scope so the
  // destructor runs *after* the corresponding work; `elapsed_ms` is read
  // after `MarkStop()` returns.
  ScopedEvent copy_h2d_ev(stream);
  ScopedEvent deconvolve_ev(stream);
  ScopedEvent normalise_ev(stream);
  ScopedEvent mask_ev(stream);
  ScopedEvent copy_d2h_ev(stream);

  // -- H2D --
  cudaMemcpyAsync(d_rgb_in.ptr, host_rgb.data(), rgb_sz,
                  cudaMemcpyHostToDevice, stream);
  copy_h2d_ev.MarkStop();
  result.timing.copy_h2d_ms = copy_h2d_ev.elapsed_ms;

  // -- Deconvolve (RGB -> OD) --
  kernels::ColorDeconvolveRgb(
      static_cast<const float*>(d_rgb_in.ptr), w, h,
      target.matrix.values.data(),
      static_cast<float*>(d_stain_od.ptr), 2, 1, stream);
  deconvolve_ev.MarkStop();
  result.timing.deconvolve_ms = deconvolve_ev.elapsed_ms;

  // -- Macenko normalise (estimated basis + target reconstruction) --
  StainMatrix est = target.matrix;
  std::array<float, 6> h_matrix_inv{};
  for (int i = 0; i < 6; ++i) h_matrix_inv[i] = target.matrix.values[i];
  std::array<float, 3> h_conc = {
      target.target_he_concentrations[0],
      target.target_he_concentrations[1],
      target.target_he_concentrations[2],
  };
  kernels::NormaliseStainFull(static_cast<const float*>(d_rgb_in.ptr), w,
                              h, params,
                              h_matrix_inv.data(),
                              h_conc.data(),
                              est,
                              static_cast<float*>(d_rgb_out.ptr), stream);
  normalise_ev.MarkStop();
  result.estimated_matrix = est;
  result.timing.normalise_ms = normalise_ev.elapsed_ms;

  // -- Tissue mask --
  if (params.compute_tissue_mask) {
    kernels::RgbToLuminance(static_cast<const float*>(d_rgb_out.ptr), w, h,
                            static_cast<float*>(d_lum.ptr), stream);
    const float threshold = kernels::OtsuThresholdDevice(
        static_cast<const float*>(d_lum.ptr), w, h, stream);
    kernels::ThresholdToMask(static_cast<const float*>(d_lum.ptr), w, h,
                             threshold,
                             static_cast<std::size_t>(params.otsu_smoothing_radius),
                             static_cast<std::uint8_t*>(d_mask.ptr), stream);
    mask_ev.MarkStop();
    result.timing.mask_ms = mask_ev.elapsed_ms;
  }

  // -- Copy back --
  result.normalised  = MakeImage(w, h, PixelLayout::kRgb);
  result.tissue_mask = MakeImage(w, h, PixelLayout::kRgb);
  {
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
  }
  copy_d2h_ev.MarkStop();
  result.timing.copy_d2h_ms = copy_d2h_ev.elapsed_ms;
  {
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
                           result.timing.mask_ms +
                           result.timing.copy_d2h_ms;
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