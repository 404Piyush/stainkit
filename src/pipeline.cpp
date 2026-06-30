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
    std::fprintf(stderr, "[CudaContext] enter\n"); std::fflush(stderr);
    int device_count = 0;
    cudaError_t err    = cudaGetDeviceCount(&device_count);
    std::fprintf(stderr, "[CudaContext] cudaGetDeviceCount -> %d (%s), count=%d\n",
                 (int)err, cudaGetErrorString(err), device_count);
    std::fflush(stderr);
    if (err != cudaSuccess || device_count == 0) {
      throw std::runtime_error(
          std::string("CudaContext: no CUDA devices available "
                      "(cudaGetDeviceCount returned ") +
          std::to_string(static_cast<int>(err)) +
          " with " + std::to_string(device_count) + " devices)");
    }
    cudaError_t set_err = cudaSetDevice(0);
    std::fprintf(stderr, "[CudaContext] cudaSetDevice -> %d (%s)\n",
                 (int)set_err, cudaGetErrorString(set_err));
    std::fflush(stderr);
    if (set_err != cudaSuccess) {
      throw std::runtime_error(
          std::string("CudaContext: cudaSetDevice(0) failed: ") +
          cudaGetErrorString(set_err));
    }
    cudaDeviceProp prop{};
    std::memset(&prop, 0, sizeof(prop));
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
    std::fprintf(stderr, "[CudaContext] cudaGetDeviceProperties -> %d (%s)\n",
                 (int)prop_err, cudaGetErrorString(prop_err));
    std::fflush(stderr);
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
    std::fprintf(stderr, "[~ScopedEvent] entry stream=%p\n", (void*)stream); std::fflush(stderr);
    cudaError_t e1 = cudaEventRecord(stop, stream);
    std::fprintf(stderr, "[~ScopedEvent] EventRecord -> %d (%s)\n", (int)e1, cudaGetErrorString(e1)); std::fflush(stderr);
    cudaError_t e2 = cudaEventSynchronize(stop);
    std::fprintf(stderr, "[~ScopedEvent] EventSync -> %d (%s)\n", (int)e2, cudaGetErrorString(e2)); std::fflush(stderr);
    cudaError_t e3 = cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::fprintf(stderr, "[~ScopedEvent] ElapsedTime -> %d (%s), ms=%f\n",
                 (int)e3, cudaGetErrorString(e3), elapsed_ms); std::fflush(stderr);
    cudaError_t e4 = cudaEventDestroy(start);
    std::fprintf(stderr, "[~ScopedEvent] Destroy start -> %d (%s)\n",
                 (int)e4, cudaGetErrorString(e4)); std::fflush(stderr);
    cudaError_t e5 = cudaEventDestroy(stop);
    std::fprintf(stderr, "[~ScopedEvent] Destroy stop -> %d (%s)\n",
                 (int)e5, cudaGetErrorString(e5)); std::fflush(stderr);
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
  std::fprintf(stderr, "[Make] enter\n"); std::fflush(stderr);
  try {
    std::fprintf(stderr, "[Make] about to allocate Pipeline\n"); std::fflush(stderr);
    auto p = std::unique_ptr<Pipeline>(new Pipeline());
    std::fprintf(stderr, "[Make] Pipeline allocated ok\n"); std::fflush(stderr);
    return p;
  } catch (const std::exception& ex) {
    std::cerr << "stainkit: failed to construct pipeline: " << ex.what()
              << std::endl;
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// SIGSEGV-recovering Make(): catches a SIGSEGV raised inside CUDA init
// (for example an ABI mismatch between the build-time and host CUDA
// runtime). Without this, an ABI mismatch terminates the whole process
// and the CLI cannot fall back to the CPU reference path.
// ---------------------------------------------------------------------------
namespace {

struct SegvRecovery {
  sigjmp_buf jmp{};
  volatile std::sig_atomic_t armed = 0;

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
    // Re-raise with default disposition so the user still sees a crash
    // if the SIGSEGV happens outside MakeOrFallback's scope.
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
  std::fprintf(stderr, "[MakeOrFallback] enter\n"); std::fflush(stderr);
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
  std::fprintf(stderr, "[MakeOrFallback] signal handlers installed\n"); std::fflush(stderr);
  auto p = Make();
  std::fprintf(stderr, "[MakeOrFallback] Make() returned\n"); std::fflush(stderr);
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
  std::fprintf(stderr, "[RunWithCpuBaseline] enter w=%zu h=%zu npix=%zu\n",
               input.width, input.height, input.width * input.height);
  std::fflush(stderr);
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
  std::fprintf(stderr, "[RunWithCpuBaseline] rgb_sz=%zu npix=%zu\n", rgb_sz, npix);
  std::fflush(stderr);
  const std::size_t od_sz   = npix * 2 * sizeof(float);
  const std::size_t lum_sz  = npix * sizeof(float);
  const std::size_t mask_sz = npix * sizeof(std::uint8_t);
  std::fprintf(stderr, "[RunWithCpuBaseline] before cpu baseline\n"); std::fflush(stderr);

  PipelineResult result;
  result.timing.image_id = input.empty() ? "<empty>" : "<gpu-run>";
  result.timing.width    = w;
  result.timing.height   = h;

  // -- CPU baseline (always measured, even if we don't need it) --
  double cpu_ms = 0.0;
  std::fprintf(stderr, "[RunWithCpuBaseline] calling cpu baseline...\n"); std::fflush(stderr);
  (void)CpuReferenceStainNormalise(input, params, target, &cpu_ms);
  result.timing.cpu_baseline_ms = cpu_ms;
  std::fprintf(stderr, "[RunWithCpuBaseline] cpu baseline done, ms=%.3f\n", cpu_ms); std::fflush(stderr);

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
  std::fprintf(stderr, "[RunWithCpuBaseline] host staging done\n"); std::fflush(stderr);

  // -- Allocate device buffers --
  std::fprintf(stderr, "[RunWithCpuBaseline] allocating device buffers...\n"); std::fflush(stderr);
  DeviceBuffer d_rgb_in(rgb_sz);
  std::fprintf(stderr, "[RunWithCpuBaseline] d_rgb_in ok\n"); std::fflush(stderr);
  DeviceBuffer d_rgb_out(rgb_sz);
  std::fprintf(stderr, "[RunWithCpuBaseline] d_rgb_out ok\n"); std::fflush(stderr);
  DeviceBuffer d_stain_od(od_sz);
  std::fprintf(stderr, "[RunWithCpuBaseline] d_stain_od ok\n"); std::fflush(stderr);
  DeviceBuffer d_lum(lum_sz);
  std::fprintf(stderr, "[RunWithCpuBaseline] d_lum ok\n"); std::fflush(stderr);
  DeviceBuffer d_mask(mask_sz);
  std::fprintf(stderr, "[RunWithCpuBaseline] d_mask ok\n"); std::fflush(stderr);

  cudaStream_t stream = ctx_->Stream(0);
  std::fprintf(stderr, "[RunWithCpuBaseline] got stream=%p\n", (void*)stream); std::fflush(stderr);

  {
    std::fprintf(stderr, "[RunWithCpuBaseline] creating ScopedEvent...\n"); std::fflush(stderr);
    ScopedEvent ev(stream);
    std::fprintf(stderr, "[RunWithCpuBaseline] ScopedEvent ok\n"); std::fflush(stderr);
    cudaError_t me = cudaMemcpyAsync(d_rgb_in.ptr, host_rgb.data(), rgb_sz,
                                     cudaMemcpyHostToDevice, stream);
    std::fprintf(stderr, "[RunWithCpuBaseline] cudaMemcpyAsync -> %d (%s)\n",
                 (int)me, cudaGetErrorString(me)); std::fflush(stderr);
    result.timing.copy_h2d_ms = ev.elapsed_ms;
    std::fprintf(stderr, "[RunWithCpuBaseline] copy_h2d_ms = %f\n",
                 ev.elapsed_ms); std::fflush(stderr);
  }

  // -- Deconvolve (RGB -> OD) --
  std::fprintf(stderr, "[RunWithCpuBaseline] entering deconvolve\n"); std::fflush(stderr);
  {
    ScopedEvent ev(stream);
    kernels::ColorDeconvolveRgb(
        static_cast<const float*>(d_rgb_in.ptr), w, h, target.matrix,
        static_cast<float*>(d_stain_od.ptr), 2, 1, stream);
    result.timing.deconvolve_ms = ev.elapsed_ms;
  }
  std::fprintf(stderr, "[RunWithCpuBaseline] deconvolve done\n"); std::fflush(stderr);

  // -- Macenko normalise (estimated basis + target reconstruction) --
  std::fprintf(stderr, "[RunWithCpuBaseline] entering normalise\n"); std::fflush(stderr);
  {
    std::fprintf(stderr, "[RunWithCpuBaseline] normalise: creating ScopedEvent\n"); std::fflush(stderr);
    ScopedEvent ev(stream);
    std::fprintf(stderr, "[RunWithCpuBaseline] normalise: ScopedEvent ok\n"); std::fflush(stderr);
    StainMatrix est = target.matrix;
    std::fprintf(stderr, "[RunWithCpuBaseline] normalise: calling NormaliseStainFull\n"); std::fflush(stderr);
    kernels::NormaliseStainFull(static_cast<const float*>(d_rgb_in.ptr), w,
                                h, params, target, est,
                                static_cast<float*>(d_rgb_out.ptr), stream);
    std::fprintf(stderr, "[RunWithCpuBaseline] normalise: NormaliseStainFull ok\n"); std::fflush(stderr);
    result.estimated_matrix = est;
    result.timing.normalise_ms = ev.elapsed_ms;
  }
  std::fprintf(stderr, "[RunWithCpuBaseline] normalise done\n"); std::fflush(stderr);

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
