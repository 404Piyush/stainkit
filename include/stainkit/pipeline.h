// stainkit/include/stainkit/pipeline.h
//
// The pipeline orchestrates loading, H<->D transfers, kernel execution and
// write-back. It is the only object the CLI / Python bindings need to
// instantiate.

#ifndef STK_INCLUDE_STAINKIT_PIPELINE_H_
#define STK_INCLUDE_STAINKIT_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include "stainkit/types.h"

namespace stainkit {

class CudaContext;  // forward decl — defined in src/pipeline.cpp

// ---------------------------------------------------------------------------
// Output bundle — every call to `Pipeline::Run` returns one of these.
// ---------------------------------------------------------------------------
struct PipelineResult {
  Image     normalised;       // stain-normalised RGB
  Image     tissue_mask;      // 1-channel uint8 (0..255) where 255 = tissue
  Image     stain_od;         // 2-channel float, packed as half-precision
                              // (only populated if requested)
  StainMatrix estimated_matrix;
  BenchmarkRecord timing;
};

// ---------------------------------------------------------------------------
// `Pipeline` — top-level entry point.
// ---------------------------------------------------------------------------
// Lifetime: the pipeline owns its CUDA context. The default constructor
// will throw at runtime if no CUDA device is available, but the static
// factory `Make()` returns `nullptr` instead of throwing so the CLI can
// gracefully fall back to a CPU implementation.
class Pipeline {
 public:
  Pipeline();
  ~Pipeline();

  Pipeline(const Pipeline&)            = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  static std::unique_ptr<Pipeline> Make();
  static bool IsCudaAvailable() noexcept;

  // Per-image run. The image data is fully copied inside this call; the
  // caller may free `input` immediately upon return.
  PipelineResult Run(const Image& input, const PipelineParams& params,
                     const StainTarget& target);

  // Per-image run that also takes a CPU baseline for benchmarking.
  PipelineResult RunWithCpuBaseline(const Image& input,
                                    const PipelineParams& params,
                                    const StainTarget& target);

  // Batch helper: iterates over `inputs`, optionally reusing pinned
  // memory, and returns the per-image results. If `params.num_streams > 1`
  // the work is overlapped across streams.
  std::vector<PipelineResult> RunBatch(
      const std::vector<Image>& inputs, const PipelineParams& params,
      const StainTarget& target);

  // Human-readable device name (for logging).
  std::string DeviceName() const;

 private:
  std::unique_ptr<CudaContext> ctx_;
};

}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_PIPELINE_H_
