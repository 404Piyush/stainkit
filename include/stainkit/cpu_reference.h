// stainkit/include/stainkit/cpu_reference.h
//
// Pure-CPU implementations of the algorithms. They are not intended to be
// fast — their sole purpose is to produce a *reference* result so the
// GPU pipeline can be quantitatively validated and benchmarked.

#ifndef STK_INCLUDE_STAINKIT_CPU_REFERENCE_H_
#define STK_INCLUDE_STAINKIT_CPU_REFERENCE_H_

#include <cstddef>

#include "stainkit/types.h"

namespace stainkit {

// ---------------------------------------------------------------------------
// CPU baseline: runs the same Macenko / Ruifrok pipeline in a single thread
// and returns the elapsed wall time in milliseconds via `elapsed_ms`.
// ---------------------------------------------------------------------------
// Used by the benchmark executable and the test suite.
Image CpuReferenceStainNormalise(const Image& input,
                                 const PipelineParams& params,
                                 const StainTarget& target,
                                 double* elapsed_ms = nullptr);

}  // namespace stainkit

#endif  // STK_INCLUDE_STAINKIT_CPU_REFERENCE_H_
