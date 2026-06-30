# Architecture

stainkit is split into three layered modules:

```
+-------------------------------------------------------+
|                      CLI / Python                     |
|           (src/main.cpp, src/bindings/)               |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|                       Pipeline                        |
|             (src/pipeline.cpp, include/stainkit/)     |
|                                                       |
|  - Manages CUDA context, streams, device buffers.      |
|  - Implements the multi-stream overlap strategy.      |
+---------------------------+---------------------------+
                            |
                            v
+---------------------------+---------------------------+
|  stainkit_cuda (CUDA)    |  stainkit_core (host)      |
|  src/kernels/*.cu         |  src/io.cpp                |
|  include/stainkit/        |  src/cpu_reference.cpp     |
|  kernels/*.h              |  include/stainkit/io.h     |
+---------------------------+---------------------------+
```

## Data flow

For a single image:

```
       Host: stb_image reads the PNG/JPG/BMP/TGA into an `Image`
                       |
                       v   (H2D, pinned memory)
       Device: float[3 * H * W]
                       |
                       v   (DeconvolveKernel, RgbToOd inline)
       Device: float[2 * H * W]   <-- (H, E) OD pair
                       |
                       v   (ComputeStainPlaneAngles)
       Device: float[H * W] angles, magnitudes
                       |
                       v   (host)
       Host: percentile pick -> StainMatrix (estimated)
                       |
                       v   (ReconstructRgbFromStain, target matrix + conc.)
       Device: float[3 * H * W]   <-- normalised RGB
                       |
                       v   (RgbToLuminance, Otsu, Binarise, Morph)
       Device: uint8_t[H * W]   <-- tissue mask
                       |
                       v   (D2H)
       Host: stb_image_write the normalised + mask + 3-panel
```

The `Pipeline::RunBatch` method slices a vector of images into
chunks of `params.num_streams` and submits them to the device with
one stream per chunk.

## Memory model

* **Pinned host memory.** The host staging buffer is allocated with
  `cudaHostAllocDefault`. Transfers are asynchronous on the stream
  they are submitted to.
* **Device buffers.** A single `DeviceBuffer` is allocated for each
  image; we re-use it across pipeline stages. The scratch buffer for
  the morphology pass is allocated and freed inside the kernel
  dispatch.
* **Constant memory.** The 9-float inverse stain matrix is uploaded
  once per pipeline invocation via `cudaMemcpyToSymbolAsync`.

## Stream semantics

A single `Pipeline` owns `kMaxStreams = 8` CUDA streams. Streams are
created in the constructor and destroyed in the destructor; the
helper `Pipeline::Run` uses stream 0 only. `RunBatch` is the entry
point for the multi-stream path.

## Error handling

All CUDA errors are checked with `cudaError_t` and converted to
`std::runtime_error` on the host. The pipeline constructor itself
throws when no CUDA device is detected; the static
`Pipeline::IsCudaAvailable()` lets the CLI and Python bindings
perform a graceful CPU fallback.

## Threading

The CPU reference implementation is single-threaded by design; it
exists to be easy to read and to be the ground truth for the GPU
output. If you need a multi-threaded CPU implementation, the
relevant code in `src/cpu_reference.cpp` can be parallelised with
`#pragma omp parallel for` (OpenMP is detected at configure time).
