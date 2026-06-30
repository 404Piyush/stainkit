# API Reference

The public C++ API lives in `include/stainkit/`. The Python module
`gpustain` exposes a strict subset of it.

## `Image` (`include/stainkit/types.h`)

```cpp
struct Image {
  std::vector<byte> pixels;
  std::size_t       width;
  std::size_t       height;
  std::size_t       stride;     // bytes per row, padded to a 4-byte boundary
  PixelLayout       layout;     // kRgb (3) or kRgba (4)
};
```

Constructors:

* `Image MakeImage(width, height, layout)` — `src/types.cpp`. Allocates
  a zero-initialised `Image` whose stride is rounded up to the
  nearest 4-byte boundary.

Helpers:

* `std::size_t ComputeStride(width, channels)` — compute the padded
  row stride.
* `float3 ClampUnit(v)` — clamp a `float3` to `[0, 1]`.
* `float Luma(v)` — BT.709 luminance.

## `Pipeline` (`include/stainkit/pipeline.h`)

```cpp
class Pipeline {
 public:
  static std::unique_ptr<Pipeline> Make();      // returns nullptr if no CUDA
  static bool IsCudaAvailable() noexcept;

  PipelineResult Run(const Image& input, const PipelineParams& params,
                     const StainTarget& target);
  PipelineResult RunWithCpuBaseline(...);
  std::vector<PipelineResult> RunBatch(const std::vector<Image>&,
                                       const PipelineParams&,
                                       const StainTarget&);
  std::string DeviceName() const;
};
```

## `PipelineResult`

```cpp
struct PipelineResult {
  Image     normalised;
  Image     tissue_mask;
  Image     stain_od;          // currently always empty; reserved
  StainMatrix estimated_matrix;
  BenchmarkRecord timing;
};
```

## Kernels (`include/stainkit/kernels/`)

Each header declares a *host-side* launcher. The `.cu` translation
unit holds the device-side kernel itself. All launchers take a
`void* stream` argument that defaults to `nullptr` (= default
stream).

* `color_deconvolution.h` — `ColorDeconvolveRgb` returns elapsed ms.
* `stain_normalization.h` — `NormaliseStainFull` runs the full
  Macenko pipeline on a single image; stages are also exposed
  separately for fine-grained reuse.
* `tissue_mask.h` — `OtsuThresholdHost` / `OtsuThresholdDevice` and
  `ThresholdToMask`.
* `od_conversion.h` — RGB↔OD and luminance reductions.
* `morphology.h` — separable binary morphology with a disc SE.

## CLI

See `stainkit --help`.

## Python

```python
import gpustain
img = gpustain.read_image("patches/patient_001.png")
result = gpustain.run(img, target="default")
gpustain.write_image(result.normalised_array(), "out_normalised.png")
gpustain.write_image(result.mask_array(),      "out_mask.png")
```

`gpustain.is_cuda_available()` returns `True` if the bindings were
built with CUDA support.
