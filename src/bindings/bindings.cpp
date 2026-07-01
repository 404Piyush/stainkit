// stainkit/src/bindings/bindings.cpp
//
// Python bindings — exposes a tiny subset of the C++ API as a pybind11
// module named `gpustain`.
//
//   import gpustain
//   img = gpustain.read_image("patch.png")
//   out = gpustain.run(img, target="default")
//   gpustain.write_image(out.normalised, "out.png")
//
// All GPU work goes through the same `Pipeline` class the CLI uses, so
// performance numbers are directly comparable.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "stainkit/io.h"
#include "stainkit/pipeline.h"
#include "stainkit/types.h"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

// Convert a (H, W, 3) uint8 numpy array into a stainkit::Image. We do not
// copy when the array is contiguous in C order.
stainkit::Image FromNumpy(
    py::array_t<std::uint8_t, py::array::c_style | py::array::c_contiguous>
        arr) {
  if (arr.ndim() != 3 || arr.shape(2) != 3) {
    throw std::invalid_argument("expected (H, W, 3) uint8 array");
  }
  stainkit::Image img;
  img.width = static_cast<std::size_t>(arr.shape(1));
  img.height = static_cast<std::size_t>(arr.shape(0));
  img.layout = stainkit::PixelLayout::kRgb;
  img.stride = stainkit::ComputeStride(img.width, 3);
  img.pixels.assign(img.stride * img.height, 0);
  for (std::size_t y = 0; y < img.height; ++y) {
    const auto* src = arr.data() + y * img.width * 3;
    auto* dst = img.pixels.data() + y * img.stride;
    std::memcpy(dst, src, img.width * 3);
  }
  return img;
}

py::array_t<std::uint8_t> ToNumpy(const stainkit::Image& img) {
  if (img.layout != stainkit::PixelLayout::kRgb) {
    throw std::invalid_argument("only RGB images are supported");
  }
  py::array_t<std::uint8_t> out({img.height, img.width, std::size_t{3}});
  auto raw = out.mutable_unchecked<3>();
  for (std::size_t y = 0; y < img.height; ++y) {
    const auto* row = img.pixels.data() + y * img.stride;
    for (std::size_t x = 0; x < img.width; ++x) {
      raw(y, x, 0) = row[3 * x + 0];
      raw(y, x, 1) = row[3 * x + 1];
      raw(y, x, 2) = row[3 * x + 2];
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(gpustain, m) {
  m.doc() = "gpustain — GPU-accelerated H&E stain normalization";

  m.def(
      "read_image",
      [](const std::string& path) {
        return FromNumpy(stainkit::ReadImage(fs::path{path}, 3));
      },
      "Read an RGB image from disk into a (H, W, 3) uint8 array.");

  m.def(
      "write_image",
      [](py::array_t<std::uint8_t> arr, const std::string& path) {
        auto img = FromNumpy(arr);
        stainkit::WriteImage(img, fs::path{path});
      },
      "Write a (H, W, 3) uint8 image to disk.");

  m.def("is_cuda_available", &stainkit::Pipeline::IsCudaAvailable,
        "Return True iff a CUDA device is detected.");

  py::class_<stainkit::PipelineResult>(m, "Result")
      .def_readonly("normalised", &stainkit::PipelineResult::normalised)
      .def_readonly("tissue_mask", &stainkit::PipelineResult::tissue_mask)
      .def_readonly("timing", &stainkit::PipelineResult::timing)
      .def("normalised_array",
           [](const stainkit::PipelineResult& r) {
             return ToNumpy(r.normalised);
           })
      .def("mask_array", [](const stainkit::PipelineResult& r) {
        return ToNumpy(r.tissue_mask);
      });

  m.def(
      "run",
      [](py::array_t<std::uint8_t> arr, const std::string& target_name) {
        auto img = FromNumpy(arr);
        auto pl = stainkit::Pipeline::Make();
        if (!pl)
          throw std::runtime_error("CUDA not available");
        stainkit::StainTarget target;
        target.name = target_name;
        if (target_name == "he-royal") {
          target.matrix.values = {0.5826f, 0.7660f, 0.2723f,
                                  0.2053f, 0.8520f, 0.4830f};
        } else if (target_name == "he-icm") {
          target.matrix.values = {0.6500f, 0.7040f, 0.2860f,
                                  0.0700f, 0.9900f, 0.1200f};
        }
        return pl->RunWithCpuBaseline(img, stainkit::PipelineParams{}, target);
      },
      py::arg("image"), py::arg("target") = std::string("default"),
      "Run the full pipeline on a single image and return the normalised "
      "RGB image and the tissue mask.");

  py::class_<stainkit::BenchmarkRecord>(m, "BenchmarkRecord")
      .def_readonly("image_id", &stainkit::BenchmarkRecord::image_id)
      .def_readonly("width", &stainkit::BenchmarkRecord::width)
      .def_readonly("height", &stainkit::BenchmarkRecord::height)
      .def_readonly("load_ms", &stainkit::BenchmarkRecord::load_ms)
      .def_readonly("copy_h2d_ms", &stainkit::BenchmarkRecord::copy_h2d_ms)
      .def_readonly("deconvolve_ms", &stainkit::BenchmarkRecord::deconvolve_ms)
      .def_readonly("normalise_ms", &stainkit::BenchmarkRecord::normalise_ms)
      .def_readonly("mask_ms", &stainkit::BenchmarkRecord::mask_ms)
      .def_readonly("copy_d2h_ms", &stainkit::BenchmarkRecord::copy_d2h_ms)
      .def_readonly("total_ms", &stainkit::BenchmarkRecord::total_ms)
      .def_readonly("cpu_baseline_ms",
                    &stainkit::BenchmarkRecord::cpu_baseline_ms);
}
