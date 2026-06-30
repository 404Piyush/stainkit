// stainkit/src/benchmarks/bench_pipeline.cpp
//
// Stand-alone benchmark binary. Loads a directory of images, runs the
// GPU pipeline (and the CPU reference for the same images) and prints
// a per-image summary to stdout. A CSV file can be emitted with --csv.
//
// Example:
//   ./stainkit-bench -i data/raw --csv data/benchmark/run-001.csv

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stainkit/cpu_reference.h"
#include "stainkit/io.h"
#include "stainkit/pipeline.h"
#include "stainkit/types.h"

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path    input_dir = "data/raw";
  fs::path    csv;
  int         num_images = -1;
  std::string target = "default";
};

bool Parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << s << "\n";
        std::exit(2);
      }
      return std::string{argv[++i]};
    };
    if (s == "-i" || s == "--input") a.input_dir = next();
    else if (s == "--csv") a.csv = next();
    else if (s == "--num-images") a.num_images = std::stoi(next());
    else if (s == "--target") a.target = next();
    else if (s == "-h" || s == "--help") {
      std::cout << "stainkit-bench -i <dir> [--csv <path>] [--num-images N]\n";
      std::exit(0);
    } else {
      std::cerr << "unknown arg: " << s << "\n";
      return false;
    }
  }
  return true;
}

stainkit::StainTarget MakeTarget(const std::string& name) {
  stainkit::StainTarget t;
  t.name = name;
  if (name == "he-royal") {
    t.matrix.values = {0.5826f, 0.7660f, 0.2723f, 0.2053f, 0.8520f, 0.4830f};
  } else if (name == "he-icm") {
    t.matrix.values = {0.6500f, 0.7040f, 0.2860f, 0.0700f, 0.9900f, 0.1200f};
  }
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!Parse(argc, argv, args)) return 2;

  std::vector<fs::path> inputs = stainkit::ListImagesIn(args.input_dir);
  if (args.num_images >= 0 &&
      static_cast<std::size_t>(args.num_images) < inputs.size()) {
    inputs.resize(static_cast<std::size_t>(args.num_images));
  }
  if (inputs.empty()) {
    std::cerr << "stainkit-bench: no images in " << args.input_dir << "\n";
    return 1;
  }

  auto pipeline = stainkit::Pipeline::Make();
  if (!pipeline) {
    std::cerr << "stainkit-bench: CUDA not available\n";
    return 1;
  }
  std::cout << "stainkit-bench: device = " << pipeline->DeviceName() << "\n";
  std::cout << "stainkit-bench: " << inputs.size() << " image(s)\n";

  const auto target = MakeTarget(args.target);
  stainkit::PipelineParams params;

  std::ofstream ofs;
  if (!args.csv.empty()) {
    fs::create_directories(args.csv.parent_path());
    ofs.open(args.csv);
    ofs << "image_id,width,height,total_ms,cpu_ms,speedup\n";
  }

  std::size_t idx = 0;
  double total_gpu = 0.0;
  double total_cpu = 0.0;
  for (const auto& p : inputs) {
    stainkit::Image img;
    try {
      img = stainkit::ReadImage(p, 3);
    } catch (const std::exception& ex) {
      std::cerr << "skip " << p << ": " << ex.what() << "\n";
      continue;
    }
    const auto rr = pipeline->RunWithCpuBaseline(img, params, target);
    const double gpu = rr.timing.total_ms;
    const double cpu = rr.timing.cpu_baseline_ms;
    total_gpu += gpu;
    total_cpu += cpu;
    const double speedup = (gpu > 0.0) ? cpu / gpu : 0.0;
    std::cout << "  [" << ++idx << "/" << inputs.size() << "] " << p.filename()
              << "  " << img.width << "x" << img.height
              << "  gpu=" << std::fixed << std::setprecision(2) << gpu
              << "ms cpu=" << cpu << "ms speedup=" << speedup << "x\n";
    if (ofs.is_open()) {
      ofs << p.stem().string() << ',' << img.width << ',' << img.height << ','
          << gpu << ',' << cpu << ',' << speedup << '\n';
    }
  }
  if (idx > 0) {
    std::cout << "stainkit-bench: average speedup = " << std::fixed
              << std::setprecision(2) << (total_cpu / total_gpu) << "x over "
              << idx << " image(s)\n";
  }
  return 0;
}
