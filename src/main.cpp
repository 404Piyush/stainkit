// stainkit/src/main.cpp
//
// CLI entry point.
//
// Usage:
//   stainkit --input  <dir>  --output <dir> [--filter macenko]
//                            [--target default|he-royal|he-icm]
//                            [--num-images N] [--benchmark]
//                            [--num-streams N] [--pinned] [--no-mask]
//                            [--csv <path>] [--help] [--version]
//
// Examples:
//   stainkit -i data/raw -o data/processed -f macenko
//   stainkit -i data/raw -o data/processed -f macenko --benchmark \
//            --csv data/benchmark/result.csv

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

constexpr const char* kVersion = "0.1.0";

struct CliArgs {
  fs::path input_dir;
  fs::path output_dir;
  std::string filter = "macenko";
  std::string target = "default";
  int num_images = -1;  // -1 == all
  int num_streams = 4;
  bool pinned_memory = true;
  bool compute_mask = true;
  bool use_estimated_basis_as_target = true;  // default: best visual
  bool benchmark = false;
  fs::path csv_path;
  bool show_help = false;
  bool show_version = false;
  bool cpu_only = false;  // fallback when CUDA is absent
};

void PrintHelp(std::ostream& os) {
  os << R"(stainkit - GPU-accelerated H&E stain normalization for digital pathology

Usage:
  stainkit [options]

Required:
  -i, --input  <dir>      Directory containing input images (PNG/JPG/BMP/TGA).
  -o, --output <dir>      Directory where processed images will be written.

Pipeline:
  -f, --filter   <name>   Pipeline filter. Only 'macenko' is implemented.
                          [default: macenko]
      --target   <name>   Target stain profile: default, he-royal, he-icm.
                          [default: default]

Execution:
      --num-images <N>    Process at most N images from the input dir.
                          -1 means "process every supported image".
                          [default: -1]
      --num-streams <N>   Number of CUDA streams to overlap I/O with compute.
                          [default: 4]
      --pinned            Use pinned (page-locked) host memory. [default]
      --no-pinned         Disable pinned memory.
      --no-mask           Skip the Otsu tissue-mask step.
      --cpu               Run the CPU reference implementation only (used when
                          no CUDA device is detected).

Output:
      --benchmark         Emit a per-image CPU vs GPU timing table.
      --csv <path>        Write benchmark results to a CSV file.

Misc:
  -h, --help              Print this help message and exit.
  -v, --version           Print the stainkit version and exit.

Examples:
  stainkit -i data/raw -o data/processed -f macenko
  stainkit -i data/raw -o data/processed -f macenko --benchmark --csv data/benchmark.csv
)";
}

bool ParseArgs(int argc, char** argv, CliArgs& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "stainkit: missing value for " << name << "\n";
        std::exit(2);
      }
      return std::string{argv[++i]};
    };
    if (a == "-h" || a == "--help") {
      out.show_help = true;
    } else if (a == "-v" || a == "--version") {
      out.show_version = true;
    } else if (a == "-i" || a == "--input") {
      out.input_dir = next(a.c_str());
    } else if (a == "-o" || a == "--output") {
      out.output_dir = next(a.c_str());
    } else if (a == "-f" || a == "--filter") {
      out.filter = next(a.c_str());
    } else if (a == "--target") {
      out.target = next(a.c_str());
    } else if (a == "--num-images") {
      out.num_images = std::stoi(next(a.c_str()));
    } else if (a == "--num-streams") {
      out.num_streams = std::stoi(next(a.c_str()));
    } else if (a == "--pinned") {
      out.pinned_memory = true;
    } else if (a == "--no-pinned") {
      out.pinned_memory = false;
    } else if (a == "--no-mask") {
      out.compute_mask = false;
    } else if (a == "--estimated-target") {
      out.use_estimated_basis_as_target = true;
    } else if (a == "--standard-target") {
      out.use_estimated_basis_as_target = false;
    } else if (a == "--cpu") {
      out.cpu_only = true;
    } else if (a == "--benchmark") {
      out.benchmark = true;
    } else if (a == "--csv") {
      out.csv_path = next(a.c_str());
    } else {
      std::cerr << "stainkit: unknown argument: " << a << "\n";
      return false;
    }
  }
  return true;
}

stainkit::StainTarget MakeTarget(const std::string& name) {
  stainkit::StainTarget t;
  t.name = name;
  if (name == "he-royal") {
    // "Royal" profile - slight bluer hematoxylin, warmer eosin.
    t.matrix.values = {0.5826f, 0.7660f, 0.2723f, 0.2053f, 0.8520f, 0.4830f};
    t.target_he_concentrations = {0.60f, 0.75f, 0.30f};
  } else if (name == "he-icm") {
    // "ICM" profile - higher eosin contrast.
    t.matrix.values = {0.6500f, 0.7040f, 0.2860f, 0.0700f, 0.9900f, 0.1200f};
    t.target_he_concentrations = {0.55f, 0.85f, 0.35f};
  } else {
    // "default" / unknown -> Macenko's reference basis.
    t.matrix = stainkit::StainMatrix::Identity();
    t.target_he_concentrations = {0.65f, 0.70f, 0.29f};
  }
  return t;
}

void WriteCsv(const std::vector<stainkit::BenchmarkRecord>& rows,
              const fs::path& path) {
  std::ofstream ofs(path);
  if (!ofs) {
    std::cerr << "stainkit: cannot open CSV for writing: " << path << "\n";
    return;
  }
  ofs << "image_id,width,height,load_ms,copy_h2d_ms,deconvolve_ms,"
         "normalise_ms,mask_ms,copy_d2h_ms,total_ms,cpu_baseline_ms,"
         "speedup\n";
  for (const auto& r : rows) {
    const double speedup =
        (r.total_ms > 0.0) ? r.cpu_baseline_ms / r.total_ms : 0.0;
    ofs << r.image_id << ',' << r.width << ',' << r.height << ',' << r.load_ms
        << ',' << r.copy_h2d_ms << ',' << r.deconvolve_ms << ','
        << r.normalise_ms << ',' << r.mask_ms << ',' << r.copy_d2h_ms << ','
        << r.total_ms << ',' << r.cpu_baseline_ms << ',' << speedup << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs args;
  if (!ParseArgs(argc, argv, args)) {
    PrintHelp(std::cerr);
    return 2;
  }
  if (args.show_help) {
    PrintHelp(std::cout);
    return 0;
  }
  if (args.show_version) {
    std::cout << "stainkit " << kVersion << "\n";
    return 0;
  }
  if (args.input_dir.empty() || args.output_dir.empty()) {
    std::cerr << "stainkit: --input and --output are required\n";
    PrintHelp(std::cerr);
    return 2;
  }
  if (!fs::exists(args.input_dir) || !fs::is_directory(args.input_dir)) {
    std::cerr << "stainkit: input directory does not exist: " << args.input_dir
              << "\n";
    return 2;
  }
  std::error_code ec;
  fs::create_directories(args.output_dir, ec);

  // -- Resolve pipeline target profile --
  const stainkit::StainTarget target = MakeTarget(args.target);

  // -- Resolve pipeline parameters --
  stainkit::PipelineParams params;
  params.num_streams = args.num_streams;
  params.use_pinned_memory = args.pinned_memory;
  params.compute_tissue_mask = args.compute_mask;
  params.overlap_io_with_compute = (args.num_streams > 1);
  params.use_estimated_basis_as_target = args.use_estimated_basis_as_target;

  // -- Discover input images --
  std::vector<fs::path> inputs = stainkit::ListImagesIn(args.input_dir);

  if (args.num_images >= 0 &&
      static_cast<std::size_t>(args.num_images) < inputs.size()) {
    inputs.resize(static_cast<std::size_t>(args.num_images));
  }
  if (inputs.empty()) {
    std::cerr << "stainkit: no supported images in " << args.input_dir << "\n";
    return 1;
  }
  std::cout << "stainkit: " << kVersion << " — processing " << inputs.size()
            << " image(s) from " << args.input_dir << "\n";

  // -- Build the pipeline (with CPU fallback) --
  std::unique_ptr<stainkit::Pipeline> pipeline;
  if (!args.cpu_only) {
    // Defensive: wrap Make() in a SIGSEGV handler so that an ABI mismatch
    // between the build-time CUDA runtime (e.g. 12.8) and the host's CUDA
    // driver (e.g. 13.0) doesn't terminate the whole process. We catch
    // the segfault, log it, and fall back to the CPU reference path.
    pipeline = stainkit::Pipeline::MakeOrFallback();
  }
  if (pipeline) {
    std::cout << "stainkit: using GPU: " << pipeline->DeviceName() << "\n";
  } else {
    std::cout << "stainkit: CUDA unavailable — falling back to CPU "
                 "reference implementation\n";
  }

  // -- Process loop --
  std::vector<stainkit::BenchmarkRecord> bench;
  bench.reserve(inputs.size());

  std::size_t idx = 0;
  for (const auto& path : inputs) {
    const auto t0 = std::chrono::steady_clock::now();
    stainkit::Image img;
    try {
      img = stainkit::ReadImage(path, 3);
    } catch (const std::exception& ex) {
      std::cerr << "stainkit: failed to read " << path << ": " << ex.what()
                << "\n";
      continue;
    }

    const std::string stem = path.stem().string();
    const fs::path out_png = args.output_dir / (stem + "_normalised.png");
    const fs::path out_msk = args.output_dir / (stem + "_mask.png");
    const fs::path out_viz = args.output_dir / (stem + "_panel.png");

    if (pipeline) {
      const auto rr = pipeline->RunWithCpuBaseline(img, params, target);
      stainkit::WriteImage(rr.normalised, out_png);
      if (args.compute_mask) {
        stainkit::WriteImage(rr.tissue_mask, out_msk);
      }
      if (params.write_visualisation) {
        try {
          stainkit::WriteVisualisationPanel(img, rr.normalised, rr.tissue_mask,
                                            out_viz);
        } catch (const std::exception& ex) {
          std::cerr << "stainkit: panel write failed for " << path << ": "
                    << ex.what() << "\n";
        }
      }
      auto r = rr.timing;
      r.image_id = stem;
      bench.push_back(r);
    } else {
      // CPU-only path.
      const auto t1 = std::chrono::steady_clock::now();
      double cpu_ms = 0.0;
      auto normalised =
          stainkit::CpuReferenceStainNormalise(img, params, target, &cpu_ms);
      const auto t2 = std::chrono::steady_clock::now();
      stainkit::WriteImage(normalised, out_png);
      // No tissue mask in CPU path to keep the reference simple.
      stainkit::BenchmarkRecord r;
      r.image_id = stem;
      r.width = img.width;
      r.height = img.height;
      r.total_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
      r.cpu_baseline_ms = cpu_ms;
      bench.push_back(r);
    }

    const auto t1 = std::chrono::steady_clock::now();
    if (args.benchmark) {
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      const auto& last = bench.back();
      std::cout << std::fixed << std::setprecision(2) << "  [" << (++idx) << "/"
                << inputs.size() << "] " << stem << "  GPU=" << last.total_ms
                << "ms  CPU=" << last.cpu_baseline_ms << "ms  wall=" << ms
                << "ms\n";
    } else {
      std::cout << "  [" << (++idx) << "/" << inputs.size() << "] " << stem
                << "\n";
    }
  }

  if (!args.csv_path.empty()) {
    fs::create_directories(args.csv_path.parent_path(), ec);
    WriteCsv(bench, args.csv_path);
    std::cout << "stainkit: benchmark CSV written to " << args.csv_path << "\n";
  }

  std::cout << "stainkit: done — outputs in " << args.output_dir << "\n";
  return 0;
}
