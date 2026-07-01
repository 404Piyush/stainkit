# stainkit

<p align="center">
  <img src="docs/screenshots/logo.png" alt="stainkit" width="180">
</p>

<p align="center">
  <strong>GPU-accelerated H&E stain normalization for digital pathology.</strong>
</p>

<p align="center">
  <a href="https://stainkit.404piyush.me/"><img alt="Live demo" src="https://img.shields.io/badge/demo-stainkit.404piyush.me-crimson"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.x-76b900.svg"></a>
  <a href="https://isocpp.org/"><img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599c.svg"></a>
  <a href="https://cmake.org/"><img alt="CMake" src="https://img.shields.io/badge/CMake-3.18%2B-064f8c.svg"></a>
</p>

Implements the Macenko (2009) stain normalization and Ruifrok-Johnston
(2001) color deconvolution algorithms on CUDA. Designed to be
embedded in whole-slide image pipelines where dozens of slides must
be processed per hour and per-slide latency matters.

---

## Why stainkit

Stain variation between labs is the #1 unsolved problem in computational
pathology: a model trained on Lab A's slides often fails on Lab B's
because the same tissue looks different colors. The Macenko and
Ruifrok-Johnstein algorithms are the industry-standard fix; stainkit
runs them on a GPU so hospitals can process thousands of slides in
batch.

Highlights:

* **Industrial-grade CUDA kernels.** Multi-stream pipeline, pinned
  host memory, asynchronous H↔D transfers.
* **Algorithmic depth.** Implements Macenko (2009) stain normalisation,
  Ruifrok-Johnston (2001) color deconvolution and Otsu tissue masking
  with morphological post-processing.
* **C++ and Python APIs.** Use it as a CLI, embed it in a C++ project
  via `libstainkit`, or `import gpustain` from Python.
* **Quantifiable.** Per-image timing columns for every pipeline stage
  and a CPU reference implementation for ground-truth comparison.
* **Reproducible.** CMake build, GoogleTest suite, GitHub Actions CI,
  Docker image, Colab notebook, **live GPU demo** at
  <https://stainkit.404piyush.me/>. Vercel frontend + Lightning T4
  backend; auto-sleep when idle (~3s cold-start on first request).

---

## Quick start

### On a CUDA machine (Linux / macOS / WSL2)

```bash
git clone https://github.com/404Piyush/stainkit.git
cd stainkit
./install.sh
./run.sh
```

`run.sh` expects a `data/raw/` directory of H&E images (PNG, JPG, BMP,
TGA). It will create `data/processed/` and `data/benchmark/`.

### On Google Colab

1. Open the bundled `colab_notebook.ipynb`.
2. Add a `GITHUB_TOKEN` secret (see [Authentication](#authentication) below).
3. Run all cells. The notebook builds the project, downloads the
   PatchCamelyon sample, runs the pipeline, and pushes the results
   back to the repo.

---

## CLI

```
stainkit --input  <dir>  --output <dir> [--filter macenko]
                        [--target default|he-royal|he-icm]
                        [--num-images N] [--benchmark]
                        [--num-streams N] [--pinned] [--no-mask]
                        [--csv <path>] [--help] [--version]
```

Examples:

```bash
# Process every image in data/raw with the default target profile.
stainkit -i data/raw -o data/processed

# Process at most 32 images and emit a benchmark CSV.
stainkit -i data/raw -o data/processed --num-images 32 --benchmark \
         --csv data/benchmark/run.csv

# Use a different target profile (slight bluer hematoxylin).
stainkit -i data/raw -o data/processed --target he-royal

# Run the CPU reference implementation only.
stainkit -i data/raw -o data/processed --cpu
```

Each processed image yields three artefacts:

| File                       | Contents                                      |
| -------------------------- | --------------------------------------------- |
| `*_normalised.png`         | Stain-normalised RGB image                    |
| `*_mask.png`               | Tissue mask (white = tissue)                  |
| `*_panel.png`              | 3-panel visualisation: input / output / mask  |

---

## Python API

```python
import gpustain

# Read an image as a (H, W, 3) uint8 array.
img = gpustain.read_image("patches/patient_001.png")

# Run the full pipeline on the GPU.
result = gpustain.run(img, target="default")

# Save artefacts.
gpustain.write_image(result.normalised_array(), "out_normalised.png")
gpustain.write_image(result.mask_array(),      "out_mask.png")

# Inspect per-stage timings.
print(result.timing.total_ms, "ms on GPU")
print(result.timing.cpu_baseline_ms, "ms on CPU")
```

The Python module is a thin pybind11 wrapper around the C++ `Pipeline`
class, so its performance is identical to the CLI.

---

## Architecture

```
                       +---------------------------+
   H&E image (RGB) --> |  Host staging (pinned)   | --+ (H2D)
                       +---------------------------+   |
                                                       v
                       +---------------------------+
                       |  CUDA stream i            |
                       |   - RGB -> OD             |
                       |   - Color deconvolution   |
                       |   - Macenko basis         |
                       |   - Reconstruct (target)  |
                       |   - Otsu tissue mask      |
                       |   - Morphology            |
                       +---------------------------+
                                                       |
                                                       v
                       +---------------------------+   | (D2H)
   Normalised RGB  <-- |  Host write-back          | <-+
   Tissue mask     <-- |  (stb_image_write)        |
   CSV timings     <-- |  (per-stage records)      |
                       +---------------------------+
```

* The pipeline owns one `cudaStream_t` per inflight image. By default
  four streams are used; pass `--num-streams` to change it.
* The CPU baseline implementation in `src/cpu_reference.cpp` is a
  single-threaded Macenko variant. It is used both for the "CPU
  baseline" timing column and for unit tests.
* Pinning is enabled by default (`--pinned`). On Linux it uses
  `cudaHostAllocDefault`; the harness can fall back to page-locked
  memory if pinned allocation fails.

---

## Project layout

```
stainkit/
├── CMakeLists.txt              # Top-level build
├── include/stainkit/           # Public C++ headers
├── src/
│   ├── io.cpp                  # stb_image wrappers
│   ├── pipeline.cpp            # Pipeline orchestration
│   ├── cpu_reference.cpp       # CPU Macenko baseline
│   ├── main.cpp                # CLI entry point
│   ├── kernels/                # CUDA kernel files (.cu)
│   ├── benchmarks/             # CPU vs GPU benchmark binary
│   └── bindings/               # pybind11 module
├── tests/                      # GoogleTest suite
├── third_party/stb/            # Vendored stb_image
├── python/gpustain/            # Python wrapper helpers + tests
├── scripts/                    # Dataset downloaders
├── docs/                       # Algorithms, API, benchmarks
├── .github/workflows/          # CI
├── Dockerfile
├── colab_notebook.ipynb
├── install.sh
├── run.sh
├── LICENSE                     # Apache 2.0
├── README.md                   # ← you are here
└── .clang-format               # Google C++ style
```

---

## Building from source

Prerequisites:

* CMake ≥ 3.18
* CUDA Toolkit ≥ 11.0 (sm_75 or newer)
* C++17 compiler (GCC 9+, Clang 10+, MSVC 16.8+)
* Python ≥ 3.8 (only if you want the pybind11 module)
* OpenMP (optional; the CPU reference uses it when available)

```bash
./install.sh                    # Release build
./install.sh --debug            # Debug + sanitizers
./install.sh --no-python        # skip the pybind11 module
./install.sh --cuda-arch=75,86  # custom architectures
```

The build drops binaries in `build/bin/`, the static libraries in
`build/lib/`, and (if enabled) the Python module in
`build/python/gpustain/`.

To make `gpustain` importable from anywhere:

```bash
PYTHONPATH="$(pwd)/build/python" python -c "import gpustain; print(gpustain.is_cuda_available())"
```

---

## Authentication (Colab + GitHub)

The Colab notebook needs a GitHub Personal Access Token (PAT) to push
the benchmark results back to the repo.

1. Go to https://github.com/settings/tokens.
2. Click **Generate new token → Generate new token (classic)**.
3. Note: `colab-stainkit`. Expiration: your choice.
4. Scope: only `repo`.
5. Generate the token and copy the value (you only see it once).
6. In Colab, open the left sidebar → **🔑 Secrets** → **+ Add new secret**.
7. Name: `GITHUB_TOKEN`. Value: paste the token. Enable **Notebook access**.

The bundled `colab_notebook.ipynb` reads the secret with
`from google.colab import userdata; userdata.get('GITHUB_TOKEN')`.

---

## Algorithms

The pipeline implements two named algorithms from the literature. The
mathematical details live in [`docs/algorithms.md`](docs/algorithms.md).

* **Macenko (2009) stain normalisation.** The estimated basis is the
  pair of unit vectors whose 1st/99th percentile projections of the
  per-pixel OD pair lie along. Reconstructed RGB uses the *target*
  matrix and user-specified concentrations.
* **Ruifrok-Johnston (2001) color deconvolution.** A 3x3 matrix
  inversion whose columns are the unit RGB vectors of the three
  reagent stains; the kernel writes the per-pixel concentrations in
  optical-density space.
* **Otsu tissue masking.** A device-side histogram reduction followed
  by the standard inter-class variance maximisation. The mask is
  post-processed with an opening and a closing of radius
  `params.otsu_smoothing_radius` to remove dust and fill small holes.

---

## Performance

A representative run on a Colab T4 (sm_75) on 10 synthetic H&E patches
at 256x256 RGB (see [`docs/benchmarks.md`](docs/benchmarks.md) for the
full table):

| Stage                  | CPU (ms/img) | GPU (ms/img) | Speedup |
| ---------------------- | -----------: | -----------: | ------: |
| Color deconvolution    |        3.5   |        0.34  |    10x  |
| Macenko normalise      |        2.0   |        1.19  |     2x  |
| Tissue mask            |        1.2   |        1.40  |     1x  |
| H2D / D2H              |        0.0   |        2.30  |     —   |
| **Total**              |    **6.05**  |    **5.22**  |   **1.16x** |

The GPU wins modestly on 256x256 patches because launch overhead and
the H2D/D2H transfers dominate. The color-deconvolution kernel is the
single biggest GPU win (10x), and on larger images the relative
advantage grows further as the kernel work scales with `npix` while
launch overhead stays constant.

See [`docs/screenshots.md`](docs/screenshots.md) for the
before/after panel visualisations and the CPU vs GPU benchmark
plot.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow. Highlights:

* Code is `clang-formatted` with the Google C++ style. The bundled
  `.clang-format` matches `BasedOnStyle: Google`.
* New kernels must have a matching GoogleTest in `tests/`.
* Public APIs (anything under `include/stainkit/`) follow
  [the Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

---

## Citation

If you use stainkit in academic work, please cite the underlying
algorithms:

```bibtex
@article{macenko2009reference,
  title={A reference image set for H\&E stain normalization},
  author={Macenko, Marc and Niethammer, Marc and Marron, J. S. and Borland, David and Woosley, John T. and Guan, Xiaomin and Schmitt, Charles and Thomas, Nancy E.},
  journal={Proceedings of the IEEE International Symposium on Biomedical Imaging},
  year={2009}
}

@article{ruifrok2001quantification,
  title={Quantification of histochemical staining by color deconvolution},
  author={Ruifrok, A. C. and Johnston, D. A.},
  journal={Analytical and Quantitative Cytology and Histology},
  volume={23},
  number={4},
  pages={291--299},
  year={2001}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
