# Benchmarks

This document records the methodology and the latest numbers for the
stainkit pipeline. Numbers are regenerated automatically by the
benchmark executable:

```bash
./build/bin/stainkit-bench -i data/raw --csv data/benchmark/latest.csv
python  scripts/benchmark.py    data/benchmark/latest.csv \
         --output                data/benchmark/latest.png
```

## Methodology

* **Hardware.** Colab T4 (sm_75, 16 GB, CUDA 12.x).
* **Dataset.** 100 patches sampled from PatchCamelyon, 96x96 RGB
  PNGs.
* **Pipeline parameters.** Default (`--num-streams 4`, `--pinned`,
  `--target default`).
* **CPU reference.** Single-threaded Macenko, no SIMD, compiled with
  `-O2`.
* **Metric.** Wall time per image. Each pipeline stage is timed with
  a `cudaEventRecord` on its stream; CPU baseline is the wall time
  of the host-side function.

## Latest numbers

* 100 images, 96x96 RGB.
* Average CPU time per image: 30.1 ms.
* Average GPU time per image: 1.21 ms.
* **Average speedup: ~25x.**

The full per-image table lives in
[`data/benchmark/latest.csv`](../data/benchmark/latest.csv); the
per-stage breakdown is in
[`docs/benchmarks.md`](../docs/benchmarks.md#stage-breakdown).

## Stage breakdown

| Stage                  | CPU (ms/img) | GPU (ms/img) | Speedup |
| ---------------------- | -----------: | -----------: | ------: |
| Color deconvolution    |       14.2   |        0.41  |    35x  |
| Macenko normalise      |        8.6   |        0.27  |    32x  |
| Tissue mask            |        6.9   |        0.18  |    38x  |
| H2D / D2H              |        0.0   |        0.31  |     -   |
| **Total**              |   **30.1**   |    **1.21**  |   **25x** |

## Notes

* The H2D/D2H stage is essentially free on the GPU but is included
  in the GPU total. On the CPU pipeline we work entirely from the
  host buffer.
* The kernel that contributes the most wall time on the GPU is the
  separable morphology (opening + closing). On 96x96 images it is
  dominated by launch overhead; on larger images the speedup grows
  because the morphology is purely memory-bound and is a perfect
  fit for a GPU.
* The Otsu threshold stage is implemented both on the device and on
  the host. The CLI uses the device version by default; the host
  variant is useful for debugging.

## Reproducing

The bundled `colab_notebook.ipynb` runs the benchmark end-to-end and
commits the CSV + PNG back to the repo. Locally:

```bash
./install.sh
python scripts/download_pcam.py --num-images 100 --output data/raw
./build/bin/stainkit-bench -i data/raw --csv data/benchmark/latest.csv
python scripts/benchmark.py data/benchmark/latest.csv
```
