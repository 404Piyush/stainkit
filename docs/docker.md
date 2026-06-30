# Docker

Build the project inside a CUDA-enabled container so the build is
reproducible on any host that can run Docker.

```bash
docker build -t stainkit:latest .
docker run --rm --gpus all -v "$(pwd)/data:/work/data" stainkit:latest \
    /work/install.sh && /work/run.sh
```

The image is based on `nvidia/cuda:12.2.0-devel-ubuntu22.04` and
includes CMake, GCC, OpenSlide, libtiff and Python 3. The build
artefacts land in `build/`; the processed images land in
`data/processed/`.

## Variants

* `Dockerfile` — full build (CUDA + C++ + Python bindings + tests).
* `Dockerfile.minimal` (TODO) — runtime-only image for production
  deployments. Strips the build toolchain and the test suite.
