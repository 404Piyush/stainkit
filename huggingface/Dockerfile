# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# stainkit — HuggingFace Space image
# ---------------------------------------------------------------------------
# Builds the stainkit C++/CUDA project inside the official NVIDIA CUDA
# 12.2 devel image, then runs the Gradio app on top of it.
#
# The Space runs on HuggingFace's free T4 small tier (16 GB VRAM) which
# matches the GPU we used for the in-repo benchmarks.
#
# Build flow:
#   1. base: nvidia/cuda:12.2.0-devel-ubuntu22.04 (gcc, nvcc, cuda runtime)
#   2. apt: build-essential, cmake, ninja, libtiff, openslide, python3
#   3. pip: gradio, numpy, Pillow
#   4. copy stainkit source + install.sh
#   5. ./install.sh builds libstainkit_core, libstainkit_cuda, and the
#      `stainkit` + `stainkit-bench` CLIs into /app/build/bin/
#   6. CMD launches app.py which shells out to the `stainkit` CLI
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

# System packages: compilers, image libraries, Python, and both curl +
# wget because the nvidia/cuda images ship inconsistently.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libopenslide-dev \
        libtiff-dev \
        ninja-build \
        python3-dev \
        python3-pip \
        wget && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --quiet --upgrade pip

WORKDIR /app

# Python dependencies (pinned in requirements.txt).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# stainkit source. The repo layout is preserved so that
# include/, src/, tests/, third_party/, etc. resolve at their
# expected paths when CMake runs.
COPY CMakeLists.txt   /app/
COPY include          /app/include
COPY src              /app/src
COPY tests            /app/tests
COPY cmake            /app/cmake
COPY scripts          /app/scripts
COPY install.sh       /app/install.sh
COPY run.sh           /app/run.sh
COPY LICENSE          /app/LICENSE
COPY README.md        /app/README.md
COPY docs             /app/docs

# Vendored stb_image is already in the repo; copy it explicitly so the
# install.sh / cmake build can find it.
COPY third_party      /app/third_party

# Pre-built sample images used by the Gradio Examples gallery. The
# Gradio app reads them from /app/screenshots/.
COPY docs/screenshots /app/screenshots

# Build the C++/CUDA project. We disable the Python bindings and the
# test suite (both rely on FetchContent over the network which is slow
# and unreliable on Space cold-starts); the CLI is what the app needs.
RUN chmod +x /app/install.sh && \
    /app/install.sh --no-tests --no-python

# Sanity-check the build.
RUN /app/build/bin/stainkit --version && \
    /app/build/bin/stainkit --help >/dev/null

# Gradio app — uploaded H&E patches get sent to the stainkit CLI per
# request and the output PNGs come back as images.
COPY huggingface/app.py /app/app.py

# Gradio's default port inside HF Spaces.
EXPOSE 7860

CMD ["python3", "/app/app.py"]