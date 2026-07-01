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

# System packages: compilers, image libraries, and both curl + wget
# because the nvidia/cuda images ship inconsistently. NOTE: do NOT
# install python3 from apt - Modal's add_python puts a separate Python
# at /usr/local/bin/python3 and the apt python3 points elsewhere. The
# two site-packages don't overlap, so installing fastapi via apt's
# pip would not be visible to Modal's runtime.
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
        wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (pinned in huggingface/requirements.txt).
COPY huggingface/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# stainkit source. Copy the entire repo so all paths (include/,
# src/, tests/, third_party/, etc.) resolve at their expected paths
# when CMake runs.
COPY . /app/

# Build the C++/CUDA project. We disable the Python bindings and the
# test suite (both rely on FetchContent over the network which is slow
# and unreliable on Space cold-starts); the CLI is what the app needs.
RUN chmod +x /app/install.sh && \
    /app/install.sh --no-tests --no-python

# Sanity-check the build.
RUN /app/build/bin/stainkit --version && \
    /app/build/bin/stainkit --help >/dev/null

# Gradio's default port inside HF Spaces.
EXPOSE 7860

CMD ["python3", "/app/huggingface/app.py"]