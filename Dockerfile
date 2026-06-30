# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# stainkit / Dockerfile
# ---------------------------------------------------------------------------
# Build a reproducible CUDA-enabled image. Use as:
#   docker build -t stainkit .
#   docker run --rm --gpus all -v $PWD/data:/work/data stainkit /work/run.sh
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System packages: build toolchain, OpenSlide, libtiff, Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libopenslide-dev \
        libtiff-dev \
        ninja-build \
        python3-dev \
        python3-pip \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Python packages used by the helper scripts.
RUN pip install --no-cache-dir --quiet numpy matplotlib

WORKDIR /work

# Pre-fetch the third-party single-header stb image library.
ARG STB_VERSION=2.28
RUN mkdir -p third_party/stb && \
    wget -qO third_party/stb/stb_image.h \
        https://raw.githubusercontent.com/nothings/stb/${STB_VERSION}/stb_image.h && \
    wget -qO third_party/stb/stb_image_write.h \
        https://raw.githubusercontent.com/nothings/stb/${STB_VERSION}/stb_image_write.h

# Copy the rest of the source. Doing this in two COPY commands lets the
# builder cache the third-party download above.
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY docs ./docs
COPY install.sh run.sh README.md LICENSE ./
COPY cmake ./cmake

# Build with the bundled install script.
RUN ./install.sh --no-tests

CMD ["/work/run.sh"]
