#!/usr/bin/env bash
# stainkit/scripts/setup_colab.sh
# ---------------------------------------------------------------------------
# One-shot environment setup for Google Colab. Installs the CUDA
# toolchain shims, the Python prerequisites, and the third-party
# libraries the build needs (libtiff, openslide).
# ---------------------------------------------------------------------------

set -euo pipefail

echo "stainkit: installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  # Detect the CUDA major version bundled with the Colab runtime so we
  # install the matching nvcc. Falls back to 12 (current Colab default).
  CUDA_MAJOR="${CUDA_MAJOR:-12}"
  CUDA_PKG="cuda-toolkit-${CUDA_MAJOR}"
  apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates \
      libtiff-dev libopenslide-dev python3-dev python3-pip \
      "${CUDA_PKG}" || {
        echo "stainkit: ${CUDA_PKG} unavailable, falling back to nvidia-cuda-toolkit" >&2
        apt-get install -y --no-install-recommends nvidia-cuda-toolkit
      }
  # Make sure /usr/local/cuda (the canonical Colab location) is on PATH.
  if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    echo 'export PATH="/usr/local/cuda/bin:${PATH}"' >> /etc/profile.d/cuda.sh
  fi
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y cmake gcc-c++ git wget libtiff-devel openslide-devel \
      python3-devel python3-pip
elif command -v brew >/dev/null 2>&1; then
  brew install cmake libtiff openslide
else
  echo "stainkit: no supported package manager found" >&2
  exit 1
fi

echo "stainkit: installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet numpy matplotlib

echo "stainkit: environment ready."
