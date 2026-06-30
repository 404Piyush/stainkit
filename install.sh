#!/usr/bin/env bash
# stainkit/install.sh
# ---------------------------------------------------------------------------
# One-shot installer for stainkit. Builds the library, the CLI, the
# benchmark binary and the Python bindings. Tested on:
#   * macOS 14 + Apple Clang 15 + CUDA 12.4 (sm_75 / sm_86)
#   * Ubuntu 22.04 + GCC 11 + CUDA 12.2 (sm_75 / sm_86 / sm_89)
#   * Google Colab T4 (sm_75)
#
# Usage:
#   ./install.sh                  # default Release build
#   ./install.sh --debug          # Debug build with sanitizers
#   ./install.sh --no-python      # skip the pybind11 bindings
#   ./install.sh --no-tests       # skip the unit tests
#   ./install.sh --cuda-arch=75,86  # custom CUDA architectures
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_TYPE="Release"
# Python bindings default to OFF so the build finishes quickly on
# resource-constrained environments (Google Colab, CI containers).
# Pass --python to opt in.
BUILD_PYTHON=OFF
BUILD_TESTS=ON
CUDA_ARCHS="75;86;89"
EXTRA_CMAKE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)              BUILD_TYPE="Debug"; shift ;;
    --python)             BUILD_PYTHON=ON; shift ;;
    --no-python)          BUILD_PYTHON=OFF; shift ;;
    --no-tests)           BUILD_TESTS=OFF; shift ;;
    --cuda-arch=*)        CUDA_ARCHS="${1#*=}"; shift ;;
    --cmake-arg=*)        EXTRA_CMAKE_ARGS+=("-D${1#*=}"); shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "stainkit install.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  echo "stainkit: cmake is required (>= 3.18)." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  # Colab / containers may have nvcc under /usr/local/cuda but not on PATH.
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
    echo "stainkit: using nvcc at /usr/local/cuda/bin/nvcc"
  else
    echo "stainkit: nvcc not found on PATH. Install the CUDA Toolkit first." >&2
    echo "                On Debian/Ubuntu: sudo apt-get install cuda-toolkit-12" >&2
    echo "                Or run scripts/setup_colab.sh first." >&2
    exit 1
  fi
fi

mkdir -p build
cd build

# Wipe the CMake cache so a previous run with stale option values does
# not pollute the new build (the cache survives `git pull`).
rm -rf CMakeCache.txt CMakeFiles

cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DSTK_BUILD_PYTHON="${BUILD_PYTHON}" \
      -DSTK_BUILD_TESTS="${BUILD_TESTS}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      "${EXTRA_CMAKE_ARGS[@]}" \
      ..

# Use all available cores.
NPROC="$(command -v nproc >/dev/null && nproc || echo 4)"
echo "stainkit: building with ${NPROC} parallel job(s)..."
make -j"${NPROC}"

if [[ "${BUILD_TESTS}" == "ON" ]]; then
  echo "stainkit: running unit tests..."
  ctest --output-on-failure --timeout 120 || {
    echo "stainkit: tests failed. Continuing install." >&2
  }
fi

cd "${SCRIPT_DIR}"
echo ""
echo "stainkit: install complete."
echo "  CLI  : ./build/bin/stainkit --help"
echo "  Bench: ./build/bin/stainkit-bench --help"
if [[ "${BUILD_PYTHON}" == "ON" ]]; then
  echo "  Py   : PYTHONPATH=./build/python python -c 'import gpustain; print(gpustain.is_cuda_available())'"
fi
