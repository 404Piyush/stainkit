#!/usr/bin/env bash
# stainkit/run.sh
# ---------------------------------------------------------------------------
# One-command demo runner. Expects a `data/raw/` directory with H&E
# images. Creates `data/processed/` and `data/benchmark/` if missing.
# Emits a small before/after montage and a CSV with per-image timings.
#
# Usage:
#   ./run.sh                          # default: all images, GPU, default target
#   ./run.sh --num 50                 # process 50 images
#   ./run.sh --cpu                    # use the CPU reference implementation
#   ./run.sh --target he-royal        # switch the stain target profile
#   ./run.sh --stream 8               # 8 CUDA streams
#   ./run.sh --no-benchmark           # skip the benchmark + CSV output
#   ./run.sh --csv path/to/file.csv   # custom CSV path
#
# Anything after `--` is forwarded verbatim to the underlying stainkit CLI.
# ---------------------------------------------------------------------------

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

INPUT_DIR="data/raw"
OUTPUT_DIR="data/processed"
BENCH_DIR="data/benchmark"
NUM_IMAGES=-1
TARGET="default"
CPU=0
STREAMS=4
PINNED=1
DO_BENCHMARK=1
CSV_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num)          NUM_IMAGES="$2"; shift 2 ;;
        --target)       TARGET="$2"; shift 2 ;;
        --cpu)          CPU=1; shift ;;
        --stream)       STREAMS="$2"; shift 2 ;;
        --no-pinned)    PINNED=0; shift ;;
        --no-benchmark) DO_BENCHMARK=0; shift ;;
        --csv)          CSV_PATH="$2"; shift 2 ;;
        --input)        INPUT_DIR="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        --) shift; break ;;
        *) echo "stainkit run.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "stainkit run.sh: input directory ${INPUT_DIR} does not exist." >&2
    echo "Either drop images into ${INPUT_DIR} or pass --input <dir>." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${BENCH_DIR}"

CLI="./build/bin/stainkit"
if [[ ! -x "${CLI}" ]]; then
    echo "stainkit run.sh: ${CLI} not found. Did you run ./install.sh ?" >&2
    exit 1
fi

if [[ -z "${CSV_PATH}" ]]; then
    CSV_PATH="${BENCH_DIR}/last_run.csv"
fi

ARGS=(
    --input        "${INPUT_DIR}"
    --output       "${OUTPUT_DIR}"
    --target       "${TARGET}"
    --num-streams  "${STREAMS}"
)
if [[ ${CPU} -eq 1 ]]; then
    ARGS+=(--cpu)
else
    ARGS+=(--pinned)
fi
if [[ ${PINNED} -eq 0 ]]; then ARGS+=(--no-pinned); fi
if [[ ${NUM_IMAGES} -gt 0 ]]; then ARGS+=(--num-images "${NUM_IMAGES}"); fi
if [[ ${DO_BENCHMARK} -eq 1 ]]; then
    ARGS+=(--benchmark)
    ARGS+=(--csv "${CSV_PATH}")
fi

echo "stainkit run.sh: invoking ${CLI} ${ARGS[*]}"
"${CLI}" "${ARGS[@]}"
RC=$?

echo ""
echo "stainkit run.sh: outputs in ${OUTPUT_DIR}"
if [[ ${DO_BENCHMARK} -eq 1 ]]; then
    echo "stainkit run.sh: benchmark in ${CSV_PATH}"
fi
exit "${RC}"