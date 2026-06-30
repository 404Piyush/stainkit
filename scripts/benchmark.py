#!/usr/bin/env python3
"""Aggregate a stainkit benchmark CSV into a small bar chart.

Usage:
  python scripts/benchmark.py data/benchmark/last_run.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def read_csv(path: Path) -> tuple[list[str], list[float], list[float]]:
    names, gpu, cpu = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["image_id"])
            gpu.append(float(row["total_ms"]))
            cpu.append(float(row["cpu_baseline_ms"]))
    return names, gpu, cpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/plot.png"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"  csv not found: {args.csv}", file=sys.stderr)
        return 1

    names, gpu, cpu = read_csv(args.csv)
    if not names:
        print("  csv is empty", file=sys.stderr)
        return 1

    speedup = [c / g if g > 0 else 0 for c, g in zip(cpu, gpu)]
    avg = sum(speedup) / len(speedup)

    print(f"  {len(names)} image(s); average speedup = {avg:.2f}x")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.3), 4))
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], gpu, width=0.4, label="GPU (ms)")
        ax.bar([i + 0.2 for i in x], cpu, width=0.4, label="CPU (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=80)
        ax.set_ylabel("milliseconds / image")
        ax.set_title(f"stainkit CPU vs GPU  (avg speedup {avg:.2f}x)")
        ax.legend()
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=120)
        print(f"  wrote plot to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
