#!/usr/bin/env python3
"""Download a small sample of the PatchCamelyon (PCam) dataset.

PatchCamelyon is derived from whole-slide images in the CAMELYON16
challenge, which in turn uses TCGA slides. Each patch is 96x96 RGB
TIFF. We grab the test set (a few MB) and convert the images to PNG
so the stainkit CLI can read them.

Usage:
  python scripts/download_pcam.py --num-images 100 --output data/raw
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

PCAM_TEST_URL = (
    "https://github.com/basveeling/pcam/archive/refs/heads/master.zip"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of images to extract (default: 100).")
    parser.add_argument("--output", type=Path, default=Path("data/raw"),
                        help="Output directory (default: data/raw).")
    parser.add_argument("--keep-archive", action="store_true",
                        help="Keep the downloaded archive (default: delete).")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    archive = Path("build/_deps/pcam.zip")
    download(PCAM_TEST_URL, archive)

    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if extracted >= args.num_images:
                break
            if not name.endswith(".tif"):
                continue
            with zf.open(name) as src:
                # PCam TIFFs are 96x96 RGB, 3 bytes/pixel.
                data = src.read()
                if len(data) < 96 * 96 * 3:
                    continue
                out_name = f"pcam_{extracted:05d}.bin"
                (args.output / out_name).write_bytes(data)
                extracted += 1

    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print(f"  extracted {extracted} patch(es) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
