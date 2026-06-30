#!/usr/bin/env python3
"""Download a small number of TCGA whole-slide images via the GDC API.

By default we ask the GDC API for files that match:
    * project_id = TCGA-BRCA
    * data_type  = Slide Image
    * data_format= SVS

We then pick the smallest few (so they fit in a Colab session) and
download them. Each WSI is multi-gigapixel; we re-tile them with
openslide at runtime and feed the tiles to stainkit.

Usage:
  python scripts/download_tcga.py --num-images 5 --output data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def query_gdc(num_files: int) -> list[dict]:
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "project_id", "value": "TCGA-BRCA"}},
                {"op": "=", "content": {"field": "data_type",   "value": "Slide Image"}},
                {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            ],
        },
        "fields": "file_id,file_name,file_size",
        "format": "json",
        "size": num_files * 4,  # over-fetch; we will sort
    }
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        GDC_FILES_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    hits = data.get("data", {}).get("hits", [])
    hits.sort(key=lambda h: h.get("file_size", 1 << 30))
    return hits[:num_files]


def download_one(file_id: str, dest: Path) -> None:
    url  = f"https://api.gdc.cancer.gov/data/{file_id}"
    print(f"  fetching {url} -> {dest}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        files = query_gdc(args.num_images)
    except Exception as ex:
        print(f"  GDC query failed: {ex}", file=sys.stderr)
        return 1

    if not files:
        print("  no TCGA files matched the query", file=sys.stderr)
        return 1

    for f in files:
        out = args.output / f["file_name"]
        if out.exists():
            print(f"  already have {out}, skipping")
            continue
        try:
            download_one(f["file_id"], out)
        except Exception as ex:
            print(f"  download failed for {f['file_name']}: {ex}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
