#!/usr/bin/env python3
"""Generate a synthetic H&E image for use in CI and demos.

The synthetic image has:
    * a pink background (eosin-dominant),
    * a purple "nucleus" in the centre (hematoxylin-dominant),
    * a few darker blobs (more hematoxylin).

This is enough to exercise the Macenko pipeline end-to-end on machines
that have no real dataset available.

Usage:
  python scripts/generate_sample_data.py --output data/raw
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path


def write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    """Write a minimal 8-bit RGB PNG."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[y * width * 3:(y + 1) * width * 3]
                   for y in range(height))
    idat = zlib.compress(raw, 9)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) +
                     chunk(b"IEND", b""))


def synth_he(width: int, height: int) -> bytes:
    """Return a `width*height*3` byte array of H&E-ish pixels."""
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            # Eosin-dominant background.
            r, g, b = 230, 200, 215
            # Hematoxylin-dominant "nucleus" at the centre.
            cx, cy = x - width * 0.5, y - height * 0.5
            r_core = (cx * cx + cy * cy) ** 0.5
            core = max(0.0, 1.0 - r_core / (min(width, height) * 0.15))
            r = int(r * (1 - core) + 80 * core)
            g = int(g * (1 - core) + 60 * core)
            b = int(b * (1 - core) + 130 * core)
            # A few extra "cells" in a grid.
            for gx in range(2, 8):
                for gy in range(2, 6):
                    px = gx * width / 8
                    py = gy * height / 6
                    d2 = (x - px) ** 2 + (y - py) ** 2
                    if d2 < (min(width, height) * 0.04) ** 2:
                        core2 = max(0.0, 1.0 - d2 / ((min(width, height) * 0.04) ** 2))
                        r = int(r * (1 - core2) + 110 * core2)
                        g = int(g * (1 - core2) + 80 * core2)
                        b = int(b * (1 - core2) + 150 * core2)
            pixels[3 * (y * width + x) + 0] = max(0, min(255, r))
            pixels[3 * (y * width + x) + 1] = max(0, min(255, g))
            pixels[3 * (y * width + x) + 2] = max(0, min(255, b))
    return bytes(pixels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_images):
        rgb = synth_he(args.width, args.height)
        write_png(args.output / f"sample_{i:03d}.png",
                  args.width, args.height, rgb)
    print(f"  wrote {args.num_images} synthetic H&E images to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
