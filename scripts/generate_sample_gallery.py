"""Generate 6 diverse synthetic H&E-like sample images.

Each sample has different stain characteristics so the demo
shows the algorithm working across varied inputs.

Output: docs/screenshots/samples/sample_NN_*.png (plus thumbnail).
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))
    idat = zlib.compress(raw, 9)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Synthetic tissue generators. Each function returns an HxWx3 bytearray.
# ---------------------------------------------------------------------------
def blank_pink(w: int, h: int) -> bytearray:
    """A flat pinkish background — minimal contrast (worst case for Otsu)."""
    out = bytearray(w * h * 3)
    for i in range(w * h):
        out[3 * i + 0] = 240  # pale pink
        out[3 * i + 1] = 210
        out[3 * i + 2] = 220
    return out


def single_nucleus(w: int, h: int) -> bytearray:
    """A pale background with one dark purple nucleus in the centre."""
    out = bytearray(w * h * 3)
    cx, cy = w / 2, h / 2
    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            r = math.sqrt(dx * dx + dy * dy)
            nucleus = math.exp(-r * r / (w * h * 0.004))
            i = 3 * (y * w + x)
            out[i + 0] = max(0, min(255, int(220 - 90 * nucleus)))
            out[i + 1] = max(0, min(255, int(180 - 80 * nucleus)))
            out[i + 2] = max(0, min(255, int(190 - 30 * nucleus)))
    return out


def many_nuclei(w: int, h: int) -> bytearray:
    """A field of densely packed round nuclei (mimics high cellularity tissue)."""
    out = bytearray(w * h * 3)
    bg = (235, 215, 220)
    for i in range(w * h):
        out[3 * i + 0] = bg[0]
        out[3 * i + 1] = bg[1]
        out[3 * i + 2] = bg[2]
    rng_cells = [
        (0.30, 0.30, 28, (90, 60, 130)),
        (0.62, 0.28, 24, (95, 70, 140)),
        (0.20, 0.65, 30, (85, 55, 125)),
        (0.74, 0.62, 22, (100, 75, 145)),
        (0.42, 0.78, 26, (95, 65, 135)),
        (0.78, 0.18, 20, (110, 80, 150)),
        (0.10, 0.45, 24, (88, 58, 128)),
        (0.55, 0.45, 28, (92, 62, 132)),
        (0.35, 0.50, 18, (105, 75, 145)),
        (0.65, 0.78, 22, (98, 70, 138)),
    ]
    for fx, fy, r, (rcol, gcol, bcol) in rng_cells:
        ccx, ccy = fx * w, fy * h
        for y in range(max(0, int(ccy - 2 * r)), min(h, int(ccy + 2 * r))):
            for x in range(max(0, int(ccx - 2 * r)), min(w, int(ccx + 2 * r))):
                d = math.sqrt((x - ccx) ** 2 + (y - ccy) ** 2)
                if d > 2 * r:
                    continue
                v = math.exp(-(d * d) / (r * r * 0.7))
                i = 3 * (y * w + x)
                out[i + 0] = max(0, min(255, int(bg[0] + v * (rcol - bg[0]))))
                out[i + 1] = max(0, min(255, int(bg[1] + v * (gcol - bg[1]))))
                out[i + 2] = max(0, min(255, int(bg[2] + v * (bcol - bg[2]))))
    return out


def tissue_with_stroma(w: int, h: int) -> bytearray:
    """Mixed tissue with eosinophilic (pink) stroma and hematoxylin (purple) glands."""
    out = bytearray(w * h * 3)
    # Pink stroma background
    for i in range(w * h):
        out[3 * i + 0] = 235
        out[3 * i + 1] = 200
        out[3 * i + 2] = 215
    # Several "glands" (clusters of dark cells)
    glands = [
        (0.25, 0.30, 35),
        (0.70, 0.35, 28),
        (0.45, 0.65, 40),
        (0.80, 0.75, 30),
    ]
    for fx, fy, r in glands:
        for y in range(max(0, int(fy * h - 2 * r)), min(h, int(fy * h + 2 * r))):
            for x in range(max(0, int(fx * w - 2 * r)), min(w, int(fx * w + 2 * r))):
                d = math.sqrt((x - fx * w) ** 2 + (y - fy * h) ** 2)
                if d > 2 * r:
                    continue
                v = math.exp(-(d * d) / (r * r * 0.6))
                i = 3 * (y * w + x)
                out[i + 0] = max(0, min(255, int(235 - 130 * v)))
                out[i + 1] = max(0, min(255, int(200 - 130 * v)))
                out[i + 2] = max(0, min(255, int(215 - 80 * v)))
    return out


def heavy_stain(w: int, h: int) -> bytearray:
    """A heavily-stained (over-saturated) image — purple everywhere."""
    out = bytearray(w * h * 3)
    for i in range(w * h):
        out[3 * i + 0] = 150
        out[3 * i + 1] = 100
        out[3 * i + 2] = 180
    return out


def pale_stain(w: int, h: int) -> bytearray:
    """A very lightly stained image — near-white background."""
    out = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            cx, cy = x - w / 2, y - h / 2
            r = math.sqrt(cx * cx + cy * cy)
            nucleus = math.exp(-r * r / (w * h * 0.01))
            i = 3 * (y * w + x)
            out[i + 0] = max(0, min(255, int(248 - 40 * nucleus)))
            out[i + 1] = max(0, min(255, int(245 - 50 * nucleus)))
            out[i + 2] = max(0, min(255, int(240 - 30 * nucleus)))
    return out


SAMPLES = [
    ("01_blank_pink",      "Blank (no tissue)",     256, blank_pink),
    ("02_single_nucleus",  "Single nucleus",        256, single_nucleus),
    ("03_many_nuclei",     "Densely cellular",      320, many_nuclei),
    ("04_stroma_glands",   "Tissue with stroma",     320, tissue_with_stroma),
    ("05_heavy_stain",     "Heavy / saturated",      256, heavy_stain),
    ("06_pale_stain",      "Light / faded",          256, pale_stain),
]


def main() -> int:
    for name, label, size, fn in SAMPLES:
        rgb = fn(size, size)
        path = OUTPUT_DIR / f"{name}.png"
        write_png(path, size, size, bytes(rgb))
        print(f"  {path.name:32s} {label:24s} {size}x{size}  {len(rgb)} bytes")
    print(f"\nWrote {len(SAMPLES)} samples to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())