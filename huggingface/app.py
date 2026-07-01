"""Gradio demo for stainkit — GPU-accelerated H&E stain normalisation.

Hosted on HuggingFace Spaces with a free T4 GPU tier. The app uploads
an H&E patch, runs the `stainkit` CLI on it, and returns the
normalised RGB image plus the tissue mask.

Why subprocess and not the pybind11 module?
------------------------------------------
The pybind11 module (`gpustain`) ships a compiled `.so` for a specific
Python version and CUDA toolkit. Building a manylinux-compatible wheel
across CUDA versions is its own project; calling the CLI is simpler
and works regardless of the Python ABI.

Build flow on HF Spaces (handled by the Dockerfile):
  1. nvidia/cuda:12.2.0-devel base image
  2. `install.sh` builds stainkit + stainkit-bench + libstainkit_*
  3. The Gradio app shells out to `./build/bin/stainkit` per request.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The Dockerfile builds the CLI into /app/build/bin/stainkit. We allow the
# path to be overridden so the same code works in CI / local dev too.
STK_BIN = Path(os.environ.get("STK_BIN", "/app/build/bin/stainkit"))
WORKDIR = Path(os.environ.get("STK_WORKDIR", "/tmp/stk_uploads"))
WORKDIR.mkdir(parents=True, exist_ok=True)

# Each request gets a fresh subdirectory so concurrent users don't trample
# each other's outputs. Gradio's queue serialises requests by default but
# we still isolate the on-disk artefacts per request.
def _request_dir(req_id: str) -> Path:
    d = WORKDIR / req_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def _save_upload(image: np.ndarray, dest: Path) -> None:
    """Write a (H, W, 3) uint8 numpy array to disk as PNG."""
    if image is None:
        raise ValueError("no image supplied")
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(dest)


def normalize(image: np.ndarray, target: str, num_streams: int,
              progress: gr.Progress = gr.Progress()) -> tuple:
    """Run stainkit on the uploaded image and return (normalised, mask, info)."""
    if image is None:
        raise gr.Error("Please upload an H&E patch first.")

    progress(0.1, desc="Saving upload…")
    req_id = f"{int(time.time() * 1000)}_{os.getpid()}"
    req_dir = _request_dir(req_id)
    in_path  = req_dir / "input.png"
    out_dir  = req_dir / "out"
    out_dir.mkdir(exist_ok=True)
    _save_upload(image, in_path)

    # The CLI emits <stem>_normalised.png, <stem>_mask.png, <stem>_panel.png
    # where <stem> is `input` minus the .png extension.
    stem = in_path.stem
    cmd = [
        str(STK_BIN),
        "--input",   str(req_dir),
        "--output",  str(out_dir),
        "--target",  target,
        "--num-streams", str(num_streams),
        "--num-images", "1",
    ]

    progress(0.3, desc="Running GPU pipeline…")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    progress(0.85, desc="Reading outputs…")

    if proc.returncode != 0:
        # Clean up and surface the error.
        shutil.rmtree(req_dir, ignore_errors=True)
        raise gr.Error(
            f"stainkit exited with code {proc.returncode}\n"
            f"stdout: {proc.stdout[-1000:]}\n"
            f"stderr: {proc.stderr[-1000:]}"
        )

    # Load the three artefacts.
    normalised_path = out_dir / f"{stem}_normalised.png"
    mask_path      = out_dir / f"{stem}_mask.png"
    panel_path     = out_dir / f"{stem}_panel.png"
    if not (normalised_path.exists() and mask_path.exists()):
        shutil.rmtree(req_dir, ignore_errors=True)
        raise gr.Error(f"stainkit did not produce expected outputs in {out_dir}")

    normalised = np.array(Image.open(normalised_path).convert("RGB"))
    mask       = np.array(Image.open(mask_path).convert("RGB"))
    panel      = np.array(Image.open(panel_path).convert("RGB"))

    progress(1.0, desc="Done.")

    info = {
        "wall_ms":          round(wall_ms, 2),
        "image_shape":      list(image.shape),
        "normalised_shape": list(normalised.shape),
        "mask_shape":       list(mask.shape),
        "target":           target,
        "num_streams":      num_streams,
    }
    # Echo the CLI's stdout lines so the user sees the GPU line and the
    # per-stage timings right in the UI.
    cli_summary = proc.stdout.strip().splitlines()[-6:]
    info["cli_log_tail"] = "\n".join(cli_summary)

    # Best-effort cleanup.
    shutil.rmtree(req_dir, ignore_errors=True)

    return normalised, mask, panel, info


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
# Pre-populate the example gallery with the synthetic H&E images that ship
# with the repo under docs/screenshots/. The Dockerfile copies these into
# the Space image at /app/screenshots/.
EXAMPLES_DIR = Path(os.environ.get("STK_EXAMPLES", "/app/screenshots"))
EXAMPLES = sorted(
    str(p) for p in EXAMPLES_DIR.glob("sample_*_panel.png")
) if EXAMPLES_DIR.exists() else []


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="stainkit — GPU H&E Stain Normalisation",
        theme=gr.themes.Soft(),
    ) as ui:
        gr.Markdown(
            """
            # stainkit — GPU H&E Stain Normalisation

            Upload a histopathology (H&E) patch and the pipeline will:
            1. estimate its haematoxylin/eosin stain basis (Macenko 2009),
            2. reconstruct the image using a target basis,
            3. compute an Otsu-based tissue mask.

            All on GPU. Built on CUDA 12 + Gradio. Runs on the free
            HuggingFace T4 tier.
            """
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(
                    label="H&E patch", type="numpy", height=320,
                    sources=["upload", "clipboard"],
                )
                with gr.Row():
                    target      = gr.Dropdown(
                        choices=["default", "he-royal", "he-icm"],
                        value="default",
                        label="Target stain profile",
                    )
                    num_streams = gr.Slider(
                        minimum=1, maximum=4, step=1, value=4,
                        label="CUDA streams",
                    )
                run_btn = gr.Button("Normalise on GPU", variant="primary")
            with gr.Column():
                out_norm = gr.Image(label="Stain-normalised", height=320)
                out_mask = gr.Image(label="Tissue mask",       height=320)
        with gr.Row():
            out_panel = gr.Image(label="3-panel view (input | output | mask)",
                                 height=200)
        with gr.Row():
            info = gr.JSON(label="Run info")

        if EXAMPLES:
            gr.Examples(
                examples=[[str(p)] for p in EXAMPLES],
                inputs=[inp],
                label="Or try one of the sample images below:",
            )

        run_btn.click(
            fn=normalize,
            inputs=[inp, target, num_streams],
            outputs=[out_norm, out_mask, out_panel, info],
            api_name="normalize",
        )
    return ui


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(max_size=8).launch(server_name="0.0.0.0", server_port=7860)