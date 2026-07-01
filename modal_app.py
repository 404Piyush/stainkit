"""Modal Labs deploy for stainkit.

Modal gives you real CUDA-capable GPUs (T4/A10G/A100) on a free tier
($30/month credit, ~100 containers + 10 concurrent GPUs).

The whole stainkit Docker image (CUDA 12.2 + the C++/CUDA build) runs
inside Modal. Each request shells out to the `stainkit` CLI per image.

Setup (one-time):
    pip install modal
    modal token new                  # opens browser, login with GitHub

Deploy:
    cd stainkit
    modal deploy modal_app.py

The deploy prints a permanent URL like:
    https://piyushutkarxb--stainkit-demo-fastapi-app.modal.run

NOTE: we do NOT use Image.from_dockerfile here because Modal needs to
own the Python install (for the runtime hook). Instead we layer:
  1. base = nvidia/cuda:12.2.0-devel-ubuntu22.04 (compiler + CUDA libs)
  2. apt_install = compilers, libtiff, libopenslide
  3. run_commands = build stainkit via ./install.sh
  4. pip_install = fastapi, uvicorn (Python web layer)
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Image: layer the CUDA base image, the apt packages, the C++ build, and
# the Python web dependencies. Modal owns the Python install so the
# apt python3 packages would never be visible to the runtime.
# ---------------------------------------------------------------------------
stainkit_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "git", "curl", "wget",
        "libtiff-dev", "libopenslide-dev", "ca-certificates",
    )
    .add_local_dir(REPO_ROOT, "/app", copy=True)
    .run_commands(
        # Sanity check: the binary exists and prints its version.
        "ls /app",
        "cd /app && bash install.sh --no-tests --no-python",
        "/app/build/bin/stainkit --version",
        "/app/build/bin/stainkit --help > /dev/null && echo CLI_OK",
    )
    .env({
        "STK_BIN":     "/app/build/bin/stainkit",
        "STK_WORKDIR": "/tmp/stk_uploads",
    })
    .pip_install("fastapi==0.115.0", "uvicorn==0.32.0")
)

app = modal.App("stainkit-demo", image=stainkit_image)

# ---------------------------------------------------------------------------
# FastAPI endpoint. POST a PNG/JPEG file as the body; get back the
# normalised image + tissue mask + 3-panel as base64 PNGs in JSON.
# ---------------------------------------------------------------------------
@app.function(
    gpu="T4",
    cpu=2,
    memory=2048,
    timeout=180,
    scaledown_window=120,             # spin down after 2 min idle
)
@modal.asgi_app()
def fastapi_app():
    web = FastAPI(title="stainkit demo")

    @web.get("/healthz")
    def healthz():
        return {"status": "ok", "gpu": "T4"}

    @web.post("/normalize")
    async def normalize_endpoint(req: Request):
        body = await req.body()
        if not body:
            raise HTTPException(400, "empty body; POST a PNG/JPEG file")

        work = Path("/tmp/stk_uploads")
        work.mkdir(parents=True, exist_ok=True)
        req_id  = f"{int(time.time() * 1000)}_{os.getpid()}"
        req_dir = work / req_id
        req_dir.mkdir(exist_ok=True)
        in_path  = req_dir / "input.png"
        out_dir = req_dir / "out"
        out_dir.mkdir(exist_ok=True)
        in_path.write_bytes(body)

        stem = in_path.stem
        cmd = [
            "/app/build/bin/stainkit",
            "--input",      str(req_dir),
            "--output",     str(out_dir),
            "--target",     "default",
            "--num-streams", "4",
            "--num-images",  "1",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            shutil.rmtree(req_dir, ignore_errors=True)
            raise HTTPException(
                500,
                f"stainkit failed: rc={proc.returncode}\n"
                f"stdout: {proc.stdout[-1000:]}\n"
                f"stderr: {proc.stderr[-1000:]}"
            )

        normalised = (out_dir / f"{stem}_normalised.png").read_bytes()
        mask       = (out_dir / f"{stem}_mask.png").read_bytes()
        panel      = (out_dir / f"{stem}_panel.png").read_bytes()
        shutil.rmtree(req_dir, ignore_errors=True)

        return JSONResponse({
            "wall_ms":           round(wall_ms, 2),
            "cli_log_tail":      "\n".join(
                proc.stdout.strip().splitlines()[-6:]),
            "normalised_png_b64": base64.b64encode(normalised).decode(),
            "mask_png_b64":       base64.b64encode(mask).decode(),
            "panel_png_b64":      base64.b64encode(panel).decode(),
        })

    return web