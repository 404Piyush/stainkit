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
    https://404piyush--stainkit-demo-fastapi-app.modal.run
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
# Image: build stainkit inside an NVIDIA CUDA 12.2 container using the
# Dockerfile that ships in the repo (huggingface/Dockerfile).
# ---------------------------------------------------------------------------
stainkit_image = (
    modal.Image.from_dockerfile(
        REPO_ROOT / "huggingface" / "Dockerfile",
        context_dir=REPO_ROOT,
    )
    .env({
        "STK_BIN":     "/app/build/bin/stainkit",
        "STK_WORKDIR": "/tmp/stk_uploads",
    })
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
    container_idle_timeout=120,        # spin down after 2 min idle
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

        # Per-request scratch directory. Modal reuses the same container
        # across requests so we keep names unique with a timestamp+pid.
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