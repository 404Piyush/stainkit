"""Lightning Studio web demo for stainkit.

Runs a FastAPI server on port 8000 in the Lightning Studio. The studio
auto-sleeps when idle (configured to 60s of no traffic) and wakes up
when a request arrives — so the GPU only bills while a request is in
flight.

Usage from a Lightning Studio terminal:

    pip install --quiet fastapi uvicorn python-multipart
    python lightning_app.py            # blocks, serves forever

Lightning exposes port 8000 via:
    lightning studio add-ports 8000 --name stainkit-dev

Endpoints:
    GET  /            - HTML landing page with a tiny upload form
    GET  /healthz     - JSON health check
    POST /normalize   - multipart/form-data with field "file"; returns
                        JSON {normalised_b64, mask_b64, panel_b64, wall_ms}

The POST handler invokes the stainkit CLI as a subprocess. We keep one
work-dir per request under /tmp/stk_uploads/<uuid>.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STK_BIN     = os.environ.get("STK_BIN",     "/teamspace/studios/this_studio/stainkit/build/bin/stainkit")
STK_WORKDIR = os.environ.get("STK_WORKDIR", "/tmp/stk_uploads")
PORT        = int(os.environ.get("PORT", "8000"))

os.makedirs(STK_WORKDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI(title="stainkit demo", version="0.1.0")

# CORS: allow any origin so a static frontend hosted on Vercel (or
# anywhere else) can POST to this backend. Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>stainkit — H&amp;E stain normalisation demo</title>
  <style>
    :root { color-scheme: light dark; }
    body { font: 16px/1.45 -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
           max-width: 760px; margin: 40px auto; padding: 0 20px; }
    h1 { font-weight: 600; }
    .panel { display: block; max-width: 100%; margin: 18px 0;
             border-radius: 6px; box-shadow: 0 0 0 1px rgba(127,127,127,.2); }
    .row { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
    .col { flex: 1 1 240px; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           background: rgba(127,127,127,.15); padding: 1px 5px; border-radius: 3px; }
    .muted { color: #888; font-size: 13px; }
    input[type=file] { margin-right: 12px; }
    button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(127,127,127,.3);
             background: rgba(127,127,127,.1); cursor: pointer; }
    pre { background: rgba(127,127,127,.12); padding: 10px; border-radius: 6px;
          overflow-x: auto; font-size: 13px; }
  </style>
</head>
<body>
  <h1>stainkit</h1>
  <p class="muted">GPU-accelerated H&amp;E stain normalisation for digital pathology. Upload any image of a histology slide; the panel below shows <em>input</em> (left), <em>Macenko-normalised</em> (middle), and the <em>Otsu tissue mask</em> (right). Built on the <a href="https://github.com/404Piyush/stainkit">stainkit</a> open-source project.</p>

  <form id="upload">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">Normalise</button>
    <span class="muted" id="status"></span>
  </form>

  <div id="result"></div>

  <p class="muted">Source: <a href="https://github.com/404Piyush/stainkit">github.com/404Piyush/stainkit</a>. Modal demo equivalent: <a href="https://piyushutkarxb--stainkit-demo-fastapi-app.modal.run">piyushutkarxb--stainkit-demo-fastapi-app.modal.run</a>.</p>

<script>
const form  = document.getElementById('upload');
const out   = document.getElementById('result');
const status= document.getElementById('status');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  out.innerHTML = '';
  status.textContent = 'uploading…';
  const f = e.target.elements.file.files[0];
  if (!f) return;
  const t0 = performance.now();
  const formData = new FormData();
  formData.append('file', f);
  try {
    status.textContent = 'running on GPU…';
    const resp = await fetch('/normalize', { method: 'POST', body: formData });
    if (!resp.ok) {
      status.textContent = 'error ' + resp.status;
      out.innerHTML = '<pre>' + (await resp.text()) + '</pre>';
      return;
    }
    const data = await resp.json();
    const dt = Math.round(performance.now() - t0);
    status.textContent = 'done in ' + data.wall_ms + ' ms (round-trip ' + dt + ' ms)';
    const img = b64 => 'data:image/png;base64,' + b64;
    out.innerHTML = `
      <img class="panel" alt="3-panel" src="${img(data.panel_b64)}">
      <div class="row">
        <div class="col"><h3>Normalised</h3><img class="panel" alt="normalised" src="${img(data.normalised_b64)}"></div>
        <div class="col"><h3>Tissue mask</h3><img class="panel" alt="mask" src="${img(data.mask_b64)}"></div>
      </div>
      <details><summary>CLI stdout</summary><pre>${data.cli_log}</pre></details>
    `;
  } catch (err) {
    status.textContent = 'failed: ' + err.message;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "binary": STK_BIN,
        "workdir": STK_WORKDIR,
        "binary_exists": Path(STK_BIN).is_file(),
    }


@app.post("/normalize")
async def normalize(file: UploadFile = File(...)) -> JSONResponse:
    request_id = uuid.uuid4().hex[:8]
    work       = Path(STK_WORKDIR) / request_id
    in_dir     = work / "in"
    out_dir    = work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the upload's extension so stainkit's image detector works.
    suffix = Path(file.filename or "").suffix or ".png"
    in_path = in_dir / f"input{suffix}"
    try:
        with in_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"upload failed: {e}")

    t0 = time.monotonic()
    cmd = [
        STK_BIN,
        "--input",   str(in_dir),
        "--output",  str(out_dir),
        "--target",  "default",
        "--num-streams", "4",
        "--num-images", "1",
        "--pinned",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=504, detail="stainkit timed out")
    wall_ms = int((time.monotonic() - t0) * 1000)

    if proc.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"stainkit exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )

    def find_first(prefix: str, suffix: str) -> Path:
        for p in out_dir.glob(f"{prefix}*{suffix}"):
            return p
        return out_dir.glob(f"{prefix}*{suffix}").__next__()

    try:
        norm = find_first("input_", "normalised.png")
        mask = find_first("input_", "mask.png")
        panel = find_first("input_", "panel.png")
    except StopIteration:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=500, detail="stainkit produced no output files")

    def b64(p: Path) -> str:
        return base64.b64encode(p.read_bytes()).decode("ascii")

    return JSONResponse({
        "request_id":     request_id,
        "wall_ms":        wall_ms,
        "cli_log":        proc.stdout[-2000:],
        "normalised_b64": b64(norm),
        "mask_b64":       b64(mask),
        "panel_b64":      b64(panel),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")