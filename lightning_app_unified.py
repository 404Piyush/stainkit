"""Unified stainkit demo: serves the static frontend + FastAPI /normalize
endpoint from a single Lightning Studio port.

When a request hits `/`, the server returns the index HTML. When it
hits `/normalize`, the server shells out to the stainkit CLI on the
GPU. When it hits `/healthz`, it returns a JSON status.

This avoids the Vercel + CORS + bot-protection dance. One URL, one
service, one place to monitor. The Lightning Studio auto-sleeps
after `auto_sleep_time` seconds of inactivity, so we only pay for
GPU time when an actual request is in flight.
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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STK_BIN     = os.environ.get("STK_BIN",     "/teamspace/studios/this_studio/stainkit/build/bin/stainkit")
STK_WORKDIR = os.environ.get("STK_WORKDIR", "/tmp/stk_uploads")
PORT        = int(os.environ.get("PORT", "8000"))
STATIC_DIR  = Path(os.environ.get("STATIC_DIR", "/teamspace/studios/this_studio/stainkit/lightning_static"))

os.makedirs(STK_WORKDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI(title="stainkit demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve /styles.css, /app.js, /config.js from disk if the static
# directory exists. Falls through to the index handler otherwise so
# the demo still works in a "backend-only" deployment.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "binary": STK_BIN,
        "workdir": STK_WORKDIR,
        "static_dir": str(STATIC_DIR),
        "binary_exists": Path(STK_BIN).is_file(),
    }


# Inline HTML in case the static files aren't deployed. This is the
# version that runs standalone (Option A: one-URL deployment).
INDEX_HTML_INLINE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>stainkit — H&amp;E stain normalisation</title>
  <style>
    :root { color-scheme: light dark; --fg:#111; --bg:#fafafa; --muted:#6a737d; --border:rgba(127,127,127,.25); --panel:rgba(127,127,127,.08); --accent:#6f42c1; }
    @media (prefers-color-scheme: dark) {
      :root { --fg:#eee; --bg:#0d1117; --muted:#8b949e; --border:rgba(255,255,255,.12); --panel:rgba(255,255,255,.05); }
    }
    * { box-sizing: border-box; }
    body { font: 16px/1.5 -apple-system, BlinkMacSystemFont, system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0; }
    header, main, footer { max-width: 880px; margin: 0 auto; padding: 20px 24px; }
    header { padding-top: 40px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
    h1 { margin: 0 0 6px; font-size: 32px; font-weight: 600; }
    h2 { margin: 0 0 12px; font-size: 20px; font-weight: 600; }
    .tagline { margin: 0 0 8px; font-size: 18px; }
    .muted { color: var(--muted); font-size: 14px; }
    main section { padding: 24px 0; border-bottom: 1px solid var(--border); }
    input[type=file] { font: inherit; margin-right: 12px; padding: 6px 10px;
      border: 1px solid var(--border); border-radius: 6px; background: var(--panel); }
    button { font: inherit; padding: 8px 18px; border-radius: 6px;
      border: 1px solid var(--border); background: var(--accent);
      color: white; cursor: pointer; }
    button:disabled { opacity: .5; cursor: wait; }
    img.panel { display: block; max-width: 100%; height: auto; margin: 16px 0;
      border-radius: 6px; border: 1px solid var(--border); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
    @media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
    .row figure { margin: 0; }
    figcaption { font-size: 13px; color: var(--muted); margin-top: 6px; text-align: center; }
    pre { background: var(--panel); padding: 12px; border-radius: 6px;
      overflow-x: auto; font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
      border: 1px solid var(--border); }
    details summary { cursor: pointer; color: var(--muted); margin-top: 12px; }
    #status { margin-left: 12px; }
    #status.busy::before { content: ""; display: inline-block; width: 10px;
      height: 10px; border-radius: 50%; background: var(--accent);
      margin-right: 8px; vertical-align: -1px; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: .35; } 100% { opacity: 1; } }
    footer { padding-bottom: 60px; color: var(--muted); font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <h1>stainkit</h1>
    <p class="tagline">GPU-accelerated H&amp;E stain normalisation for digital pathology.</p>
    <p class="muted">Source on <a href="https://github.com/404Piyush/stainkit">GitHub</a>.</p>
  </header>
  <main>
    <section>
      <h2>Upload an image</h2>
      <p class="muted">PNG, JPEG, BMP or TGA. The Lightning Studio boots a Tesla T4 GPU on first request; subsequent requests are warm.</p>
      <form id="upload">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Normalise on GPU</button>
        <span class="muted" id="status">idle</span>
      </form>
    </section>
    <section id="result-section" hidden>
      <h2>Result</h2>
      <div id="result"></div>
      <details><summary>CLI stdout</summary><pre id="cli-log"></pre></details>
    </section>
    <section id="error-section" hidden>
      <h2>Error</h2><pre id="error-log"></pre>
    </section>
  </main>
  <footer>Hosted on a <a href="https://lightning.ai">Lightning AI Studio</a> · T4 GPU · auto-sleep when idle.</footer>
  <script>
  const form = document.getElementById('upload');
  const out = document.getElementById('result');
  const status = document.getElementById('status');
  const errSection = document.getElementById('error-section');
  const errLog = document.getElementById('error-log');
  const cliLog = document.getElementById('cli-log');
  const resultSection = document.getElementById('result-section');

  function setStatus(text, busy = false) {
    status.textContent = text;
    status.classList.toggle('busy', busy);
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errSection.hidden = true;
    const f = e.target.elements.file.files[0];
    if (!f) return;
    const btn = form.querySelector('button');
    btn.disabled = true;
    setStatus('uploading…', true);
    const fd = new FormData();
    fd.append('file', f);
    try {
      setStatus('warming up GPU (cold start ≈ 3s)…', true);
      const r = await fetch('/normalize', { method: 'POST', body: fd });
      if (!r.ok) {
        const t = await r.text();
        throw new Error('HTTP ' + r.status + ': ' + t.slice(0, 800));
      }
      const data = await r.json();
      const img = b64 => 'data:image/png;base64,' + b64;
      out.innerHTML = `
        <img class="panel" alt="3-panel comparison" src="${img(data.panel_b64)}">
        <div class="row">
          <figure><img class="panel" alt="Normalised" src="${img(data.normalised_b64)}">
            <figcaption>Macenko stain normalisation</figcaption></figure>
          <figure><img class="panel" alt="Tissue mask" src="${img(data.mask_b64)}">
            <figcaption>Tissue mask (white = tissue)</figcaption></figure>
        </div>
        <p class="muted">Server wall-clock: <strong>${data.wall_ms} ms</strong>.</p>`;
      cliLog.textContent = data.cli_log || '(no cli output)';
      resultSection.hidden = false;
      setStatus('done in ' + data.wall_ms + ' ms');
    } catch (err) {
      errSection.hidden = false;
      errLog.textContent = String(err);
      setStatus('failed');
    } finally {
      btn.disabled = false;
    }
  });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Prefer the on-disk version (allows iteration without redeploying
    # the server) but fall back to the inline copy.
    if STATIC_DIR.is_dir():
        idx = STATIC_DIR / "index.html"
        if idx.is_file():
            return idx.read_text(encoding="utf-8")
    return INDEX_HTML_INLINE


@app.post("/normalize")
async def normalize(file: UploadFile = File(...)) -> JSONResponse:
    request_id = uuid.uuid4().hex[:8]
    work       = Path(STK_WORKDIR) / request_id
    in_dir     = work / "in"
    out_dir    = work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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