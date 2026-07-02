# Hosting & cold-start behaviour

The live demo at <https://stainkit.404piyush.me/> is a two-tier deployment:

```
Vercel (frontend, free tier, never sleeps)
   |
   |  POST /normalize  (multipart upload)
   v
Lightning AI Studio "stainkit-demo-v2" (Tesla T4 GPU, free tier)
   |
   v
stainkit CLI binary (--target default --num-streams 4 --num-images 1)
```

Lightning AI Studios on the free tier **auto-stop after ~5 minutes of
idle**. There is no paid "always-on" tier without a credit card, so
we use a two-layer keep-alive:

1. **GitHub Actions** — `.github/workflows/keep-alive.yml` runs every 2
   minutes and pings `https://...cloudspaces.litng.ai/healthz`. As long
   as the studio receives a request within its idle window it stays
   running. Run history: <https://github.com/404Piyush/stainkit/actions/workflows/keep-alive.yml>.

2. **Frontend ping** — `app.js` calls `/healthz` every 60 s while the
   demo page is open in a browser. This catches the gap when a user has
   the page loaded but hasn't uploaded yet, and is faster than the
   GitHub Action cadence.

If the studio does fall asleep despite both layers (e.g., nobody has
the page open and the cron is throttled), the frontend retries the
POST up to 6 times with exponential backoff (~60 s total). If the
studio is still asleep after that, the user sees a friendly error
pointing them to this file.

## Starting a stopped studio manually

The studio is named `stainkit-demo-v2` in teamspace
`contronymduh/scalable-media-generation-project`. To wake it:

```python
from lightning_sdk import Studio
s = Studio(name="stainkit-demo-v2")
s.start(machine="T4_SMALL")  # ~30-60 s to come online
```

Then SSH in and start the FastAPI server:

```bash
cd /teamspace/studios/this_studio/stainkit
nohup python3 -m uvicorn lightning_app_unified:app \
  --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
```

## Build on a fresh studio

The studio image does not ship with CUDA toolkit or `nvcc`, so the
stainkit binary has to be rebuilt on first start:

```bash
export PATH=/home/zeus/miniconda3/envs/cloudspace/bin:\
/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:\
/usr/local/cuda/bin:$PATH
cd /teamspace/studios/this_studio/stainkit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTK_BUILD_TESTS=OFF \
  -DSTK_BUILD_PYTHON=OFF
make -C build -j2 stainkit
```

If the conda env doesn't have `nvcc`:

```bash
conda install -y -c nvidia/label/cuda-12.8.1 \
  cuda-nvcc cuda-cudart-dev cuda-cccl
```

## Why not Modal / HuggingFace Spaces / Railway?

- **Modal** — free tier caps at $1/month and freezes the remaining
  $29 in credits until a card is added. The repo's deploy workflow
  originally targeted Modal; the workflow file has been removed.
- **HuggingFace Spaces** — ZeroGPU is free but does not allow
  subprocess execution, so we cannot shell out to the stainkit CLI.
- **Railway / Render / Fly** — free tiers do not include GPUs.

Lightning AI's free tier is the only no-card option that gives us
real CUDA access on a T4. The trade-off is the auto-stop behaviour
documented above.