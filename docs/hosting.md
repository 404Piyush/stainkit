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

Lightning AI Studios on the free tier give **30 GPU-hours/month** and
**auto-stop after ~5 minutes of idle**. There is no paid "always-on"
tier without a credit card.

To stay under the 30-hour budget, the backend is **not** kept warm by a
cron job. Instead:

1. **Frontend ping** — `app.js` calls `/healthz` every 60 s while the
   demo page is open in a browser. As long as someone has the page
   loaded, the studio stays warm. Zero compute used when nobody is
   looking.
2. **Frontend retry** — `fetchWithWakeup()` retries the POST up to 6
   times with exponential backoff (~60 s total) on cold start, so the
   user sees a "warming up" message instead of an error.
3. **Manual wake** — `.github/workflows/keep-alive.yml` can be
   triggered from the Actions tab if the studio is fully stopped and
   nobody can be on the page to ping it.

## Cold-start flow

User opens <https://stainkit.404piyush.me/>:

1. Page loads, immediately pings `/healthz`. If studio is asleep, the
   ping fails silently.
2. Page renders the upload form. User picks an image and clicks Run.
3. POST `/normalize` fails. Frontend retries 6 times (~60 s) with
   "backend is waking up" message.
4. If still failing after 60 s, error message explains the user can
   either keep the page open for 60 s and retry, or trigger the
   `keep-alive` workflow manually at
   <https://github.com/404Piyush/stainkit/actions/workflows/keep-alive.yml>.

## Waking a stopped studio manually

Two ways to restart the backend:

### Option 1 — GitHub Actions UI

Go to <https://github.com/404Piyush/stainkit/actions/workflows/keep-alive.yml>,
click "Run workflow". The workflow pings `/healthz` 3 times — the first
ping triggers Lightning to start the studio (takes ~60 s) and the
subsequent pings confirm it is healthy.

### Option 2 — Lightning CLI

```python
from lightning_sdk import Studio
s = Studio(name="stainkit-demo-v2")
s.start(machine="T4_SMALL")  # ~30-60 s
```

Then SSH into the studio and start the FastAPI server:

```bash
cd /teamspace/studios/this_studio/stainkit
nohup python3 -m uvicorn lightning_app_unified:app \
  --host 0.0.0.0 --port 8000 > /tmp/uvicorn.log 2>&1 &
```

## Why the GitHub Action is manual-only

A scheduled ping every 5 minutes costs ~3 minutes of compute per hour,
~72 minutes/day, ~36 hours/month — blowing past the 30-hour free
budget by itself. The frontend ping is free because it only runs
while the page is open, and only one user (the one trying the demo)
pays for that compute. As soon as the user closes the tab, the studio
auto-stops within 5 minutes.

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