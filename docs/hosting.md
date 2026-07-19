# Hosting & cold-start behaviour

The live demo at <https://stainkit.404piyush.me/> is a three-tier
deployment where each tier stays in its free tier:

```
Vercel                                          (frontend, free, never sleeps)
   |
   |  POST /api/normalize   (multipart upload, single request)
   v
Deno Deploy                                    (orchestrator, free, no KV cost)
   |  resume → forward → pause → return
   v
JarvisLabs A30 GPU instance                    (paid per-second, paused by default)
   |
   v
stainkit CLI binary (--target default --num-streams 4 --num-images 1)
```

Each request runs the full pipeline in a single HTTP call (`POST
/api/normalize`). The orchestrator resumes the GPU on demand, forwards
the file, pauses the GPU, and returns the result PNGs — typically in
about 0.5 s of GPU work plus a 20-60 s cold start if the instance was
paused.

## Why this stack

* **Vercel** — static frontend, free tier, ~0 ms cold start.
* **Deno Deploy** — serverless KV-backed orchestrator. Free tier has
  generous request limits; we use ~1 request per user action.
* **JarvisLabs** — ₹166/month for 20 GB of paused storage (since we
  pause between every request) and ~₹0.65 of compute per normalize.
  No credit card required; pays from a prepaid balance.

Compared to the earlier Lightning AI Studio / Modal / Hugging Face
attempts, this combination:

* has no auto-stop / wake-up dance — the GPU is paused, not destroyed,
  when idle;
* has no credit-card requirement;
* has no KV cost (the orchestrator's KV is attached to a Deno Deploy
  free-tier database);
* has no upload-size limit beyond Deno Deploy's request body buffer
  (~64 KB without streaming; the frontend caps uploads at 64 KB until
  we wire up streaming).

## Cold-start flow

User opens <https://stainkit.404piyush.me/> and submits an image:

1. Browser `POST`s the file to the orchestrator's `/api/normalize`.
2. The orchestrator `GET users/fetch/{id}` and resumes the instance
   if it isn't already running.
3. Orchestrator `GET` polls the GPU's `/healthz` until it returns 200
   (cold A30 takes 20-40 s).
4. Orchestrator `POST`s the file to the GPU's `/normalize`. The GPU
   does the Macenko work in ~500 ms.
5. Orchestrator `POST misc/pause` to pause the GPU (background, so
   it doesn't block the response).
6. Orchestrator returns the result (`normalised_b64`, `mask_b64`,
   `panel_b64`) to the browser. Frontend decodes and displays.

User reloads the page five minutes later: the GPU is paused; cold
start repeats but only the orchestrator's pause→resume cycle is
involved; the frontend displays progressive status while waiting.

## Self-healing orchestrator

JarvisLabs sometimes rotates instance IDs between pauses (the new
instance has the same startup script but a different `machine_id`).
The orchestrator catches the 404 from `GET users/fetch/{stale_id}`
and falls back to a `users/fetch` listing, picking the most recent
A30 instance with our startup script ID — so manual env-var updates
are not needed for normal operation.

## Triggering a manual wake-up

Cold start is automatic. If something gets stuck (for example, the GPU
instance is destroyed by JarvisLabs and a new one is provisioned):

```bash
gh secret set JL_API_KEY --body "<your-token>"
gh secret set DENO_DEPLOY_TOKEN --body "<your-token>"
gh workflow run ci.yml --repo 404Piyush/stainkit-orchestrator
```

That redeploys the orchestrator with fresh env vars.

## Why not Modal / HuggingFace Spaces / Railway / Render / Fly?

* **Modal** — free tier caps at $1/month and freezes the remaining $29
  in credits until a card is added.
* **HuggingFace Spaces** — ZeroGPU is free but does not allow subprocess
  execution, so we cannot shell out to the stainkit CLI.
* **Railway / Render / Fly** — free tiers do not include GPUs.
* **Lightning AI Studio** — we used this earlier but the 30 GPU-hr/month
  budget was too tight for cold-start latency. The orchestrator
  reference code is kept in commit history for reference but is no
  longer deployed.
