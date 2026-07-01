---
title: stainkit — GPU H&E Stain Normalisation
emoji: 🧬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# stainkit — GPU-Accelerated H&E Stain Normalisation

A live demo of the [stainkit](https://github.com/404Piyush/stainkit) GPU pipeline for histopathology image processing. Upload an H&E patch and the pipeline:

1. **Estimates the haematoxylin + eosin stain basis** using Macenko (2009) percentile-based estimation.
2. **Reconstructs the image** using a target basis (default, `he-royal`, or `he-icm`).
3. **Computes an Otsu tissue mask** with morphological post-processing.

## Hardware note

This Space uses the **CPU Basic** tier because HuggingFace no longer
offers free T4 GPUs for Spaces (ZeroGPU does not allow subprocess
calls, which our pipeline needs). The CPU reference implementation in
`src/cpu_reference.cpp` runs the same Macenko algorithm, just slower
on a CPU core than on a CUDA SM.

For a real T4 demo with the same code, see the
[Modal Labs deploy](https://github.com/404Piyush/stainkit/blob/main/modal_app.py)
in the repo — that one runs on Modal's free $30/month GPU credit.

## Architecture

```
Gradio (app.py)
   │
   ▼  subprocess.run(["./build/bin/stainkit", ...])
stainkit CLI  ─────►  CUDA pipeline:
   │                    ├─ ColorDeconvolveRgb (Ruifrok-Johnston 2001)
   │                    ├─ NormaliseStainFull  (Macenko 2009)
   │                    └─ TissueMask         (Otsu + morphology)
   ▼
normalised.png + mask.png + panel.png
```

The Docker image builds the C++/CUDA project from source on top of
`nvidia/cuda:12.2.0-devel-ubuntu22.04`. The Dockerfile is in
[`huggingface/Dockerfile`](https://github.com/404Piyush/stainkit/blob/main/huggingface/Dockerfile)
in the main repo.

## References

* Macenko, M. et al. "A reference image set for H&E stain normalization"
  (2009).
* Ruifrok, A. C. & Johnston, D. A. "Quantification of histochemical
  staining by color deconvolution" (2001).

## Source

https://github.com/404Piyush/stainkit