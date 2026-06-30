# Algorithms

This document is the single source of truth for the mathematical
contracts that the pipeline implements. If you change a kernel, you
should update the relevant section here so that the documentation
matches the code.

## Notation

* `I(x, y)` — the input RGB image in `[0, 1]` (gamma-decoded).
* `OD = -log(I)` — optical density. `1e-6` is used as a floor to keep
  the logarithm bounded.
* `H`, `E` — the unit RGB vectors of the hematoxylin and eosin
  stains respectively. Stored as columns of a 3x3 stain matrix `M`.
* `C` — the per-pixel concentration vector. For a 2-channel
  deconvolution `C = (H, E)`, for the 3-channel case `C = (H, E, R)`.
* `T` — the *target* stain matrix; this is the basis the user wants
  the output image to be expressed in.

## Ruifrok-Johnston color deconvolution

Given a 3x3 stain matrix whose columns are the unit RGB vectors of
each stain, the per-pixel concentration vector is the matrix-vector
product

```
C(x, y) = M^-1 . OD(x, y)
```

We use a 3-channel basis (H, E, residual) so the math works even when
one of the channels carries no chemical. The third column is taken
as the cross product of H and E and re-normalised; this keeps the
basis orthonormal to within rounding error.

Implementation: `src/kernels/color_deconvolution.cu` keeps `M^-1` in
CUDA `__constant__` memory and dispatches one thread per pixel.

## Macenko stain normalization

The Macenko (2009) algorithm separates stain normalisation into four
stages:

1. **Deconvolve the input image** into the (H, E) OD pair using
   Ruifrok-Johnston with a *reference* stain matrix.
2. **Project onto the stain plane.** Each OD pair `(c_h, c_e)` becomes
   `(angle, magnitude)` where
   ```
   angle = atan2(c_e, c_h)
   magnitude = sqrt(c_h^2 + c_e^2)
   ```
3. **Estimate the basis.** The 1st and 99th percentile angles are
   taken to be the directions of H and E in the stain plane. They are
   lifted back to 3D unit vectors; the third column of the matrix is
   the cross product.
4. **Reconstruct with the *target* matrix.** Given the estimated
   basis and the user-supplied target concentrations `c_h^*, c_e^*`,
   the new RGB image is
   ```
   I_norm = exp(-T . (c_h^*, c_e^*, c_residual))
   ```

Implementation: `src/kernels/stain_normalization.cu` runs the four
stages. Stage 1 is a fused RGB→OD + deconvolution kernel (no
intermediate buffer). Stage 2 is a per-pixel (angle, magnitude)
kernel. Stage 3 is host-side (cheap percentile picking). Stage 4 is a
per-pixel reconstruction kernel.

## Otsu tissue masking

We compute the BT.709 luminance, build a 256-bin histogram, and
choose the threshold `t*` that maximises the inter-class variance
between background and tissue:

```
t* = argmax_t   w_b(t) * w_f(t) * (mu_b(t) - mu_f(t))^2
```

where `w_b`, `w_f` are the weights and `mu_b`, `mu_f` are the means
of the two classes at threshold `t`.

Implementation: `src/kernels/tissue_mask.cu` runs the histogram and
threshold kernels entirely on the device using shared-memory
privatisation + atomic adds. A morphological opening and closing of
radius `params.otsu_smoothing_radius` cleans the mask.

## Multi-stream execution

The pipeline can run N images concurrently by assigning each one to
a different CUDA stream. The host uses pinned memory for the input
buffer; the kernel launches are `cudaMemcpyAsync` and `Kernel<<<...,s>>>`
on the stream `s`. Stage events (`cudaEvent_t`) record the elapsed
time of each pipeline stage for the per-image benchmark CSV.

## Caveats

* The stain-matrix estimation assumes the input is a *single-stain*
  slide (H&E). Slides with additional stains (e.g. IHC) will produce
  nonsense.
* The Otsu threshold is sensitive to the presence of large uniform
  background regions; the morphological post-processing pass removes
  most artefacts but is not a substitute for a proper background
  detector.
* The CPU reference implementation uses a 2D lifting of the angles
  (a crude `(cos, sin, 0)` basis). This is enough to make the
  reference visually plausible but should not be used for
  scientific work.
