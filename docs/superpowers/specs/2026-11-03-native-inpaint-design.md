# Native basic inpainting design

> This document records the initial four-channel scope. Nine-channel support is
> specified in [native nine-channel inpainting](2026-11-03-native-nine-channel-inpaint.md).

## Intent

Add mask-aware img2img to the Python-free Scala worker so NUI can repaint selected
regions while preserving the source image outside the mask.

## Goals

- Accept NUI's existing mask path, inversion, blur and fill fields for img2img.
- Implement the common four-channel U-Net path: noise the source latents and blend
  preserved regions back after every denoising step.
- Composite the final decoded image through the processed mask so unmasked pixels
  remain exactly equal to the resized source.
- Reject unsupported nine-channel inpaint U-Nets and full-resolution crop mode
  explicitly; never reinterpret them as ordinary img2img.

## Non-goals

- Nine-channel inpainting checkpoints, full-resolution masked crop, ADetailer,
  ControlNet inpaint, hires inpaint, soft-inpainting hooks and all fill algorithms.
- Byte-identical parity with Pillow's Gaussian blur; mask values and geometry must
  be deterministic, and exact preservation outside the processed mask is required.

## Approaches

1. Final-pixel compositing only: simple, but the denoising trajectory ignores the
   mask and produces weak boundaries. Rejected.
2. Four-channel latent blending plus final-pixel compositing: follows the common
   Diffusers four-channel path and preserves exact outside-mask pixels. Chosen as
   the first native inpaint path; nine-channel models remain an explicit error.
3. Nine-channel U-Net conditioning: closest to dedicated inpaint checkpoints but
   requires a second masked-image latent input and batch-aware mask forwarding.
   Deferred as a later capability rather than silently approximated.

## Design

Worker preprocessing reads the RGB source and grayscale mask, applies inversion,
the existing resize geometry and three bounded separable box-blur passes that
approximate a Gaussian blur. Initial scope
supports the default original-pixel fill and rejects other fill modes and masked
full-resolution crop with an actionable unsupported error. The pipeline accepts
an optional white-means-repaint mask and original image alongside the existing
img2img source. It downsamples the mask to latent resolution with nearest-neighbor
sampling, restores the noised source outside the mask after each scheduler step,
then blends the decoded output with the resized original at pixel resolution.

## Constraints

- Keep the existing txt2img/img2img outputs unchanged when no mask is supplied.
- CPU and CUDA use the same mask semantics; existing U-Net/VAE device ownership
  and cancellation checks remain in force.
- Validate mask dimensions, finite values, path and options before generation.
- Keep changes on the current scala-torch feature branch and NUI work on its
  existing `feat/scala-torch-native-engine` branch.

## Open for later work

Nine-channel U-Net concatenation, full-res crop/paste, fill modes 0/2/3, mask
edge parity with Pillow, standard inpaint checkpoint acceptance, and NUI GUI
inpaint interaction remain required before claiming complete NUI parity.
