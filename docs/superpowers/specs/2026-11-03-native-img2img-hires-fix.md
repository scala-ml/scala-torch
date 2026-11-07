# Native img2img Hires Fix

## Intent

Run NUI's separate `img2img_hires_fix` hook through the native scala-torch
worker, so a normal img2img or inpaint request can upscale its completed first
image and refine it in a second diffusion pass without Python.

## Goals

- Support the built-in `Lanczos` and `Nearest` pixel upscalers plus NUI's
  `None`-means-Lanczos alias at exact configured target dimensions.
- Match NUI's ratio/dimension rounding, step, denoise, CFG, sampler, scheduler,
  and prompt override defaults for this hook.
- Feed the upscaled image to the existing image-source path; preserve the
  worker's cancellation, progress, recovery, and atomic publication behavior.
- Support standard four-channel and nine-channel U-Nets, including masked
  inpaint requests whose first result is then refined across the whole image.
- Keep all scala-torch references in the NUI `engine` adapter and its child JVM
  classpath; do not add model/tensor jars to NUI's SBT dependency graph.

## Non-goals

- ESRGAN, LDSR, Waifu2x, Spandrel, or other model-backed upscalers.
- Hires LoRA switching, ControlNet reprocessing, and unsupported hook
  interactions during the refinement pass. These must fail explicitly when
  active instead of being silently ignored.
- Changing the Python implementation or making Python optional for other
  unsupported NUI requests.

## Approaches

1. Reuse the existing txt2img latent Hires implementation. Rejected because the
   NUI plugin is a distinct pixel-upscale-then-img2img operation and has
   different configuration and metadata semantics.
2. Decode the first generated image, apply NUI's built-in pixel resampler, and
   run a second image-source pass through the existing pipeline. Selected; it
   matches the plugin's operation and reuses validated model/sampler paths.
3. Add neural upscaler model loading in the same change. Rejected for this
   increment because those model formats and execution paths do not yet exist
   in scala-torch.

## Constraints and behavior

- The plugin is active only when `hooks.img2img_hires_fix` is an object. NUI's
  defaults are ratio 2, `R-ESRGAN 4x+`, and 0.33 denoise; prompts/steps/CFG/
  sampler/scheduler inherit from the base request. Since scala-torch does not
  yet load neural upscaler models, users must select `Lanczos` or `Nearest`;
  the unchanged NUI default fails explicitly as unsupported.
- When both explicit width and height are positive, use them; otherwise use
  `floor(inputDimension * ratio)` independently, then floor each to a multiple
  of eight with a 64-pixel minimum. Reject values over 16384 per side or the
  worker image allocation limit.
- `denoise=0` returns the resized image without a second diffusion pass.
- `steps=0`, `cfg=0`, blank prompt overrides, and `Use same ...` values inherit
  their base request values, matching NUI's `effective` helper.
- Only `Lanczos`, `Nearest`, and `None` are accepted as pixel upscalers in this
  increment. Names beginning with `Latent` and model-backed names return an
  explicit unsupported error.
- Use the same per-image seed for the refinement. The second pass reports
  `pass=img2img_hires`; only its final image is published.
- Keep inpaint output overlay disabled during refinement; a fully white latent
  mask is used for nine-channel U-Nets.
- Active unsupported LoRA/ControlNet refinement combinations fail before
  publishing output. Cancellation or failure between passes removes staging
  files.

## Verification

- Unit tests pin config validation, target sizing, and nearest-neighbor pixel
  mapping.
- Tiny worker tests cover deterministic Lanczos and Nearest refinement,
  denoise-zero behavior, overrides, masked inpaint/nine-channel routing,
  unsupported upscalers, cancellation, and final dimensions.
- NUI `Generator` integration submits the actual hook through its child-JVM
  boundary and verifies output dimensions without creating a Python runtime.

