# Native ControlNet for NUI

## Intent

Allow NUI's existing generation request to condition scala-torch inference with
ControlNet, so standard control-guided image generation can run without Python.
NUI continues sending protocol-v1 JSON; model loading and residual injection
belong to the scala-torch worker.

## Current behavior

NUI resolves selected ControlNet files and serializes enabled units, images,
weights, guidance intervals, preprocessors, resize mode and control mode. The
native worker currently treats active `controlnet` as a non-default option and
rejects it. `ConditionalUNet2d` exposes only the final prediction, while its CPU
and CUDA paths execute down and middle blocks internally. ControlNet therefore
needs a model loader plus a typed residual seam at those block boundaries.

## Goals

- Load a standard Diffusers SD 1.x ControlNet directory from safetensors.
- Compute its conditioning image embedding and down/middle residuals using the
  same timestep and text context as the base U-Net.
- Inject residuals at the matching U-Net skip and middle tensors on both CPU and
  CUDA execution paths.
- Apply unit strength and guidance start/end consistently with NUI's request.
- Preserve protocol-v1 and keep all tensor/model types in scala-torch; NUI passes
  paths and plain JSON only.
- Reject unsupported active units with a precise error rather than silently
  generating without control.

## Initial delivery

The initial delivery covered one enabled SD 1.x ControlNet from a Diffusers
directory, a supplied image with the `none` or `invert` preprocessor, the three
NUI control modes, resize modes 0/1/2, strength, and start/end fractions. It
does not claim completion of the broader Python-removal objective. Multiple
units, SDXL ControlNet, model-specific preprocessors, union/LLLite adapters,
ControlNet Hires behavior, and full plugin parity remain later work.

## Approaches

1. **Duplicate the base U-Net implementation for ControlNet.** This keeps
   loading isolated but duplicates block execution and makes CPU/CUDA behavior
   drift likely.
2. **Share the base down/middle block execution and expose typed residuals.**
   This adds a small model seam and lets both CPU and CUDA use the same block
   definitions. Use this approach.
3. **Keep ControlNet in a Python sidecar.** This preserves upstream behavior but
   leaves the requested runtime dependent on Python and is out of scope.

## Constraints and decisions

- Support only the exact Diffusers configuration and tensor names validated by
  the loader; reject unknown fields and unused weights.
- Use safetensors and bounded CPU allocations. Do not introduce a PyTorch,
  Diffusers, or Python runtime dependency into scala-torch.
- Run ControlNet and U-Net on the selected U-Net device. Respect existing
  cancellation and CUDA allocation limits.
- Interpret NUI guidance intervals against normalized denoising progress; a
  unit contributes zero residual outside its interval.
- Balanced mode uses the configured unit weight. The two soft control modes
  use NUI's 0.825 block-decay policy; “ControlNet is more important” injects
  control residuals only into conditional CFG samples.
- The NUI work remains on `feat/scala-torch-native-engine`; no commits are made.

## Acceptance

- Synthetic SD1 ControlNet weights load with strict config/key/shape checks.
- CPU and CUDA tests verify residual count, shapes, scale, interval boundaries,
  CFG conditional masking and cancellation cleanup.
- The NUI child-JVM path sends one real unit without Python and returns a PNG.
- Unsupported image processors or model families fail explicitly.

## Follow-up progress

- Multiple enabled SD1 ControlNet units now pass through per-unit image
  preprocessing, independent guidance intervals and weights, and CPU/CUDA
  residual summation. Multiple CUDA units use host-offloaded ControlNet weights
  and streaming residual summation to reduce persistent and peak VRAM use.
  Txt2img latent Hires and img2img Hires Fix resize each retained preprocessed
  control map directly for every pass's dimensions, matching the Python path
  and avoiding a second interpolation of the base-size result.
  Runtime parity, Hires+ControlNet output quality, and VRAM measurements remain
  unverified.

## Risks

- No ControlNet checkpoint is present in the local NUI model directory, so a
  real-weight acceptance run requires a separate licensed model fixture.
- ControlNet roughly adds another down/middle U-Net pass and more activations;
  a 6 GiB GPU may require sequential weight offload or tiled VAE behavior.
- NUI resize and preprocessor parity must be checked against its Python oracle
  before claiming broad ControlNet compatibility.
