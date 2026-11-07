# Native nine-channel inpainting design

## Intent

Run NUI masked img2img with Diffusers inpainting U-Nets whose input width is
`2 * latentChannels + 1` (9 for the standard four-channel VAE), without Python.

## Design

- Keep one `EulerImagePipeline` for CPU and CUDA U-Net execution.
- Extend `DiffusionImageSource` with optional RGB `maskedImage` pixels. The
  existing `mask` remains the repaint mask and final overlay mask.
- For nine-channel U-Nets, encode `maskedImage` with the VAE and concatenate
  `[schedulerInput, latentMask, maskedImageLatents]` on channel dimension before
  each positive and negative U-Net prediction. Do not apply the four-channel
  latent-preservation blend; retain the existing final pixel overlay.
- For four-channel U-Nets, preserve the current behavior and reject an
  unexpected `maskedImage` value.
- A nine-channel U-Net with no explicit mask uses an all-white repaint mask and
  zero masked-image latent, matching the inpaint-pipeline conditioning used for
  text-to-image and unmasked img2img requests. Final overlay still occurs only
  when a source mask was supplied.
- The worker derives `maskedImage` from its fill-prepared model image, setting
  pixels to zero wherever the processed mask is greater than or equal to 0.5, matching the
  Diffusers inpaint pipeline's binary mask conditioning.
- Accept only U-Net input channels equal to VAE latent channels or
  `2 * latentChannels + 1`; output channels must still equal latent channels.

## Validation and limits

- Unit tests assert exact channel order and mask thresholding using a synthetic
  widened tiny checkpoint; actual nine-channel trained weights are not present
  in the local fixtures.
- Worker tests exercise load, masked img2img, output dimensions, final overlay,
  cancellation, and both CPU/CUDA placement where CUDA is available.
- NUI `Generator` integration runs with the synthetic checkpoint and verifies
  the same JSON Lines boundary and Python-free app home.
- This establishes architecture and request-path support, not image quality or
  parity with a dedicated inpainting checkpoint. Full-resolution crops and mask
  blur remain at their existing documented parity level.

## Constraints

- Keep NUI's engine API and JSON Lines protocol unchanged.
- Do not add scala-torch types or jars to NUI shared/UI modules or assembly.
- Keep all NUI edits on `feat/scala-torch-native-engine`.
