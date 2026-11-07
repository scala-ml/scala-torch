# Native Latent Hires Design

## Intent

Implement NUI's latent Hires Fix second pass in scala-torch so enabled Hires requests can run through the Python-free native worker. This advances NUI's end-to-end native generation path while keeping model and tensor code in scala-torch's `models` module and the NUI adapter in its `engine` module.

## Goals

- Support all six NUI latent upscaler modes: bilinear, bilinear antialiased, bicubic, bicubic antialiased, nearest and nearest-exact.
- Match NUI/A1111 target sizing, center crop, second-pass step count and denoising-strength rules.
- Keep the base pass in latent space and avoid a VAE decode/encode round trip for latent upscalers.
- Run second-pass conditioning through the same model, sampler, cancellation, publication and error paths as regular native generation.
- Preserve SDXL original-size/crop conditioning across the second pass, with a tiny SDXL integration case so the resized target geometry is covered.
- Verify the behavior through scala-torch model tests and the NUI `Generator`/child-JVM integration on the existing tiny model.

## Non-goals

- Pixel/model upscalers (Lanczos, Nearest, LDSR, Waifu2x) in this change.
- Hires prompt overrides, separate sampler/scheduler/CFG overrides, Hires LoRA profiles, or inpainting-model Hires.
- Claiming normal SD1.5/SDXL quality or full Python removal from this milestone.

## Approaches

1. Decode the first pass, resize pixels, VAE-encode, then run img2img. This is simpler but differs from NUI's latent modes and adds VAE error and memory cost.
2. Return the first pass's denoised latent, resize it with the selected interpolation semantics, then use it as the clean latent source for the second scheduler pass. This matches the NUI latent path and is selected.
3. Implement all image upscalers at the same time. This needs independent model formats/weights and would obscure correctness of the built-in latent path.

## Constraints and behavior

- Hires is enabled only by `hires.enabled=true`; disabled requests retain existing behavior.
- Second-pass dimensions follow NUI `HiresParams.targetSize` and the worker must center-crop latent dimensions to the final target when the requested two-sided resize requires a larger aspect-preserving intermediate.
- `second_pass_steps` defaults to base steps. The active schedule count follows NUI's `hires_steps` calculation from requested steps and `denoising_strength`.
- Only supported latent upscaler names are accepted. Unsupported image upscalers return the existing explicit `unsupported` error and are never silently treated as latent.
- Hires runs for txt2img and unmasked img2img. The img2img first pass returns its denoised latents, avoiding a VAE decode/encode round trip before the same latent second pass. Masked inpainting Hires and nine-channel inpainting-model Hires fail explicitly because their spatial mask conditioning is not carried through latent resizing.
- Hires-specific prompt/sampler/scheduler/CFG overrides, `hires_loras`, active Hires-specific tweak hooks and Hires seed/noise plans fail explicitly; they are never dropped by default stripping.
- Existing batch/iteration seed ordering applies to both passes; second-pass progress identifies `pass=hires` and totals the work accurately.
- Hires prompt overrides, per-pass sampler/scheduler/CFG changes, inpainting and independent Hires LoRA profiles fail explicitly in this milestone.
- No PyTorch or scala-torch model/tensor types are added to NUI shared modules or its SBT dependencies. Any NUI integration change stays on `feat/scala-torch-native-engine`.

## Verification

Unit tests use differential golden tensors produced by PyTorch `torch.nn.functional.interpolate` to pin all six interpolation modes, sizing/cropping and step-strength boundaries. Tiny SD1 and SDXL worker tests prove deterministic two-pass output, requested final dimensions, per-pass progress, cancellation/error cleanup and unchanged disabled-Hires behavior. The NUI `NativeGeneratorSuite` proves the actual `Generator` serialization and child JVM path without creating a Python environment.

## Open implementation choices

Keep image and latent source representations distinct in `EulerImagePipeline`; preserve existing image-source and inpainting validation. Use the smallest new pipeline API that exposes post-denoising latents and accepts a clean latent source with an explicit strength.
