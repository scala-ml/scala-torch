# Native SD1 Single-File Checkpoint Support

## Outcome

Allow the NUI ScalaTorch backend to load a conventional Stable Diffusion 1.x
`.safetensors` checkpoint directly, without Python or an intermediate Diffusers
directory. This is one migration increment; it does not redefine the full NUI
Python-removal target.

## Scope

- Accept a single-file safetensors state dictionary using the common LDM/A1111
  key layout for a 4-channel SD1.x base model.
- Load the U-Net, VAE and CLIP weights into the existing scala-torch model
  implementations, with strict key and shape validation.
- Keep the selected model path and all model-specific loading in the
  `scalaTorchEngine` child-worker boundary. The NUI core, UI and Python provider
  continue using their current interfaces.
- Preserve Diffusers-directory loading without changing its defaults.
- Use a canonical SD1 scheduler configuration when the checkpoint has no
  scheduler file, while continuing to honor an explicit scheduler override.

## Explicitly unsupported

- `.ckpt` and other pickle-based weight containers. No arbitrary pickle
  globals or Python deserialization are permitted.
- SD2, SDXL, inpainting, instruct-pix2pix, EMA selection, custom CLIP variants,
  quantized weights and checkpoint-specific YAML overrides in this first slice.
- Automatic network downloads during model loading.

Unsupported and contradictory checkpoint layouts must fail before replacing
the active pipeline. They must never trigger Python fallback.

## Architecture

Read safetensors metadata and payloads through scala-torch's bounded
`SafeTensors` reader. Map supported LDM names to the existing Diffusers-style
parameter names lazily; do not materialize a second complete state dictionary
or write a converted checkpoint. Detect only the supported SD1 family from
required tensor names and shapes, and reject extra model families or ambiguous
metadata. Load transactionally so a failed file leaves the currently loaded
pipeline usable.

The NUI launch adapter must permit a regular `.safetensors` file as its configured
model root and resolve the scheduler default for that model form. The worker
catalog reports the configured file as the one loadable checkpoint. No change to
the Python provider or NUI UI dependency graph is part of this feature.

The implementation should follow the pinned Diffusers conversion behavior for
common LDM SD1 checkpoints, adapting its key mapping rather than importing any
Python code: [Diffusers 0.33.1 single-file conversion source](https://raw.githubusercontent.com/huggingface/diffusers/v0.33.1/src/diffusers/loaders/single_file_utils.py).

## Acceptance

- `models/compile` succeeds with no Python dependency added to scala-torch.
- A supported SD1 safetensors checkpoint loads through the real NUI
  `AppServices`/`Generator` worker route in an app home with no `python/`
  directory and produces a valid txt2img image.
- The same checkpoint path is discoverable from NUI's configured checkpoint
  directory; unsupported files fail with a specific native error.
- Existing Diffusers-directory generation and Python-backend selection remain
  available.
