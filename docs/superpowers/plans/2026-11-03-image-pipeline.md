# Native prompt-to-image path for NUI integration

User explicitly authorized overnight autonomous work: advance scala-torch and
start integrating the current local NUI when the components are ready. Immediate
application goal remains Python removal with preserved existing behavior.

1. Add classifier-free guidance with exact shape/finite-scale checks and gradients.
2. Compose native CLIP fast-profile text, Gaussian initial noise, Euler denoising
   and VAE decode into an inference-only API. Accept caller-supplied schedules and
   noise for reproducibility, report completed-step progress and check cooperative
   cancellation before work, between model calls and before/after decode. Validate
   model compatibility and allocation bounds before expensive generation work.
3. Convert raw RGB NCHW image tensors to PNG using Java ImageIO with finite/shape
   checks and create-new output semantics. Preserve caller-owned existing files.
4. Exercise the full public tiny pipeline, guidance, seed/noise reproducibility,
   negative prompts, cancellation/recovery, pixel conversion and PNG readback.
   Run a standalone tiny-model image export and inspect it; random tiny weights
   establish execution, not semantic image quality. Review and commit.

Next application step: JVM worker adapter behind current NUI's protocol boundary.
This slice is not NUI prompt-compatibility, full scheduler-config loading, SDXL,
CUDA diffusion, all plugins, or removal of NUI's Python runtime. Slow tokenizer,
weighted long-prompt chunks and other existing application semantics stay open.

## Stop checkpoint

User requested documentation, commit and termination before NUI integration.
The API and PNG work passes16targeted tests, read-only review found no blockers,
and TinyDiffusionRun completed a32x32 export that was visually inspected as
colored noise from tiny test weights. This is execution evidence, not image
quality or NUI integration. No expanded/full final-tree regression was run.
See root HANDOFF.md for the requested future direction and stop/resume boundary.
