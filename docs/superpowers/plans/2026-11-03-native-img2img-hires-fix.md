# Native img2img Hires Fix Plan

**Goal:** Execute NUI's `img2img_hires_fix` hook in scala-torch using built-in
Lanczos/Nearest pixel resizing (`None` aliases Lanczos) followed by a second pass.

**Spec:** `docs/superpowers/specs/2026-11-03-native-img2img-hires-fix.md`

## Files and interfaces

- Create `models/src/main/scala/scalatorch/models/worker/NativeImg2ImgHires.scala`
  for strict config parsing, target math, nearest resampling, and image tensor
  conversion.
- Test it in `models/src/test/scala/scalatorch/models/worker/NativeImg2ImgHiresSuite.scala`.
- Modify `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`
  to run the optional second pass before final staging/publication.
- Extend `models/src/test/scala/scalatorch/models/NativeDiffusionBackendSuite.scala`
  with tiny model end-to-end cases.
- Extend NUI `engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala` and
  `docs/scala-torch-native-engine.md`; keep NUI on
  `feat/scala-torch-native-engine`.

## Tasks

### Task 1: Config and built-in pixel resizing

- [x] Test NUI config defaults, ratio sizing, explicit target sizing, floor-to-eight,
  denoise zero, invalid values, and unsupported upscaler names.
- [x] Implement `NativeImg2ImgHires.Config`, `target`, Lanczos resize through
  `DiffusionImageResize`, nearest resize, and RGB-to-tensor conversion.
- [x] Run the focused suite and compare nearest mapping against a hand-computed
  2x2-to-4x4 fixture.

### Task 2: Worker second pass

- [x] Add worker tests for deterministic two-pass output and final dimensions,
  second-pass progress, prompt/CFG/sampler/step overrides, denoise zero, and
  unsupported active LoRA/ControlNet interactions.
- [x] Add synthetic nine-channel inpaint routing coverage; the NUI child-JVM test
  exercises the standard four-channel image-source refinement path.
- [x] Add cancellation after the first pass and assert no output publication.
- [x] Parse the hook strictly and refine the first generated image before the
  current transactional staging/publication block.
- [x] Run focused worker tests plus img2img/inpaint and latent Hires regressions
  against the tiny SD1 fixture (10 passed, 2 SDXL tests skipped without SDXL env).

### Task 3: NUI child-JVM path and audit

- [x] Add a `Generator` test submitting the hook with `Lanczos`, verify returned
  dimensions and absence of a Python environment.
- [x] Update NUI branch documentation and the migration inventory with supported
  built-in resamplers and explicit model-upscaler gap.
- [x] Run the NUI integration test (passed, 50.918s) and `git diff --check` in
  both repositories.

