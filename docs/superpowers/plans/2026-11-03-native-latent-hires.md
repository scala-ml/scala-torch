# Native Latent Hires Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run NUI latent Hires Fix as a native two-pass generation request.

**Architecture:** `EulerImagePipeline` exposes the sampled latent and accepts a latent source for img2img-strength scheduling. `NativeHires` owns NUI-compatible latent interpolation, size math and step math. `NativeDiffusionBackend` composes two passes and publishes only the final image through the current transaction path.

**Tech Stack:** Scala 3, scala-torch tensors, existing sbt model tests, NUI Scala `Generator` integration test.

**Spec:** `docs/superpowers/specs/2026-11-03-native-latent-hires-design.md`

## Global Constraints

- Keep model/tensor implementation in scala-torch `models`; do not add model jars or types to NUI shared modules or SBT dependencies.
- Keep NUI changes on `feat/scala-torch-native-engine`.
- Preserve disabled Hires behavior, existing output publication/cancellation, and explicit errors for unsupported settings.
- Implement only the six latent modes named in the spec; never silently reinterpret another upscaler as latent.

## Review Focus

- Fractional target sizes and two-sided aspect crop must match Python's integer truncation and floor-to-eight behavior; pin raw/target/intermediate dimensions and latent crop offsets aligned to eight image pixels.
- Denoising strength 0 and 1 have special schedule boundaries; pin active step count, not only nominal steps.
- A latent source must bypass image VAE encoding, while existing img2img/inpainting image sources retain validation and behavior.
- Cancellation between passes must not publish partial files; prove staging cleanup.
- Invalid/unsupported Hires overrides, active `hires_loras`, Hires tweak hooks, img2img Hires and nine-channel inpainting-model Hires must fail before expensive model work and keep the request error explicit.
- SDXL Hires must retain original-size/crop time IDs while changing target size; pin with tiny SDXL worker coverage.

## Files and Interfaces

- Create `models/src/main/scala/scalatorch/models/worker/NativeHires.scala` for strict request parsing, `targetSize`, `upscaleSize`, `steps`, latent resize and crop.
- Test it in `models/src/test/scala/scalatorch/models/worker/NativeHiresSuite.scala`, using checked-in differential golden arrays captured from PyTorch's `torch.nn.functional.interpolate` for all six modes, including antialias downsampling.
- Modify `models/src/main/scala/scalatorch/models/EulerImagePipeline.scala` with `DiffusionLatentSource`, `generateLatents`/shared latent-return path, and source initialization from already encoded diffusion latents.
- Test the pipeline API in the existing pinned-model `models/src/test/scala/scalatorch/models/PretrainedVaeSuite.scala`, alongside SDXL conditioning regression coverage in `SdxlImagePipelineSuite.scala` where applicable.
- Modify `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala` to parse Hires, execute base/second schedules per image and publish final outputs.
- Extend `models/src/test/scala/scalatorch/models/NativeDiffusionBackendSuite.scala` with enabled-Hires fixture requests and negative cases.
- On NUI branch `feat/scala-torch-native-engine`, extend `engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala` and `docs/scala-torch-native-engine.md` with the end-to-end proof.

## Tasks

### Task 1: NUI-compatible Hires math and latent interpolation

- [x] Write failing tests for all six named interpolation modes against checked-in PyTorch-generated golden values (up/downsample, antialias and nearest coordinate ties), output shape, crop alignment, target/intermediate sizing, and `hires_steps` boundary cases.
- [x] Run `sbt --server 'models/testOnly *NativeHiresSuite'`; expected RED was missing `NativeHires`.
- [x] Implement strict Hires math and interpolation in `NativeHires` with dimension bounds and explicit unsupported-mode errors.
- [x] Run the focused suite: 6 tests passed against local PyTorch 2.7.1 golden values.

### Task 2: Latent output and latent-source sampling API

- [x] Write a failing pipeline test in `PretrainedVaeSuite` that gets denoised latents from a pass and uses them as the clean latent source for a strength-limited second schedule while the VAE encoder is configured to fail; assert final decoded image shape and finite values.
- [x] Run the focused pipeline suite and confirm the missing API is the failure.
- [x] Implement `DiffusionLatentSource` and a latent-return method while retaining existing image-source checks, inpaint composition and VAE recovery behavior.
- [x] Run the focused pipeline suite plus existing img2img/inpainting pipeline tests (included in the 33-test combined run).

### Task 3: Native two-pass Hires worker flow

- [x] Update `NativeGenerationDefaultsSuite` first so enabled Hires reaches strict worker parsing while Hires-specific overrides are not silently stripped.
- [x] Add failing worker tests for deterministic two-pass Hires, final target dimensions, all six modes, SDXL original-size/crop retention, separate base/hires progress accounting, disabled-Hires regression, unsupported upscaler/override rejection, img2img/nine-channel rejection, and cancellation after base pass but before Hires pass publication.
- [x] Run the focused backend suite and confirm enabled Hires fails before implementation.
- [x] Implement per-image base latent generation, resize/crop, second-pass denoising with strength schedule, decode final output, and transactional publication.
- [x] Run focused backend tests and pipeline regressions: 33 tests passed across the combined targeted suites. The full `models` suite remains unrun.

### Task 4: NUI child-JVM integration

- [x] Add NUI `Generator` integration coverage submitting latent Hires for tiny SD1 through `Generator`; assert final dimensions, metadata and one published final image with no Python directory. Tiny SDXL routing is covered by the scala-torch worker suite.
- [x] Run the opt-in integration test and confirm the worker currently rejects Hires.
- [x] Implement only NUI test/docs changes needed to expose and document the accepted request contract; keep product changes in scala-torch worker.
- [x] Run `engine/testOnly *NativeHiresGeneratorSuite` with the tiny SD1 checkpoint (1 passed, 38.058s) and `git diff --check` in both repositories.

## Follow-up: unmasked img2img latent Hires

- [x] Extend latent-return pipeline calls to accept the ordinary image source, so the first img2img pass can return its denoised latents without decoding.
- [x] Route unmasked img2img Hires through the existing latent resize and second-pass path; keep masked/inpainting requests explicitly unsupported.
- [x] Direct Java worker CUDA smoke on the pinned tiny SD model: 64x64 img2img, two requested base steps at 0.5 strength (one active step), 96x96 latent Hires, two second-pass steps, one final PNG (2.35s).
- [x] NUI `NativeHiresGeneratorSuite` through `AppServices`/`Generator`: txt2img Hires then unmasked img2img Hires, retained request settings, final 128x128 PNG, no Python environment (75.849s).
- [ ] Run targeted pipeline/worker suites before claiming broader pipeline coverage; NUI Generator and direct CUDA worker paths are verified separately.

