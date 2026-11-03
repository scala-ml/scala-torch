# Native Nine-Channel Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run NUI's nine-channel inpainting U-Nets in the Python-free Scala worker.

**Architecture:** Extend `DiffusionImageSource` and the common Euler image loop so both CPU and CUDA receive channel-concatenated inpaint conditions. The worker prepares masked-image pixels from the fill-prepared image, while the existing mask remains responsible for final overlay.

**Tech Stack:** Scala 3, scala-torch tensors, Java image IO, NUI JSON Lines worker, sbt/munit.

**Spec:** `docs/superpowers/specs/2026-11-03-native-nine-channel-inpaint.md`

## Global Constraints

- Preserve the JSON Lines protocol and NUI Engine API.
- NUI shared/UI modules and assembly must not gain scala-torch or PyTorch dependencies.
- Four-channel inpainting and unmasked generation outputs remain unchanged.
- NUI edits stay on `feat/scala-torch-native-engine`.

## Review Focus

- Wrong input channel order: assert `[sample, mask, masked-image latent]` values in the pipeline test.
- CFG negative branch receives different conditioning: assert both positive and negative predictions see the same inpaint channels.
- Soft mask threshold differs from Diffusers: test values immediately below, equal to, and above `0.5`.
- Text-to-image with a nine-channel model lacks conditions: test full repaint mask and zero masked-image condition.
- Four-channel path regresses: retain existing four-channel inpaint and CUDA image pipeline suites.

---

### Task 1: Pipeline inpaint conditioning

**Files:**
- Modify: `models/src/main/scala/scalatorch/models/EulerImagePipeline.scala`
- Test: `models/src/test/scala/scalatorch/models/ImageToImageSuite.scala`

**Interfaces:** Add `maskedImage: Option[Tensor[Float]] = None` to
`DiffusionImageSource`; pixel data is CPU Float32 NCHW RGB in `[-1,1]`.

- [x] Add a failing test that constructs a 9-channel test U-Net and verifies the
  exact 9-channel input (sample, mask, masked-image latent) reaches prediction.
- [x] Run the focused test and confirm it fails because the pipeline rejects the
  9-channel U-Net or lacks the condition input.
- [x] Accept four-channel and `2*latent+1` U-Net inputs; validate source masks and
  masked-image tensors; derive latent masks by nearest sampling; encode masked
  image with the VAE; concatenate channels before predictor calls.
- [x] Add threshold-boundary and no-mask full-repaint tests; run focused CPU and
  CUDA image-pipeline suites when CUDA is available.

### Task 2: Worker request preparation

**Files:**
- Modify: `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`
- Modify: `models/src/main/scala/scalatorch/models/worker/NativeInpaint.scala`
- Test: `models/src/test/scala/scalatorch/models/NativeDiffusionBackendSuite.scala`

- [x] Add a worker test using a temporary tiny checkpoint whose
  `conv_in.weight` and config are widened from 4 to 9 input channels.
- [x] Run the test with the prior rejection guard and confirm the worker reports
  `nine-channel inpainting U-Nets are not implemented`.
- [x] Build masked-image pixels from the processed input and mask, pass them to
  `DiffusionImageSource`, and remove the nine-channel rejection while validating
  supported channel widths.
- [x] Verify masked generation, output size and preserved outside-mask pixels;
  run the existing worker cancellation checks alongside it.

### Task 3: NUI Generator integration

**Files:**
- Modify: `engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala`
- Modify: `docs/scala-torch-native-engine.md`

- [x] Add a `Generator` integration request with the synthetic widened
  checkpoint and assert output canvas, overlay pixels, and absence of a Python
  directory.
- [x] Run `engine/testOnly *NativeGeneratorSuite` with the worker classpath and
  tiny model fixture; fix the synthetic fixture header after the first run
  revealed fractional JSON offsets where safetensors requires integers.
- [x] Run the integration after worker support and document synthetic fixture
  coverage separately from real-checkpoint image quality.
