# Native Basic Inpainting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support deterministic mask-based img2img with the existing four-channel Diffusers U-Net path used by NUI.

**Architecture:** Extend `DiffusionImageSource` with an optional repaint mask and original RGB image. The image pipeline will nearest-resize the mask to latent size, restore the noised source latent outside the mask after each scheduler step, and composite the decoded pixels through the full-size mask. The worker will load, resize, invert, and blur the mask and will continue to reject unsupported full-res and non-original fill modes.

**Tech Stack:** Scala 3, munit, Java ImageIO, existing CPU/CUDA U-Net and VAE execution.

**Spec:** `docs/superpowers/specs/2026-11-03-native-inpaint-design.md`; NUI reference behavior is in `C:/Users/Mizushima/dev/stablediffusion-nui/python/sdfx/inpaint.py`.

## Global Constraints

- Existing unmasked txt2img and img2img outputs remain unchanged.
- White means repaint; black means preserve; inversion occurs before resize and blur.
- Unmasked final output pixels equal the resized source exactly.
- CUDA and CPU share mask semantics, and no path silently falls back to Python.
- Keep scala-torch on `feat/tensor-foundation`; keep NUI changes on `feat/scala-torch-native-engine`.

## Review Focus

- A fully black/white or malformed mask must not leak or change ownership; test black/white masks and invalid dimensions/ranges in `ImageToImageSuite`.
- Scheduler suffix blending must use each next sigma, including zero at the final step; test deterministic Euler and DPM++ runs.
- Masked img2img must still preserve pixels outside the mask after resize; test worker PNG readback on a synthetic source.
- Unsupported options and nine-channel models must fail before writing/publishing outputs; test full-res and fill-mode rejection.
- Cancellation must still prevent publication and release worker staging; keep the existing cancellation assertion with masked requests.

---

### Task 1: Pipeline latent and pixel mask semantics

**Files:**
- Modify: `models/src/main/scala/scalatorch/models/EulerImagePipeline.scala`
- Test: `models/src/test/scala/scalatorch/models/ImageToImageSuite.scala`

**Interfaces:** Append optional `mask: Option[Tensor[Float]] = None` and `originalImage: Option[Tensor[Float]] = None` to `DiffusionImageSource`. Require both or neither, CPU Float32, shape `(1,1,H,W)` / `(1,3,H,W)`, finite mask values in `[0,1]`, and matching spatial dimensions.

- [x] Add an ImageToImageSuite case with a fixed source/noise and half-black/half-white mask. Assert the masked result is deterministic and final black-mask pixels equal the original exactly.
- [x] Run `sbt "models/testOnly *ImageToImageSuite"`; observed the new API fail compilation before implementation.
- [x] Implement validation, nearest mask downsampling, noised-source blending after every scheduler step, and final pixel-space compositing.
- [x] Run `models/testOnly *NativeInpaintSuite *NativeImg2ImgSuite *ImageToImageSuite *SdxlImagePipelineSuite *CudaImagePipelineSuite`; 10 passed, 0 failed, with CUDA enabled.
- [x] Commit the pipeline milestone (`75a2225`).

### Task 2: Worker mask preprocessing and safe request handling

**Files:**
- Create: `models/src/main/scala/scalatorch/models/worker/NativeInpaint.scala`
- Modify: `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`
- Test: `models/src/test/scala/scalatorch/models/NativeImg2ImgSuite.scala`

**Interfaces:** `NativeInpaint.prepare(source: BufferedImage, mask: BufferedImage, width: Int, height: Int, resizeMode: Int, blurRadius: Int, invert: Boolean, cancelled: () => Boolean): NativeInpaint.Prepared`; `Prepared` contains masked RGB, original RGB, and CPU Float32 mask tensor.

- [x] Change the worker test from expecting active-mask rejection to requiring a successful masked request and exact unmasked PNG pixels.
- [x] Run `sbt "models/testOnly *NativeImg2ImgSuite"`; observed active-mask rejection before implementation.
- [x] Load the mask with existing bounded image I/O, apply inversion, existing NUI resize geometry and three separable box-blur passes approximating Gaussian blur; accept only fill=original and full-res=false for this slice.
- [x] Pass mask/original image into `DiffusionImageSource`; explicitly reject unsupported nine-channel models, full-res crop and non-original fill.
- [x] Run the worker and pipeline suites with CUDA enabled; 10 passed, 0 failed. Worker assertions include denied cancellation leaving no published files.
- [x] Commit the worker milestone (`08183db`).

### Task 3: NUI Generator integration and branch documentation

**Files:**
- Modify: `C:/Users/Mizushima/dev/stablediffusion-nui/engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala`
- Modify: `C:/Users/Mizushima/dev/stablediffusion-nui/docs/scala-torch-native-engine.md`

- [x] Extend the real Generator SDXL test with a local mask; assert source-sized output and exact original pixels outside the mask.
- [x] Run the opt-in Python-free Generator test with saved CUDA settings and the tiny SDXL model; passed on GTX 1650, reported `cuda:0`, and created no Python directory.
- [x] Record tested mask scope, unsupported cases and the dependency boundary in NUI docs; changes stayed on the existing NUI feature branch.
- [x] Commit the NUI integration milestone (`b198ca8`).

## Deferred

Nine-channel U-Net input concatenation, full-resolution crop/paste, fill modes 0/2/3, exact Pillow blur parity, dedicated inpaint checkpoints, and plugin-based inpainting remain open migration requirements.
