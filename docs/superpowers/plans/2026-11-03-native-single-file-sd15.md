# Native SD1 Single-File Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load common SD1 `.safetensors` checkpoints in NUI through the Python-free ScalaTorch worker.

**Architecture:** Add a lazy name-mapping view over a single safetensors archive, then feed its validated tensors into the existing CLIP, U-Net and VAE constructors. Extend worker startup and model discovery to accept a configured checkpoint file while leaving Diffusers-directory behavior unchanged.

**Tech Stack:** Scala 3, existing scala-torch `SafeTensors`, `PretrainedClip`, `PretrainedUNet`, `PretrainedVae`, NUI `EngineClient` protocol-v1 child worker.

**Spec:** `docs/superpowers/specs/2026-11-03-native-single-file-sd15.md`

## Global Constraints

- Keep Python/PyTorch runtime and model conversion code out of scala-torch and NUI's native path.
- Accept only safetensors; never deserialize `.ckpt` pickle content.
- Do not write a full converted checkpoint or hold a second full copy of its weights.
- Reject unsupported family/layout/config before publishing a replacement pipeline.
- Preserve the existing Diffusers-directory route and the Python provider.
- Do not add or run tests in this implementation; compile and real-worker acceptance are the allowed verification gates.

## Review Focus

- Unknown, duplicate, missing and shape-mismatched LDM tensors must fail explicitly before model publication.
- A malformed/truncated archive or a file changed during loading must not leave a partial model active.
- SD2/SDXL/inpaint and EMA-only layouts must be rejected rather than guessed as SD1.
- F16/BF16 source tensors must use the existing bounded Float32 conversion path.
- Loading errors must not silently select Python or overwrite a previously loaded model.

## Files and Interfaces

- Create `models/src/main/scala/scalatorch/models/SingleFileSd1Config.scala` for supported header/family detection and the canonical SD1 scheduler/model config values.
- Create `models/src/main/scala/scalatorch/models/SingleFileTensorView.scala` for a lazy mapping from expected Diffusers names to source LDM safetensors names, including collision and shape checks.
- Modify `models/src/main/scala/scalatorch/models/PretrainedClip.scala`, `PretrainedUNet.scala`, and `PretrainedVae.scala` to accept the validated tensor view without copying the complete archive.
- Modify `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala` to load file-backed SD1 models transactionally and retain directory loading.
- Modify NUI `scalaTorchEngine/.../ScalaTorchLaunch.scala` only as needed to validate a `.safetensors` root and resolve scheduler configuration.
- Update `docs/scala-torch-native-engine.md` with supported layouts and explicit limitations after runtime acceptance.

## Tasks

### Task 1: Safetensors family detection and lazy tensor view

- [ ] Inspect the current `SafeTensors` metadata/read lifecycle and define exact supported key prefixes for common SD1 checkpoints.
- [ ] Add bounded SD1 family/config inspection and the lazy mapped view; reject collisions, missing required prefixes, unsupported input-channel counts and unknown extra tensor families.
- [ ] Compile `io` and `models`; manually inspect error messages for malformed and unsupported header shapes.

### Task 2: Map CLIP, VAE and U-Net parameters

- [ ] Implement the common LDM-to-Diffusers key mapping for SD1 text encoder, VAE and U-Net without materializing a converted state map.
- [ ] Route mapped reads into existing constructors, validating required tensors, shapes and complete consumption before returning the pipeline.
- [ ] Compile `models`; compare the mapped key inventory against the pinned Diffusers 0.33.1 converter source and the local tiny SD1 component topology.

### Task 3: Worker and NUI launch integration

- [ ] Allow a configured `.safetensors` file as `NativeDiffusionBackend` root; retain current directory checks for Diffusers models.
- [ ] Resolve canonical SD1 scheduler defaults when no explicit scheduler is configured, and preserve explicit scheduler overrides.
- [ ] Advertise the file path as the configured native checkpoint and accept the same path in `load_model`.
- [ ] Compile `models` and NUI `scalaTorchEngine`/`engine`/`mcp`/`ui` modules.

### Task 4: Runtime acceptance and documentation

- [ ] Locate or provision a legally usable conventional SD1 safetensors checkpoint; keep it outside tracked source files.
- [ ] Launch the NUI native worker from an isolated app home with no `python/` directory, load the file and generate one low-resolution image through `AppServices`/`Generator` or MCP stdio.
- [ ] Confirm the Python process is absent, the image dimensions and seed are returned, a failed unsupported load preserves the active pipeline, and the Diffusers-directory route still works.
- [ ] Record the exact supported checkpoint layout and runtime evidence in `docs/scala-torch-native-engine.md`.
