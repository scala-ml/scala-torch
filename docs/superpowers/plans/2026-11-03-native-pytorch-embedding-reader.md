# Native PyTorch Textual-Inversion Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load NUI's common `.pt` textual-inversion tensors in scala-torch without Python or arbitrary pickle execution.

**Architecture:** Add a bounded ZIP/pickle reader for plain tensor containers to scala-torch `io`; adapt its typed result into the existing immutable CLIP embedding path in `models`. Keep the NUI request contract unchanged and retain warnings for rejected legacy files.

**Tech Stack:** Scala 3, JDK ZIP APIs, scala-torch IO/model tests, existing NUI child-JVM integration.

**Spec:** `docs/superpowers/specs/2026-11-03-native-pytorch-embedding-reader.md`

## Global Constraints

- Keep PyTorch/JAR implementation out of NUI; it only passes paths to scala-torch worker.
- Never execute arbitrary Python pickle globals or dynamically load classes.
- Use bounded archive entries, pickle operations, tensor dimensions, and allocations.
- Support only the explicitly specified current ZIP-based tensor subset; reject all other payloads.
- Preserve SD1/SDXL safetensors behavior, loaded-token lifecycle and request warnings.
- Keep NUI changes on `feat/scala-torch-native-engine`.

## Review Focus

- ZIP path traversal, duplicate names, archive bombs, absent/truncated storage, and malformed central directory must reject before large allocation.
- Pickle globals/reduce/build/persistent-ID paths must only resolve the exact tensor reconstruction vocabulary and never invoke arbitrary code.
- Shape/stride/storage offsets must reject negative, overflowing, overlapping or non-contiguous tensor views.
- SD1 and SDXL maps must reject mixed/missing/extra tensor keys and mismatched widths/vector counts without partial pipeline mutation, while allowing only NUI's documented scalar metadata fields.
- Rejected files must warn and leave subsequent ordinary generation identical to baseline.

## Files and Interfaces

- Create `io/src/main/scala/scalatorch/io/PyTorchTensorArchive.scala`: `read(path: Path, limits: PyTorchArchiveLimits = ...): PyTorchTensorArchive` returns a strict value tree containing only maps, sequences, strings, integers and `StoredTensor(dtype, shape, values)`.
- Create `io/src/test/scala/scalatorch/io/PyTorchTensorArchiveSuite.scala`: builds small ZIP/pickle fixtures and checks values, layouts, bounds and rejection.
- Modify `models/src/main/scala/scalatorch/models/ClipTextualInversion.scala`: add `readPyTorch(path, isSdxl, clipLWidth, clipGWidth)` and normalize the four specified NUI payload layouts to `ClipTextualInversion`.
- Modify `models/src/test/scala/scalatorch/models/ClipTextualInversionSuite.scala`: test SD1/SDXL payload normalization and type/shape rejection.
- Modify `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`: route `.pt` through the restricted reader and keep warnings transactional.
- Modify `models/src/test/scala/scalatorch/models/NativeDiffusionBackendSuite.scala`: replace warning-only expectation with real `.pt` activation plus hostile/unsupported payload warning and baseline recovery.
- Modify NUI `engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala` and `docs/scala-torch-native-engine.md`: exercise `.pt` through `Generator` without Python and document exact supported subset.

## Tasks

### Task 1: Restricted ZIP and pickle tensor reader

- [ ] Add tests for a single F32 tensor and a tensor in primitive maps, including shape and exact values.
- [ ] Run `sbt --server 'io/testOnly *PyTorchTensorArchiveSuite'`; confirm the new reader API is absent.
- [ ] Implement ZIP entry allowlisting and resource bounds, then the minimal stack/memo pickle VM for protocols 2/4/5 and the storage/tensor globals needed by plain tensors.
- [ ] Add RED/GREEN cases for F16/BF16, GPU device tags, list/tuple/map containers, duplicate keys, unsupported globals/opcodes, malformed persistent IDs, non-contiguous/out-of-bounds storage, truncated files, and each configured size/count limit.
- [ ] Run `sbt --server 'io/testOnly *PyTorchTensorArchiveSuite'` and `sbt --server 'io/test'`.

### Task 2: NUI embedding layouts

- [ ] Add tests for top-level tensor, `emb_params`, `string_to_param["*"]`, and exact SDXL `clip_l`/`clip_g` map layouts.
- [ ] Run `sbt --server 'models/testOnly *ClipTextualInversionSuite'`; confirm `.pt` decoding is unsupported.
- [ ] Implement normalization using `PyTorchTensorArchive` and existing finite-value, width and vector-count checks.
- [ ] Test wrong encoder families, extra keys, missing tensors, wrong widths, nonfinite values and mismatched SDXL vector counts.
- [ ] Run `sbt --server 'models/testOnly *ClipTextualInversionSuite'`.

### Task 3: Worker lifecycle and NUI integration

- [ ] Add a worker test proving an active SD1 `.pt` embedding changes output identically to the same safetensors vectors; add corresponding tiny SDXL dual-encoder coverage and invalid-file recovery.
- [ ] Run the focused worker suite and verify expected RED before wiring the reader.
- [ ] Route `.pt` files through `readPyTorch`; only install the immutable pipeline after every encoder/tokenizer update validates.
- [ ] Run `sbt --server 'models/testOnly *NativeDiffusionBackendSuite *PretrainedVaeSuite'` with tiny SD1/SDXL fixtures.
- [ ] Add a NUI `Generator` `.pt` embedding integration test using the real Java child worker and confirm the isolated home contains no Python directory.
- [ ] Run `sbt --server 'engine/testOnly *NativeGeneratorSuite'` on `feat/scala-torch-native-engine`.
- [ ] Update docs, then run `git diff --check` in scala-torch and NUI; verify both branch names and statuses.
