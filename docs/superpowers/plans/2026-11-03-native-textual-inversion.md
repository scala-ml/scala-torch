# Native Textual Inversion Implementation Plan

> **Required subskill:** `superpowers:executing-plans` (inline execution is authorized by the ongoing request to continue).

**Goal:** Use safetensors textual-inversion embeddings in the Python-free NUI worker.

**Architecture:** Add tokenizer token IDs and CLIP token rows by constructing extended immutable CLIP models. Parse selected files and prepare every encoder update before swapping the worker's loaded pipeline. Keep all tensor/model handling inside scala-torch `models`; the NUI engine continues sending paths over its existing protocol.

**Spec:** `docs/superpowers/specs/2026-11-03-native-textual-inversion.md`

## Review focus

- SDXL must update the matching token and row in both encoders.
- Multi-vector tokens must expand in order in positive and negative prompts.
- Invalid later files must not leave earlier encoder/tokenizer changes installed.
- `embedding_paths` must be read from the worker envelope, not generation params.
- `.pt` must warn explicitly instead of silently disappearing or invoking Python.

### Task 1: Extend CLIP tokenization and models

**Files:** `ClipFastTokenizer.scala`, `ClipTextModel.scala`, `SdxlTextConditioning.scala`, CLIP tests.

- [x] Add failing tests for tokenizer registration, multi-vector IDs, collision checks, and encoder row append/validation.
- [x] Run the focused tests and verify expected missing API/behavior failures.
- [x] Implement immutable added-token tokenizer copies and extended CLIP token embeddings.
- [x] Run tokenizer/model tests.

### Task 2: Safetensors parsing and worker lifecycle

**Files:** new `ClipTextualInversion.scala`, `NativeDiffusionBackend.scala`, worker tests.

- [x] Add worker coverage for SD1.x/SDXL files, active/inactive filtering, malformed files, `.pt` warning and cache/reset behavior. The initial parser/API cases were verified red; cache and worker-boundary cases were added as regression tests after the first implementation.
- [x] Run focused tests; the active/inactive, malformed, cache and unload cases pass on tiny SD1/SDXL fixtures.
- [x] Implement bounded safetensors reads, per-file transactional pipeline replacement, token caching and warnings in generation results.
- [x] Run worker/model suites with tiny SD1 and SDXL fixtures; broader cancellation regression remains open.

### Task 3: NUI Generator integration

**Files:** `engine/src/test/scala/sdfx/engine/NativeGeneratorSuite.scala`, `docs/scala-torch-native-engine.md`.

- [x] Add a temporary safetensors embedding to the NUI catalog and exercise `Generator` with its token.
- [x] Run `engine/testOnly *NativeGeneratorSuite` with the tiny model and scala-torch worker classpath.
- [x] Document safetensors coverage and `.pt` limitation separately.

### Task 4: Audit and verify

- [x] Update the current migration inventory to distinguish safetensors inference support from remaining `.pt` and editor/merge work.
- [x] Run focused Scala worker/model suites with tiny SD1 and SDXL CPU fixtures; run NUI Generator integration on CPU SD1 and CUDA SDXL. CUDA case passed on GTX 1650 SUPER (4 GiB).
- [x] Run `git diff --check` in both repositories.
