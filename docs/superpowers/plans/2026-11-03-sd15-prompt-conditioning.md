# SD1.5 NUI Prompt Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make native SD1.5 prompt embeddings follow NUI's standard single-CLIP parsing, weighting, chunking, clip-skip and sequence-padding semantics.

**Architecture:** Extract common CLIP chunk encoding, mean-preserving emphasis and sequence concatenation/padding from `SdxlTextConditioning`. Add a standard SD1.5 single-encoder conditioner and route the non-SDXL branch of `EulerImagePipeline` through it; keep SDXL's dual-encoder and blank-negative behavior unchanged.

**Tech Stack:** Scala 3.3.8, scala-torch models, MUnit, pinned local tiny SD1.5 Diffusers model, optional Python/PyTorch oracle generation for test fixtures only.

**Spec:** `docs/superpowers/specs/2026-11-03-sd15-prompt-conditioning-design.md`

## Global Constraints

- Keep the scala-torch branch `feat/tensor-foundation`; NUI-specific sources remain on NUI's `feat/scala-torch-native-engine`.
- Runtime adds no Python or PyTorch dependency.
- Standard SD1.5 clip-skip 1 uses the final normalized CLIP output; skip N>1 selects the Nth hidden state from the end and applies final LayerNorm.
- Blank SD1.5 negative prompts are encoded normally; only SDXL retains its existing zero-negative behavior.
- Positive and negative sequences are padded to equal chunk count with empty-prompt embeddings.
- Emphasis mean correction uses a shared dtype-relative threshold (8 machine epsilons of mean absolute magnitude); below it, avoid unstable rescaling.
- Emphasis mean correction uses a shared dtype-relative threshold (8 machine epsilons of mean absolute magnitude); below it, avoid unstable rescaling.
- With guidance scale at or below one, the SD1.5 path does not encode the negative prompt.
- Fixture parity tolerance is maximum absolute error `< 3e-5`.

## Review Focus

- Blank negative prompt: SD1.5 must encode it, then align chunk lengths; cover in the oracle suite.
- Unequal positive/negative chunk counts: pad with encoded empty prompt chunks; cover long-positive and long-negative fixtures.
- Explicit `truncatePrompt`: retain only one weighted 77-token chunk; cover a long prompt fixture.
- Clip skip at supported boundaries: correct layer selection and LayerNorm, reject beyond depth; cover skip-2/skip-3 and invalid skip assertions.
- Configured tokenizer PAD differing from EOS: replace post-EOS positions without changing semantic tokens; cover via a local tokenizer variant where supported by the tiny vocabulary.

---

### Task 1: Generate independent SD1.5 prompt fixtures and define failing behavior

**Files:**
- Create: `scripts/reference_sd15_prompt.py`
- Create: `models/src/test/resources/tiny-sd15-prompt/manifest.json` and little-endian tensor fixtures
- Create: `models/src/test/scala/scalatorch/models/SingleClipTextConditioningSuite.scala`

**Interfaces:**
- Fixture script consumes local NUI `python/sdfx/prompt.py`, the pinned tiny SD1.5 Diffusers root, and an output directory.
- The suite expects `SingleClipTextConditioning(tokenizer, encoder, maxElements)` and `encode(prompt, negativePrompt, clipSkip, truncatePrompt, cancelled)` returning positive and negative tensors.

- [x] Add the oracle script for plain, clip-skip 2/3, weighted syntax, long positive/negative `BREAK`, unequal chunk counts, and explicit truncation. Record NUI source, model revision, tokenizer/model input hashes and Transformers/Torch versions in the manifest.
- [x] Generate fixture tensors using the isolated reference venv and local tiny model; confirm each binary hash and declared shape.
- [x] Write the MUnit suite to compare positive/negative shapes and values at `< 3e-5`, assert equal sequence length, reject invalid clip-skip/allocation, and check cancellation between chunks.
- [x] Run `models/testOnly scalatorch.models.SingleClipTextConditioningSuite`; expected RED was observed before implementation.

### Task 2: Share CLIP prompt mechanics and implement single-encoder conditioning

**Files:**
- Create: `models/src/main/scala/scalatorch/models/ClipPromptEncoding.scala`
- Create: `models/src/main/scala/scalatorch/models/SingleClipTextConditioning.scala`
- Modify: `models/src/main/scala/scalatorch/models/SdxlTextConditioning.scala`
- Test: `models/src/test/scala/scalatorch/models/SingleClipTextConditioningSuite.scala`
- Test: `models/src/test/scala/scalatorch/models/SdxlTextConditioningSuite.scala`

**Interfaces:**
- `ClipPromptEncoding.chunks(tokenizer, text)` returns fixed 77-token `PromptChunk`s from `DiffusionPrompt`.
- `ClipPromptEncoding.emphasize(hidden, weights)` applies NUI's token multipliers and mean correction.
- `ClipPromptEncoding.weightedChunks(tokenizer, encoder, text, skipLastLayers, cancelled)` returns each chunk's `ClipTextOutput` with its token weights applied to the selected hidden state; pooled output remains available to SDXL.
- `ClipPromptEncoding.concatenate(chunks, maxElements)` and `padTo(input, chunkCount, emptyChunk, maxElements)` build checked context tensors.
- `SingleClipTextConditioning.encode(..., encodeNegative = true)` returns `SingleClipPromptPair(positive, negative: Option[Tensor[Float]])`; when negative is encoded, both sequences have aligned lengths.

- [x] Run the new single-encoder suite and record the expected missing-class RED.
- [x] Implement shared chunk tokenization, emphasis, checked chunk concatenation and empty-chunk padding; move SDXL to those helpers without changing its dual-encoder pooling or blank-negative logic.
- [x] Implement SD1.5 layer selection: clip-skip 1 uses normalized final output; N>1 uses `encoder.encode(..., skipLastLayers=N-1).hidden` followed by final LayerNorm. Apply `DiffusionPrompt` weights and align positive/negative chunk counts with an encoded empty prompt. `encodeNegative=false` must not parse or encode negative text.
- [x] Preserve `truncatePrompt=true` as the first weighted chunk while default false keeps all chunks.
- [x] Run both conditioning suites; all oracle tensors and existing SDXL fixtures pass below `3e-5`.

### Task 3: Connect the shared image pipeline and verify NUI worker behavior

**Files:**
- Modify: `models/src/main/scala/scalatorch/models/EulerImagePipeline.scala`
- Test: `models/src/test/scala/scalatorch/models/EulerImagePipelineSuite.scala` or `NativeDiffusionBackendSuite.scala` (use the existing suite that owns tiny-model worker setup)
- Verify: `models/src/test/scala/scalatorch/models/SdxlImagePipelineSuite.scala`, `NativeDiffusionBackendSuite.scala`

**Interfaces:**
- Non-SDXL generation uses `SingleClipTextConditioning`; SDXL generation continues to use `SdxlTextConditioning`.
- Request `truncatePrompt` remains honored. CFG disabled must avoid encoding the negative prompt.

- [x] Add a worker regression showing weighted/chunked standard SD1.5 prompts reach the U-Net with aligned contexts and deterministic seed replay.
- [x] Run targeted SD1.5 worker and SDXL pipeline/worker regressions with the pinned tiny model.
- [x] Run all `models` tests (257 total: 205 passed, 52 resource/device-gated skipped, 0 failed) and inspect the skips.

### Task 4: Record scope and commit

**Files:**
- Modify: `docs/research/native-img2img.md` or the current prompt implementation status document
- Modify: `docs/superpowers/plans/2026-11-03-sd15-prompt-conditioning.md`

- [x] Record tested NUI semantics, fixture provenance, numerical tolerance and remaining slow-tokenizer/textual-inversion gaps.
- [x] Run `git diff --check`; inspect the final diff and verify NUI edits stay on `feat/scala-torch-native-engine`.
- [ ] Commit the tested scala-torch implementation and fixture assets on `feat/tensor-foundation`.

## Execution Notes

- Task 2 changes both encoders to share implementation; oracle fixtures are the guard against regressions.
- Keep source/model weights out of committed fixtures; commit only bounded prompt embedding tensors and provenance metadata.
- Remaining non-goals from the spec stay open in the NUI migration inventory.
