# SDXL Text Conditioning and Pipeline Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Do not spawn agents.

**Goal:** Connect the verified SDXL U-Net to native text conditioning, image generation and NUI requests while retaining existing SD1.5 behavior.

**Architecture:** Extend CLIP with raw hidden-state selection and pooled projection without changing its existing normalized apply method. Add a dual-encoder conditioning object, then supply its context and text-time additions to the shared image sampler. Match NUI prompt semantics from its local source, including empty negative prompts and clip skip.

**Tech Stack:** Existing Scala tensors/modules/tokenizer, safetensors, Java25 CUDA runtime, native worker JSON protocol. Python only for independent reference fixtures.

**Spec:** `docs/research/nui-migration-audit.md`; local NUI `python/sdfx/prompt.py` functions `_encode_chunk`, `sdxl_layer_choice`, `encode_sdxl`, `_chunks_for`, `_emphasize`, `pad_to_match`; `python/sdfx/engine.py` SDXL prompt preparation around lines819-838. Independent Transformers4.49.0/Diffusers0.33.1 fixtures from pinned tiny SDXL revisionfa06b5c3d6d45fabd978486616c6dae068519e85.

## Global constraints

- Runtime invokes no Python. Native generation must work with a Python-absent NUI home.
- Preserve SD1.5 defaults and strict config/weight validation. Do not reinterpret its clip_skip or change existing normalized output.
- NUI SDXL clip_skip1/2 uses raw penultimate features; larger values select hidden_states[-clip_skip]. No final norm on these features.
- NUI whitespace-only negative prompt becomes zero context and zero pooled text; do not substitute encoded empty text.
- NUI legacy eos_token_id2 normally pools maximum token ID; when tokenizer EOS exists and IDs exceed it, pool first actual EOS. Modern EOS uses first matching token.
- Base SDXL text-time IDs are original(height,width),crop(top,left),target(height,width); default original/target are requested dimensions and crop0,0. Do not advertise refiner support with its different conditioning schema.
- NUI changes stay on feat/scala-torch-native-engine; no main merge/push. Existing active immutable SD1.5 acceptance process must remain undisturbed.

## Review focus

- Raw selected features must differ from normalized SD1.5 output; test both against reference (Task1).
- Legacy/modern EOS, repeated padding EOS, missing EOS and added token IDs have different pooling behavior (Task1).
- Two tokenizers may produce unequal chunk counts; NUI repeats the last shorter-encoder chunk, then pads positive/negative conditioning with empty chunks (Task2).
- Empty negative prompts must zero both pooled and per-token values, including whitespace-only text (Task2).
- Worker load/set_vae and metadata hashing must include encoder2/tokenizer2 and survive failed model replacement without corrupting loaded state (Task3).

## Task1: CLIP raw states and projected pooling

Files: modify `models/src/main/scala/scalatorch/models/ClipTextConfig.scala`, `ClipTextModel.scala`, `PretrainedClip.scala`; create `models/src/test/scala/scalatorch/models/SdxlClipSuite.scala`; extend `scripts/reference_sdxl.py`.

Interfaces: append defaulted projection dimension/architecture data to ClipTextConfig and optional `textProjection: Option[Linear]`/EOS ID to ClipTextModel. Add `ClipTextOutput(hidden: Variable,finalHidden: Variable,pooled: Variable,projected: Option[Variable])` and `encode(tokens: Tensor[Long],skipLastLayers: Int=1,legacyEosOverride: Option[Long]=None): ClipTextOutput`. Selected hidden is raw; pooled is gathered from the fully executed, final-normalized sequence. Run the stack once and retain only requested hidden state.

- [x] Write failing tests using checked-in text_encoder-penultimate/final and text_encoder_2-penultimate/projected fixtures; require maxabs<2e-5.
- [x] Implement raw state selection, projection-free and WithProjection configs/loaders; text_projection.weight must be exactly(projection_dim,hidden_size), no bias. Preserve strict unused-weight checks and public load return type.
- [x] Add oracle cases for legacy maximum-token pooling, modern first EOS, repeated EOS, added-ID override, no EOS fallback and skip bounds. Validate allocation budgets and trainability/gradient path.
- [x] Run SdxlClipSuite,PretrainedClipSuite and existing ClipTextModel tests; commit when all pass.

## Task2: NUI dual-encoder prompt semantics

Files: create `models/src/main/scala/scalatorch/models/SdxlTextConditioning.scala` and `DiffusionPrompt.scala`; extend `ClipFastTokenizer.scala` only where explicit pad/token config is required; create `SdxlTextConditioningSuite.scala` and `DiffusionPromptSuite.scala`.

Interfaces: `SdxlTextConditioning.encode(prompt: String,negativePrompt: String,clipSkip: Int): SdxlPromptPair`; `SdxlPromptPair` holds positive/negative context Tensor[Float] and positive/negative pooled Tensor[Float]. Constructor owns both tokenizer/model pairs. Prompt parser/chunker returns token IDs and per-token weights; keep tokenizer normalization profile explicit rather than claiming slow-tokenizer ftfy parity.

- [x] Generate independent NUI encode_sdxl fixtures for plain text,empty/whitespace negative,clip_skip1/2/3,weighted spans,escaped delimiters and prompts crossing75-token boundaries.
- [x] Implement NUI chunking/emphasis and raw dual-CLIP concatenation; pooled text comes from encoder2's first chunk before emphasis. Match unequal chunk repeat/padding behavior.
- [x] Test positive/negative lengths, EOS/padding choices and model dimension mismatch. Explicitly track textual-inversion/extension support as remaining work if its weight injection is not yet available; do not silently consume directives with wrong semantics.
- [x] Run the conditioning and tokenizer regression suites with fixtures; commit verified behavior.

Task2 evidence: DiffusionPromptSuite has122 source-derived parser/chunker cases;
SdxlTextConditioningSuite compares44 tensors from11 NUI encode_sdxl cases, including
different encoder chunk counts (encoder2 unmerged BPE) and explicit non-EOS pad.
Maximum absolute error1.431e-6. Combined prompt/tokenizer/CLIP suites passed12tests.
Optional reference_sdxl_prompt.py records model input hashes and NUI source hash.
Conditioner supports base dual encoders only and is not yet connected to worker.
Tokenizer uses documented fast NFC profile; slow-tokenizer ftfy parity remains.
Textual inversion, LoRA/directive processing and embedding injection remain pending:
this class neither strips extension tags nor claims their application. Worker must
route supported extensions or report unsupported ones when integrating Task3.
Tokenizer load/fromText now accept explicit padToken; default EOS behavior remains.
Worker still needs to read model tokenizer_config padding during SDXL loading.

## Task3: Shared generation pipeline and worker loading

Files: modify `EulerImagePipeline.scala`, `worker/NativeDiffusionBackend.scala`; add `SdxlImagePipelineSuite.scala` and `NativeSdxlBackendSuite.scala`. Keep shared VAE encode/decode, scheduler and staging/recovery paths.

Interfaces: append optional SDXL conditioning to pipeline construction; keep existing constructor calls valid. Supply positive/negative DiffusionAddedConditioning to UNetExecution.withPredictor. Worker selects supported model family from explicit model_index/config, loads both text/tokenizer trees and includes them in checkpoint hash. Use an explicit sufficient maxParameters budget for full SDXL U-Net; retain checked limits.

- [ ] Add fixed-noise tiny SDXL Euler/DPM txt2img and img2img oracle tests, CPU/CUDA offload, CFG on/off, negative-empty/nonempty and seed replay.
- [ ] Integrate text/size conditioning into shared sampler; preserve one latent scaling and img2img active schedule semantics. Validate dimensions before heavy work.
- [ ] Test failed load/set_vae retains published state, correct model family/hash, cancellation/staging/recovery, and strict unsupported variants.
- [ ] Run SD1.5 image/worker regressions plus SDXL suites; commit.

## Task4: NUI and real-model acceptance

- [ ] Extend NUI opt-in NativeScalaEngineSuite/NativeGeneratorSuite to actual tiny SDXL typed requests and PNG save/replay on the dedicated branch.
- [ ] Provision hash-verified normal SDXL base weights without redistributing them; use immutable runtime snapshots, real txt2img/img2img, inspect final images, and record memory/time limits honestly.
- [ ] Preserve remaining goal requirements: full GUI/MCP transports, standard checkpoint formats, inpainting/extensions, and packaged Python-absent execution. SDXL base success alone is not overall completion.

## Rulings from source inspection

The earlier U-Net plan mentioned generic empty-negative semantics. NUI explicitly zeros negative conditioning when neg.strip()=="", while Diffusers only does so for negative_prompt=None. Follow NUI at its public worker boundary; oracle full-pipeline calls must supply matching embeddings or None deliberately. Tiny legacy CLIP uses eos_token_id2, so default Transformers pooling and NUI's actual-EOS override need separate fixtures rather than treating either as universally correct.
