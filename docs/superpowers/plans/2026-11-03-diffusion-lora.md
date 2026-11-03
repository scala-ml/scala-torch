# Diffusion LoRA Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply standard safetensors linear LoRA adapters to native SD1.x and SDXL requests from NUI.

**Architecture:** Add an exception-safe adapter scope in `nn`, load strict LoRA factor pairs against canonical linear-weight aliases from CLIP/U-Net, and apply request-scaled adapters in both CPU and CUDA `Linear` execution. The native worker resolves NUI's `loras` and `lora_paths` request fields into that scope without changing the base pipeline.

**Tech Stack:** Scala 3.3.8, scala-torch `nn`/`models`/`io`, MUnit, SafeTensors, NUI Scala 3 worker.

**Spec:** `docs/superpowers/specs/2026-11-03-diffusion-lora-design.md`

## Global Constraints

- Keep scala-torch changes on `feat/tensor-foundation`; keep NUI changes on `feat/scala-torch-native-engine`.
- Runtime accepts safetensors only and adds no Python/PyTorch dependency.
- Base checkpoint parameters remain immutable; each request scopes its own adapters and scales.
- NUI's UI/shared modules do not acquire tensor or adapter implementation dependencies.
- Unsupported formats and target layers reject explicitly before generation.

## Review Focus

- Incomplete or mismatched factor pairs must fail before model execution.
- Zero/negative adapter scales must be handled as defined (zero skips, finite negative applies).
- Multiple adapters targeting one layer must add rather than replace each other.
- Text encoder and U-Net multipliers must remain independent, including SDXL's second encoder.
- Cancellation/failure must remove thread-local scope state and not contaminate the next request.

---

### Task 1: Request-scoped low-rank execution

**Files:**
- Create: `nn/src/main/scala/scalatorch/nn/LinearAdapterScope.scala`
- Modify: `nn/src/main/scala/scalatorch/nn/Linear.scala`
- Modify: `models/src/main/scala/scalatorch/models/CudaDiffusionRuntime.scala`
- Test: `nn/src/test/scala/scalatorch/nn/LinearAdapterScopeSuite.scala`

**Interfaces:**
- `LinearAdapterScope.withAdapters[A](adapters: Map[Parameter, Vector[(LowRankAdapter, Float)]])(body: => A): A`
- `LinearAdapterScope.forWeight(weight: Parameter): Vector[(LowRankAdapter, Float)]`
- `LinearAdapterScope.parameters: Vector[Parameter]`

- [x] Test CPU linear with two additive scoped adapters, independent weights, nested scope restoration and exception cleanup.
- [x] Verify the CPU test fails because scoped adapters are not yet applied.
- [x] Apply scope adapters in `Linear.apply`; include scoped adapter factors in CUDA runtime uploads and apply them beside any model-attached adapter.
- [x] Run the new suite plus existing `LowRankAdapterSuite`; the new CUDA overlay test passed on GTX 1650 SUPER in resident and host-offloaded modes.

### Task 2: Strict safetensors loader and model aliases

**Files:**
- Create: `models/src/main/scala/scalatorch/models/DiffusionLoRA.scala`
- Modify: `models/src/main/scala/scalatorch/models/PretrainedClip.scala`
- Modify: `models/src/main/scala/scalatorch/models/PretrainedUNet.scala`
- Test: `models/src/test/scala/scalatorch/models/DiffusionLoRASuite.scala`
- Test resources: small generated standard LoRA safetensors fixtures and manifest

**Interfaces:**
- CLIP/U-Net loaders expose canonical Diffusers and Kohya aliases for supported linear weights while preserving the existing `load` return API.
- `DiffusionLoRA.load(path, aliases, multiplier): Map[Parameter, Vector[(LowRankAdapter, Float)]]` validates and loads the adapter's matched factor pairs.

- [x] Write loader tests for canonical, flattened Kohya and PEFT pairs, alpha default/override, unknown target, missing pair, wrong shapes and unsupported factor keys.
- [x] Capture canonical aliases while loading CLIP, both SDXL encoders and U-Net; keep aliases separate by component.
- [x] Implement bounded SafeTensorSet factor loading and strict alias resolution; support only 2D linear factors and recognized metadata.
- [x] Verify loader cases pass; the full models suite passed 194 tests with 66 environment/device-gated skips.

### Task 3: NUI request integration

**Files:**
- Modify: `models/src/main/scala/scalatorch/models/worker/NativeGenerationDefaults.scala`
- Modify: `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`
- Modify: `models/src/test/scala/scalatorch/models/NativeDiffusionBackendSuite.scala`
- Modify: NUI `engine/src/test/scala/sdfx/engine/NativeScalaEngineSuite.scala` only if the existing request serialization needs regression coverage.

**Interfaces:**
- Parse `params.loras` records (`name`, `weight`, optional `unet`); resolve every name through top-level `lora_paths`.
- Text encoders receive `weight`; U-Net receives `unet.getOrElse(weight)`.
- Reject nonempty `manual_loras`, `hires_loras`, dynamic/block/step fields, unresolved paths and unsupported files before first progress event.
- Wrap prompt encoding and denoising in `LinearAdapterScope.withAdapters`.

- [x] Write worker regression using a synthetic adapter against the tiny SD1.5 model: nonzero changes output, zero is a no-op, repeat is deterministic, and a later adapter-free request matches its baseline.
- [x] Implement request/path parsing and scoped adapter lifecycle; ensure each iteration uses the correct independent TE/U-Net scales.
- [x] Run SD1.5 and SDXL worker suites and the full `models/testOnly *` suite.
- [x] Verify NUI `Generator` to native worker LoRA request routing in `NativeGeneratorSuite` on the `feat/scala-torch-native-engine` branch.

### Task 4: Record and commit

**Files:**
- Modify: `docs/research/nui-current-migration.md` or the latest NUI native integration record
- Modify: `docs/superpowers/plans/2026-11-03-diffusion-lora.md`
- Modify: NUI `docs/scala-torch-native-engine.md`

- [x] Record supported key families, exclusions, fixture provenance, CPU/CUDA results and the remaining real-checkpoint visual gate.
- [x] Run `git diff --check` in both repositories and verify each remains on its requested branch.
- [x] Commit tested scala-torch and NUI changes locally; do not push or mark the full migration complete.
