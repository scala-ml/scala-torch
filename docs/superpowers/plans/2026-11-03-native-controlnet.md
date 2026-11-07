# Native ControlNet Implementation Plan

> **For agentic workers:** Implement inline in the existing worktree. Keep all current dirty files. Do not commit.

**Goal:** Apply standard Diffusers SD 1.x ControlNet units through NUI's native worker request.

**Architecture:** Share existing U-Net down/middle block definitions and add a typed residual interface consumed by CPU and CUDA prediction. Load ControlNet weights in scala-torch `models`, decode/resize the request image in the worker, and keep NUI communication on protocol-v1 JSON.

**Tech Stack:** Scala 3, scala-torch tensor/autograd/CUDA runtime, safetensors, NUI JSON Lines worker.

**Spec:** `docs/superpowers/specs/2026-11-03-native-controlnet-design.md`

## Global Constraints

- No Python, PyTorch, or Diffusers runtime dependency.
- NUI and scala-torch remain separate classpaths; only paths and JSON cross the worker boundary.
- Keep existing CPU/CUDA execution, cancellation, memory budget, and protocol behavior intact.
- Keep NUI modifications on `feat/scala-torch-native-engine`; do not commit.

## Review Focus

- Residual topology must match the base U-Net's exact skip order and shape.
- CFG conditional-only residuals must align with the predictor's positive/negative ordering.
- Guidance start/end boundaries must be deterministic for one-step and multi-step schedules.
- Control image resize/invert must preserve NUI's mode semantics and channel normalization.
- CUDA failure/cancellation must close both ControlNet and U-Net resources without masking the original error.

---

### Task 1: Typed residual seam in U-Net execution

**Files:** Modify `models/src/main/scala/scalatorch/models/ConditionalUNet2d.scala`, `UNetExecution.scala`, `CudaDiffusionRuntime.scala`, and `EulerImagePipeline.scala`.

**Interfaces:** Define a typed `ControlNetResiduals` carrying ordered down residuals and one middle residual. Both predictor paths consume the same per-step control result.

- [x] Define ControlNet residual values in the base U-Net's input/downsample/resnet skip order and represent positive/negative CFG residuals separately.
- [x] Validate CPU residual count, shape, dtype and device before addition; CUDA validates matching skip/middle shapes on upload.
- [x] Add residuals to CPU and CUDA U-Net skips and middle activation.
- [x] Compile `models`; existing call sites retain the no-ControlNet default path.

### Task 2: Standard Diffusers SD1 ControlNet loader and compute

**Files:** Create `ControlNetConfig.scala`, `ConditionalControlNet.scala`, and `PretrainedControlNet.scala`; modify `CudaDiffusionRuntime.scala` for CUDA execution.

**Interfaces:** `PretrainedControlNet.load(directory, ...)` returns a frozen model that computes `ControlNetResiduals` from latent sample, timestep, context and RGB control image.

- [x] Parse and strictly validate supported ControlNet configuration fields, including omitted up-block types and image-stem settings.
- [x] Load conditioning embedding, down/middle blocks and zero-convolution weights from safetensors with parameter/attention budgets.
- [x] Implement CPU residual computation using shared diffusion blocks.
- [ ] Implement CUDA residual computation in the selected U-Net session, including cleanup on failure.
- [ ] Compile `models` and `worker`.

### Task 3: Native worker request integration

**Files:** Modify `models/src/main/scala/scalatorch/models/worker/NativeGenerationDefaults.scala` and `NativeDiffusionBackend.scala`; create a focused native ControlNet request helper.

**Interfaces:** Parse enabled units' model/image/preprocessor/weight/start/end/control mode and resolve each model through `controlnet_paths`.

- [x] Validate enabled SD1 units (bounded to 16) and fail clearly for unsupported active fields.
- [ ] Decode the image and implement `none`/`invert` plus resize modes 0/1/2.
- [x] Apply independent start/end intervals, per-block soft weighting and conditional-only CFG policy; sum unit residuals.
- [ ] Include ControlNet files in model identity/hash handling and close resources per request.
- [ ] Compile the full worker distribution.

### Task 4: NUI child-JVM verification and migration record

**Files:** Modify the current NUI ScalaTorch adapter only if request serialization is incomplete; update `docs/scala-torch-native-engine.md` and migration inventory.

- [ ] Confirm the existing Generator request includes resolved ControlNet path and image path without backend classes.
- [ ] Run targeted synthetic CPU/CUDA verification and a real-model NUI child-JVM request when a licensed fixture is available.
- [ ] Inspect returned image and logs; record exact model, dimensions, device, time, and known limitations.
- [ ] Keep the overall Python-free parity objective open until all remaining requirements are accepted.
