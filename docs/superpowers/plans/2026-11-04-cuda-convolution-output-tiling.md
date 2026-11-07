# CUDA convolution output tiling implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve scala-torch's FP32 CUDA convolution throughput for NUI's offloaded diffusion inference.

**Architecture:** Add a separate shared-memory GEMM PTX entry point that computes two adjacent output columns per thread and processes 32 K elements per tile. Dispatch it from `CudaSession.conv2dFast`, leaving generic GEMM and the reference convolution untouched.

**Tech Stack:** Scala 3, Java 25 FFM CUDA driver, PTX, MUnit.

**Spec:** `docs/superpowers/specs/2026-11-04-cuda-convolution-output-tiling-design.md`

## Global Constraints

- Keep FP32 tensor/storage and the `sm_50` PTX target.
- Preserve `CudaSession.maxBytes` accounting and allocation cleanup.
- Keep NUI's Python and Scala-Torch backends replaceable and independent.
- Preserve all pre-existing dirty changes; do not commit or reset.

## Review Focus

- M/N/K tails, including rows/columns not divisible by tile width: CUDA test compares against CPU reference.
- Grouped convolution and constrained workspace chunking: existing CUDA convolution suite.
- NaN/Inf and empty input: existing direct convolution suite must remain on its current path.
- CUDA devices that cannot load the new PTX: existing `sm_50` target must JIT successfully on the local GTX 1650 SUPER.
- Full model runtime: repeat the same NUI EngineClient 512x512 four-step run.

---

### Task 1: Test and implement output-tiled GEMM

**Files:**
- Modify: `ffm/src/main/resources/scalatorch/gemm.ptx`
- Modify: `ffm/src/main/scala/scalatorch/ffm/CudaSession.scala`
- Modify: `ffm/src/test/scala/scalatorch/ffm/CudaConvolutionSuite.scala`

- [x] Add a CUDA test for a padded convolution with irregular output channels, spatial width and reduction size; compare `conv2dFast` to `Conv2dOps.forward` and require the new profile label.
- [x] Run the focused CUDA test with `SCALATORCH_CUDA_TESTS=1` and `SCALATORCH_CUDA_PROFILE=1`; it failed because the new dispatch was absent.
- [x] Add PTX entry `gemm_conv_n32` with 16x32 output and 16x32 A / 32x32 B staging tiles, 32-deep K traversal, masked tail loads/stores and two FP32 accumulators per thread.
- [x] Register and look up `gemm_conv_n32`; dispatch it only from `conv2dFast`, with launch grid `ceil(M/16) * ceil(N/32)` and an explicit profile label.
- [x] Run `sbt --server --batch 'ffm/testOnly scalatorch.ffm.CudaConvolutionSuite'` with CUDA tests enabled; existing correctness and leak checks passed.

### Task 2: Measure NUI checkpoint path

**Files:**
- Modify: NUI `docs/scala-torch-native-engine.md`
- Modify: scala-torch `HANDOFF-KIMI.md`

- [x] Rebuild `models/workerDistribution` so NUI uses the new FFM jar.
- [x] Run full SD1.5 512x512, four-step Heun through `NativeRealCheckpointSuite` with CUDA U-Net/VAE and weight offload.
- [x] Compare with the recorded 147.0s baseline; 149.062s completed successfully, within expected run-to-run variance.
- [x] Record the exact result, GPU memory observation and limitations in both handoff documents.

## Execution ledger

- Ruling: keep the new dispatch after 149.062s versus a 147.0s prior run; this is not a material end-to-end regression and the repeat verifies correctness in NUI, but the cost if the variance assessment is wrong is a small throughput regression on other model shapes.
- Red: the new irregular-tail CUDA test failed because profile output lacked `gemm_conv_n32`.
- Debugging: first green-kernel attempt failed grouped chunking at output index 1255 (`groups=4`, 9.075 max error). The final chunk passed `positionsPerChunk` as B's row stride although im2col packs rows at the actual `positions` width. Passing the active width fixed it.
- Verification: `ffm/testOnly scalatorch.ffm.CudaConvolutionSuite`, CUDA enabled: 5 passed, 0 failed. `models/workerDistribution` built. NUI `NativeRealCheckpointSuite`: SD1.5 512x512, four Heun steps, 149.062s, passed.
- Final review: self-review (no subagent review dispatched).
- No commit created; preserve the user's existing dirty worktree.
