# CUDA dense attention FP32 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce noncausal attention cost in scala-torch's CUDA inference path while preserving stable softmax and NUI behavior.

**Architecture:** Add dedicated FP32 score, softmax and output kernels. Dispatch only from `scaledDotProductAttention`; retain the FP64 causal kernel unchanged and reduce dense scratch from two Float32 values to one per score.

**Tech Stack:** Scala 3, Java 25 FFM CUDA driver, PTX, MUnit.

**Spec:** `docs/superpowers/specs/2026-11-04-cuda-dense-attention-fp32-design.md`

## Global Constraints

- Preserve causal attention behavior and the public `scaledDotProductAttention` signature.
- Preserve NaN propagation, GQA mapping, max-subtracted softmax, cancellation-free allocation cleanup, and session memory accounting.
- Keep FP32 tensors, the `sm_50` target, Java 25 FFM and Python-free runtime dependencies.
- Do not commit or overwrite unrelated dirty work.

## Review Focus

- Large logits and NaNs: stable max-subtracted softmax and NaN propagation remain correct.
- Unequal query/key/value widths and GQA head mapping: compare with CPU reference.
- Tight workspace budgets and empty output: allocation behavior remains bounded.
- Causal path: keep existing kernel and tests unchanged.
- Actual image inference: full-checkpoint NUI run verifies the optimized path.

---

### Task 1: FP32 dense attention kernels

**Files:**
- Modify: `ffm/src/main/resources/scalatorch/transformer.ptx`
- Modify: `ffm/src/main/scala/scalatorch/ffm/CudaSession.scala`
- Modify: `ffm/src/test/scala/scalatorch/ffm/CudaDenseAttentionSuite.scala`

- [ ] Extend the existing GQA dense-attention test to capture profile output and require `attention_scores_fp32`, `attention_prepare_fp32`, and `attention_output_fp32` when profiling is enabled.
- [ ] Run `ffm/testOnly scalatorch.ffm.CudaDenseAttentionSuite` on CUDA with profiling enabled; observe the new-dispatch assertion fail against the existing FP64 kernels.
- [ ] Implement the three dedicated FP32 PTX kernels, keeping max-subtracted softmax and explicit NaN propagation.
- [ ] Register and dispatch the FP32 entries for dense attention only; allocate one FP32 value per score, update workspace and launch sizing, and pass a Float32 scale.
- [ ] Run focused dense and causal CUDA attention suites and the relevant FFM compile.

### Task 2: NUI full-model measurement

**Files:**
- Modify: NUI `engine/src/test/scala/sdfx/engine/NativeRealCheckpointSuite.scala`
- Modify: NUI `docs/scala-torch-native-engine.md`
- Modify: scala-torch `HANDOFF-KIMI.md`

- [ ] Rebuild `models/workerDistribution`.
- [ ] Run SD1.5 512x512/four Heun steps through NUI EngineClient with CUDA placement, weight offload and profiling enabled.
- [ ] Compare dense attention kernel totals and full runtime with the prior 148.369s profile run; confirm worker handshake and PNG dimensions.
- [ ] Record exact results and any numerical/performance limitations in the NUI and Kimi handoff docs.
