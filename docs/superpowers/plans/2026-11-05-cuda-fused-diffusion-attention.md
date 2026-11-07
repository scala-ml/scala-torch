# Fused CUDA diffusion attention implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fuse noncausal CUDA attention for up to 4096 key tokens to reduce global scratch traffic and kernel launches in native diffusion inference.

**Architecture:** Provide an experimental, opt-in shared-memory FP32 attention kernel for supported dense shapes. One CTA computes one batch/head/query row with stable softmax and GQA mapping. Keep the current score/prepare/output kernels as the default and as fallback for longer keys; leave causal attention untouched.

**Tech Stack:** Scala 3, Java 25 FFM CUDA driver, PTX, MUnit, native-only NUI MCP.

**Spec:** `docs/superpowers/specs/2026-11-05-cuda-fused-diffusion-attention-design.md`

## Global Constraints

- Preserve FP32 tensor layouts and CPU-reference tolerance.
- Preserve GQA mapping, max-subtracted softmax and dense NaN propagation.
- Keep shared memory bounded to a fixed 4096-key limit and obey `CudaSession.maxBytes`.
- Keep causal attention and the long-key fallback unchanged.
- Preserve Python-free runtime dependencies and do not commit or overwrite unrelated dirty work.

## Review Focus

- GQA and uneven key/value widths map each query head to the correct K/V head.
- NaN logits propagate to every output component in the affected row.
- Single key and empty/invalid shapes preserve existing validation and output behavior.
- 4096-key boundary fits shared memory; 4097 keys dispatch to the legacy budget-sliced path.
- Value widths above 256 dispatch to legacy kernels rather than leaving output channels unwritten.
- Allocation failure, kernel failure and session close release all tensors.

---

### Task 1: CUDA regression coverage

**Files:**
- Modify: `ffm/src/test/scala/scalatorch/ffm/CudaDenseAttentionSuite.scala`

**Interfaces:** Existing `CudaSession.scaledDotProductAttention` remains unchanged.

- [x] Extend profile assertions to verify default-off legacy dispatch and opt-in `attention_fused_fp32` dispatch on supported key lengths.
- [x] Add a profile-enabled 4097-key case proving dispatch uses the legacy kernels, plus 4096-key and 257-value-channel boundary cases.
- [x] Observe the new-dispatch test fail before registering/dispatching the kernel.

### Task 2: Fused PTX kernel and dispatch

**Files:**
- Modify: `ffm/src/main/resources/scalatorch/transformer.ptx`
- Modify: `ffm/src/main/scala/scalatorch/ffm/CudaSession.scala`
- Test: `ffm/src/test/scala/scalatorch/ffm/CudaDenseAttentionSuite.scala`

**Interfaces:** Register `attention_fused_fp32`; dispatch it only when `SCALATORCH_CUDA_FUSED_ATTENTION=1`, only for dense attention with `ktokens <= 4096`, passing the existing tensor addresses/layout metadata. The kernel processes one query row per 256-thread CTA and leaves the current public method signature intact.

- [x] Add the failing tests and observe both missing-kernel and default-off dispatch failures before implementation.
- [x] Implement the PTX kernel: parallel per-thread key scores, shared-memory max/NaN reduction, max-subtracted `exp2.approx` and sum reduction, then weighted V accumulation.
- [x] Add opt-in dense dispatch without global score scratch for `ktokens <= 4096`; preserve the default path, larger-key fallback and causal dispatch.
- [x] Run the focused CUDA suite with the feature disabled and enabled; both pass 4/4 with CPU-reference parity, NaN propagation, key/value boundary dispatch and allocation cleanup.
- [x] Measure the 20-step packaged NUI result. Attention was slower than the legacy aggregate, so keep fused dispatch opt-in pending further kernel tuning.

### Task 3: Native-only NUI verification

**Files:**
- Modify: `HANDOFF-KIMI.md`
- Modify: NUI `docs/scala-torch-native-engine.md`
- Build: Scala-Torch `models/workerDistribution`

- [x] Rebuild the worker distribution and package native-only NUI with the experimental fused kernel.
- [x] Generate 512x512, 20-step Euler at a fixed seed through the packaged worker; record completion, dimensions, SHA-256 and wall time.
- [x] Capture sampled whole-device VRAM and CUDA profile totals; note that samples can miss peaks and profile counters overlap.
- [x] Rebuild and package again with fused dispatch default-off; packaged 512x512 four-step smoke completed via the legacy score kernels in 70.7s.

### Follow-up: tune fused kernel before default enablement

- [ ] Profile per key length and head width to isolate why the 256-thread shared-memory CTA is slower than the three-kernel baseline.
- [x] Try warp-level shuffle reductions. CUDA parity passes, but the packaged run regressed from 252.8s / 71.453s attention to 265.5s / 82.319s attention; revert this variant.
- [x] Try a multi-query CTA tile that reuses K/V while preserving enough query parallelism and the 4096-key memory bound. Focused CUDA suite passed 5/5 in default, fused-only, multi-only, and combined opt-in modes. Two packaged SD1.5 512x512 Euler 20-step pairs at fixed seed 4295 measured multi-query at 75.217/75.230s per 1,280 calls vs single-query at 71.710/71.794s (4.8-4.9% slower). End-to-end wall time showed no improvement (256.74s single vs 260.85s multi in the second pair). Keep both experimental paths opt-in; output hash matched between multi-query repeats, while single-query hashes differed, so cross-kernel parity is not established.
- [ ] Keep the environment opt-in until repeated 512px 20-step packaged runs show a reproducible wall-time win without higher VRAM or output regressions.
