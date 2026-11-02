# Transformer Building Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add differentiable token Embedding, RMSNorm and SiLU for Scala transformer models.

**Architecture:** CPU Float32 operations in an autograd helper retain all saved parents for stale-parameter checks. Embedding consumes immutable Int64 index tensors, validates before narrowing to numscala row kernels, and scatter-adds repeated row gradients. RMSNorm normalizes the last axis with Double accumulations and an analytical pullback. SiLU uses a stable sigmoid and explicit infinite-input limits. Thin nn case classes derive parameter discovery; a Double Module instance treats epsilon as non-parameter configuration.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, numscala, MUnit.

**Spec:** `docs/superpowers/specs/2026-11-02-scala-torch-design.md`, approved M2/M3 components.

## Global Constraints

- Python-free CPU Float32; Int64 token indices stay immutable and nondifferentiable.
- No silent wrapping or truncation of out-of-range token IDs.
- Explicit noGrad/detach snapshots and parameter invalidation semantics remain intact.
- Padding-index behavior, sparse gradients, other normalization families and full model execution remain unimplemented.

## Review Focus

- Repeated token IDs must accumulate gradients, never overwrite.
- Scalar and empty token shapes must preserve output shape and zero gradients.
- RMSNorm input and weight pullbacks must match finite differences, including empty batches and noncontiguous inputs.
- SiLU must avoid overflow at finite extremes and Inf*0 at limiting branches.
- New frozen transforms must retain dependency lineage.

## Task 1

Files: new `autograd/.../TransformerOps.scala`; Variable method entrypoints; new `nn/.../{Embedding,RMSNorm}.scala`; Module[Double] instance; `nn/.../TransformerLayersSuite.scala`.

Interfaces: `Variable.embedding(indices: Tensor[Long])`, `.rmsNorm(weight: Variable, epsilon: Double=1e-6)`, `.silu()`; `Embedding(weight)` with apply(indices), init(vocabulary,width,seed); `RMSNorm(weight,epsilon=1e-6)` with apply(input), init(width,epsilon).

- [ ] Write known embedding outputs/repeated-gradient tests and invalid-ID/scalar/empty tests; finite-difference RMSNorm and SiLU tests; stable extreme/empty/noncontiguous cases; derived module tests and a trainable token lookup integration.
- [ ] Run the missing-API tests and verify RED.
- [ ] Implement numscala gather/scatter-add, stable math and detached first-order pullbacks.
- [ ] Review, fix demonstrated defects, run full suite including actual GPU gate, update evidence, commit.

## Remaining scope

This does not complete the original goal. Safetensors/config/tokenizer loading, attention/RoPE/KV cache, real LLM comparisons, GPU backend/performance, diffusion and complete NUI migration remain required.
