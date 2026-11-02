# Causal Attention and RoPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add differentiable grouped-query causal attention and Llama-style rotary positions for real model assembly.

**Architecture:** Separate autograd helper for CPU Float32 rank-4 `(batch,heads,tokens,width)` values. Stable Double softmax probabilities and accumulations with Float32 outputs; analytical q/k/v pullbacks accumulate shared KV heads. Causal keys are indexed from zero; queryStart offsets query positions for decode chunks. RoPE pairs contiguous half-vectors (Llama rotate_half convention), using positionStart and theta. All saved parents retain invalidation checks.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, MUnit.

**Spec:** Approved design `docs/superpowers/specs/2026-11-02-scala-torch-design.md`, M3 attention/RoPE components.

## Global Constraints

- Python-free CPU Float32; no implicit device movement.
- Hq must be divisible by Hkv, matching K/V heads and token lengths; Q/K widths match. Value width may differ.
- Only causal masking, no dropout/padding/custom mask in this stage.
- Default maximum score workspace: 16,777,216 elements (Double probabilities), configurable, always Int-indexed.
- Nonempty queries need at least one key. Empty query/batch outputs retain shape and zero input gradients.
- Reference loops are correctness implementation, not FlashAttention/performance parity.

## Review Focus

- Masked future keys/values must never contaminate output or gradients, even when they contain NaN.
- Grouped query heads must share and sum K/V gradients in contiguous head groups.
- Decode queryStart must align to the same absolute causal positions as full prefill.
- RoPE must use half-vector rather than interleaved pairing and its transpose rotation for backward.
- Validate shape/workspace products before allocations or narrowing.

## Task 1

Files: `autograd/.../AttentionOps.scala`, Variable entrypoints, `autograd/.../AttentionSuite.scala`.

Interfaces: `Variable.causalAttention(key,value,queryStart=0,maxScoreElements=16777216L)`, `.rope(positionStart=0,theta=10000.0)`.

- [ ] Write known softmax/causal values, masked-NaN, grouped-head gradient, full/decode equivalence, central-difference q/k/v, rotary forward/inverse-gradient, empty/invalid/workspace and saved-parameter tests.
- [ ] Run missing API RED.
- [ ] Implement stable forward, detached first-order pullbacks and bounded workspace.
- [ ] Review, fix demonstrated defects, run complete suite including actual GPU gate, document scope, commit.

## Next gate

Assemble a real Llama-family model from public config and safetensors, then implement config/tokenizer/KV-cache and numerical reference comparisons. These operations alone do not verify a real LLM.
