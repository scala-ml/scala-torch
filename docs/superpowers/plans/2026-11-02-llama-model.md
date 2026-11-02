# Llama Model Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a published Llama-family checkpoint in Scala and compare its logits against an independent runtime.

**Architecture:** A models module composes Embedding, RMSNorm, q/k/v/o Linear projections, causal GQA/RoPE and gated SiLU MLP residual blocks. Public config parsing uses bounded strict Jackson with exact integer fields and explicit rejection of unsupported architectural variants. A single-file safetensors loader checks all names/shapes and handles tied embeddings. Differentiable forward returns all logits; inference logits and greedy ID generation use noGrad. Model, output and attention workspace limits bound allocations.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, existing nn/tensorio. Rust Candle 0.9.1 generates development reference data only.

**Spec:** Approved design `docs/superpowers/specs/2026-11-02-scala-torch-design.md`, partial M3.

## Global Constraints

- No Python dependency for implementation, tests or inference.
- Support standard Llama with SiLU, default unscaled half-pair RoPE, no projection biases/dropout/sliding window. Reject unsupported variants.
- Config limits, tied shape/name consistency and immutable weight ownership must hold.
- The public tiny-random checkpoint proves architecture interoperability, not useful language generation.
- Tokenizer, sharding and persistent KV-cache remain separate required work; greedy IDs initially recompute full prefixes.

## Review Focus

- Projection head reshapes must preserve batch/token ordering.
- HF tensor names and tied lm_head omission must map correctly.
- Unsupported config cannot silently change model behavior.
- Public logits comparison must use fixed revision and recorded file hashes, not self-generated expected outputs.
- Differentiable forward must keep parameter registration and gradients intact while inference uses noGrad.

## Task 1

Files: models `LlamaConfig.scala`, `LlamaModel.scala`, test `LlamaSuite.scala`; optional public-checkpoint test/reference fixtures and generator.

Interfaces: LlamaConfig.fromJson/read, LlamaLimits; LlamaModel.init/load, forward(tokens), logits(tokens), generateIds(prompt,maxNewTokens), namedWeights.

- [ ] Write config rejection/bounds tests, tiny synthetic model shapes/gradients/ties, generation constraints and public checkpoint test.
- [ ] Run missing API RED; implement config, compositional model and strict single-file loader.
- [ ] Pin public repository revision and hashes; generate full-vocabulary logits for fixed token prefixes using Candle 0.9.1, no Python.
- [ ] Compare Scala full-prefix logits against the independent reference with explicit tolerances and verify greedy argmax IDs.
- [ ] Review, fix demonstrated defects, run full suite with actual public checkpoint and GPU gates enabled; commit evidence, not external model weights.
