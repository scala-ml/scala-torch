# Training Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save and resume Python-free Scala training with exact weights and optimizer moments/counters.

**Architecture:** `Checkpoint.save(model, optimizer, path, step)` writes a versioned binary training-only format to a sibling temporary file, appends SHA-256, and atomically replaces the destination. `Checkpoint.load(model, optimizer, path): Long` stages every tensor and optimizer slot, validates names/aliases/shapes/trainability/optimizer membership/hyperparameters/checksum/trailing bytes, then commits. Metadata counts are checked against the target model before allocation; text is bounded. This is not a replacement for the future safetensors public model loader.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25 standard streams/digests/files, existing nn.

**Spec:** `docs/superpowers/specs/2026-11-02-scala-torch-design.md` (approved), M2 checkpoint/restart gate.

## Global Constraints

- No Python, pickle, Java object deserialization, or executable checkpoint contents.
- Save/load/optimizer steps are not concurrent on the same model.
- Checkpoints are taken at optimizer-step boundaries; gradients and RNG/data-loader state are not serialized.
- Existing destination must survive a failed save; invalid load must not change weights, gradients or optimizer state.
- Persist optimizer state by canonical parameter names, not ephemeral Variable identities.

## Review Focus

- AdamW moments/counters must reproduce uninterrupted training exactly after restore.
- Optimizer subsets and shared aliases must map to the same target identities.
- Corrupt/truncated/trailing data and oversized metadata must reject before commit.
- Wrong model/optimizer configuration must not partially restore.
- Restored arrays must be independent from optimizer snapshot objects.

## Task 1: State snapshots and persistent checkpoint

Files: `nn/.../Optimizer.scala`, new `nn/.../Checkpoint.scala`, new `nn/.../CheckpointSuite.scala`.

- [ ] Write save/resume tests for AdamW and SGD, missing-gradient slots and tied parameters; compare exact final model state to uninterrupted training.
- [ ] Write corruption/truncation/trailing/config mismatch and failed-save tests asserting unchanged model state.
- [ ] Run `sbt 'nn / Test / testOnly scalatorch.nn.CheckpointSuite'`; verify missing API failure.
- [ ] Add immutable optimizer snapshots and validate all slot lengths, finite moments, nonnegative second moments and positive saved counters before restore.
- [ ] Implement bounded streaming parsing, raw Float32 bit preservation, digest validation, temporary-file cleanup and atomic replacement. Require matching optimizer hyperparameters rather than silently changing the caller's training configuration.
- [ ] Review and verify the full test suite including the actual GPU gate; commit.

## Remaining M2 work

Embedding, normalization, additional activations and other declared M2 requirements remain open. Deterministic resume with stochastic layers or data-loader state needs additional serialized state before claiming that broader capability.
