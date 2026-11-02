# Modules and Optimizers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an actual MLP using Scala-defined modules and reusable optimizers.

**Architecture:** `nn` depends on autograd. `Parameter` owns a current immutable Variable leaf; replacing it preserves gradients but invalidates saved graphs. `Module[A]` is a Scala 3 derived parameter-tree typeclass for case classes, Parameters, Option and Vector; traversal retains named aliases while optimizer lists deduplicate identities. Linear uses conventional weight shape (out,in). SGD and AdamW compute all candidate updates before committing any, reject nonfinite gradients, and preserve gradients until zeroGrad.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, MUnit, numscala.

**Spec:** `docs/superpowers/specs/2026-11-02-scala-torch-design.md` (approved), M2 follow-on from `2026-11-02-autograd.md`.

## Global Constraints

- Python-free CPU Float32 training; no silent device conversion.
- Raw Tensors remain immutable through public APIs.
- Full LLM, diffusion and NUI requirements remain open.

## Review Focus

- Shared parameters update once despite multiple names.
- Old graphs, including saved frozen parameters, reject backward after update before publishing gradients.
- Optimizer validation is atomic across all parameters.
- AdamW advances bias-correction steps only for parameters with gradients.
- State dict load validates all keys/shapes and tied aliases before any update.

## Task 1: Parameter, derived modules, Linear, optimizers

Files: `nn/src/main/scala/scalatorch/nn/{Parameter,Module,Linear,Optimizer}.scala`; update Variable saved-dependency validation; tests in `nn/src/test/scala/scalatorch/nn/TrainingSuite.scala`.

Interfaces: `Parameter(data, trainable=true)` exposes value/data/grad/zeroGrad; `Module.namedParameters(model)`, `.parameters(model)`, `.stateDict(model)`, `.loadStateDict(model,state)`; `Linear(weight,bias) derives Module` with `.apply(input: Variable)` and `.init(in,out,seed,bias=true)`. `SGD(parameters,learningRate,weightDecay=0)` and `AdamW(parameters,learningRate,beta1=0.9,beta2=0.999,epsilon=1e-8,weightDecay=0.01)` expose step/zeroGrad.

- [ ] Write tests for derived names/aliases, known Linear outputs, seeded initialization, shared-weight SGD, known AdamW steps, invalid gradients/hyperparameters, stale graphs, atomic state load and MLP XOR learning.
- [ ] Run `sbt 'nn / Test / testOnly scalatorch.nn.*'` and verify missing-API failure.
- [ ] Implement the stated interfaces and graph invalidation, retaining immutable snapshots.
- [ ] Verify tests, review, fix demonstrated defects, run full existing suite, commit.

## Task 2: Persistent checkpoint continuation

- [ ] Implement safe weight/optimizer-state file serialization and verified training restart, including optimizer counters/moments and corruption validation. In-memory state_dict alone does not complete this requirement.
- [ ] Complete Embedding/norm/activation and the other M2 gates before declaring M2 done.
