# Cross-Entropy Training Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline.

**Goal:** Train causal language models using stable differentiable class-index cross entropy on last-axis logits.

**Architecture:** LossOps computes shifted log-sum-exp in Double and a first-order softmax-minus-onehot pullback, retaining Float logits and two Double row statistics rather than a full Double probability tensor. Targets are Int64 and match every dimension except the final class axis. Sum/Mean/None reductions and ignoreIndex are explicit.

**Tech Stack:** Scala3.3.8, Java25, sbt2.0.10, existing CPU autograd/AdamW/Llama.

**Spec:** docs/superpowers/specs/2026-11-02-scala-torch-design.md, M2 losses and M4 LLM fine-tuning prerequisites.

## Global Constraints
- CPU Float32 input/output, Int64 targets; classes are on the last axis (documented layout).
- Finite logits required, including ignored rows; no implicit softmax input.
- Mean divides by nonignored count. Empty/all-ignored Mean is NaN with zero gradient; Sum is zero; None preserves target shape with ignored zeros.
- Class weights, soft targets, label smoothing and higher derivatives are not in this increment.
- Stable shifted computation must not lose log-sum terms for huge equal logits.

## Review Focus
- Known losses and seeded unreduced gradients.
- Noncontiguous logits/targets and every-coordinate finite differences.
- Ignore mask/reduction denominator and zero-valid-row behavior.
- Invalid targets/shapes/nonfinite input, noGrad and graph invalidation.
- End-to-end Llama next-token loss reduction and prediction improvement through AdamW.

## Task1
Files: autograd/LossOps.scala, Variable.crossEntropy, LossSuite.scala, models/LanguageTrainingSuite.scala, docs/examples.
Interfaces: enum LossReduction {None,Sum,Mean}; Variable.crossEntropy(targets:Tensor[Long], reduction:LossReduction=Mean, ignoreIndex:Long = -100):Variable.
- [x] Write analytical/gradient/edge validation tests and small Llama next-token training test; observe missing API RED.
- [x] Implement loss and pullback; targeted GREEN, report initial/final training loss.
- [x] Full suites with all public/CUDA gates; review, document and commit.
