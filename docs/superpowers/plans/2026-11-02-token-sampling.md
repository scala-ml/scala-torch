# Seeded Token Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline.

**Goal:** Support reproducible temperature/top-k/top-p Llama text generation while preserving default greedy behavior.

**Architecture:** Immutable validated SamplingConfig and per-generation TokenSampler own randomness. Compute stable Double softmax, restrict top-k first, then retain smallest descending-probability prefix reaching top-p, renormalize. Zero temperature is greedy, lowest ID wins ties. Share the sampler between cached and uncached generation paths.

**Tech Stack:** Scala3.3.8, Java25, sbt2.0.10; java.util.Random for explicit seeded JVM reproducibility.

**Spec:** docs/superpowers/specs/2026-11-02-scala-torch-design.md, M3 sampling.

## Global Constraints
- Existing default greedy IDs/text and call signatures remain source-compatible.
- Reject empty/nonfinite logits, nonfinite/negative temperature, negative top-k, and top-p outside (0,1].
- topK0 means all tokens; topK larger than vocabulary clamps to vocabulary.
- Zero temperature ignores filters; temperature>0 enables sampling.
- Seed repeats per generation, no global shared RNG; no cross-framework seed-stream claim.

## Review Focus
- Top-p boundary includes token crossing threshold; top-k precedes top-p.
- Extreme finite Float logits and tiny temperature cannot overflow softmax.
- Greedy ties remain lowest ID; filters leave at least one eligible token.
- Seed replay and empirical categorical frequencies.
- Cache/uncached generation and text continuation use the same policy.

## Task 1
Files: models/TokenSampler.scala, SamplerSuite.scala, LlamaModel.scala, TextGeneration.scala, relevant tests/docs.
Interfaces: SamplingConfig(temperature:Double=0,topK:Int=0,topP:Double=1,seed:Long=0); TokenSampler(config).sample(logits:Array[Float]):Long; TokenSampler.probabilities(logits,config):Array[Double]. Add final optional sampling parameter to generateIds and TextGeneration.generate.
- [x] Write analytical distribution, filter ordering/boundaries, validation/extreme numeric, seeded-frequency and model replay/cache parity tests.
- [x] Observe missing APIs RED using models/Test/testOnly SamplerSuite.
- [x] Implement sampler and integrate both generation APIs; targeted GREEN.
- [x] Run all module suites with all public/CUDA gates; run seeded trained text experiment; review, document and commit.
