# Trained Text Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline.

**Goal:** Generate text offline from pinned SmolLM2 weights in Scala, verified against independent Candle logits and greedy IDs.

**Architecture:** Compose existing tokenizer and cached Llama model in a text-generation API/CLI. Keep raw prompt mode and explicit SmolLM2 ChatML formatting separate; do not execute arbitrary Jinja. Reference generator loads original BF16 safetensors into Candle F32 CPU, matching Scala execution dtype.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25; Rust Candle 0.9.1/tokenizers 0.22.1 only for reference generation.

**Spec:** docs/superpowers/specs/2026-11-02-scala-torch-design.md, M3 offline text generation/logits comparison.

## Global Constraints
- Pin SmolLM2 revision 12fd25f77366fa6b3b4b768ec3050bf629380bac and artifact hashes.
- Python-free inference, no implicit downloads or chat-template execution.
- Preserve token-ID generation API, context checks and EOS behavior.
- Return continuation IDs and decode the continuation as a complete sequence.
- No performance parity claim; trained 135M model does not prove large-model/GPU support.

## Review Focus
- Prompt/continuation boundaries and zero-token output.
- Default system insertion and explicit system messages exactly match pinned template.
- Vocabulary mismatch and empty raw prompts rejected.
- Real tied embeddings/GQA/BF16 weights exercised, not only synthetic tests.
- Independent full vocabulary logits at fixed tolerance and greedy generation IDs.

## Task 1
Files: models/TextGeneration.scala, models/TextRun.scala, models/TextGenerationSuite.scala, Rust reference-generator/src/bin/smollm.rs, trained reference resources/docs.
Interfaces: TextGeneration.generate(model,tokenizer,prompt,maxNewTokens):GeneratedText(ids:Vector[Long],text:String); SmolChat.format(messages:Vector[ChatMessage],addGenerationPrompt=true):String. TextRun DIRECTORY PROMPT [MAX_NEW_TOKENS [--smollm-chat]].
- [x] Write exact chat template, continuation-only/zero-token, mismatch/empty tests; run testOnly TextGenerationSuite and observe missing API RED.
- [x] Implement composition and explicit formatter/CLI; target tests GREEN.
- [x] Generate independent pinned model fixtures (first four cached steps full logits and up to16 greedy IDs), write gated public test (atol=2e-4,rtol=2e-4 fixed before comparison).
- [x] Run full suites with all public/CUDA gates, run actual CLI, review, document and commit.
