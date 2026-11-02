# Byte-level BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline.

**Goal:** Read the pinned SmolLM2 tokenizer directly and reproduce Rust tokenizers token IDs and decoded text without Python.

**Architecture:** Immutable Scala byte-level BPE, bounded strict JSON reader, explicit supported pipeline validation. Support no normalization/postprocessor, optional individual Digits before ByteLevel, literal special tokens, rank-ordered adjacent merges and byte decoding. Reject unsupported behavior instead of silently approximating. Text generation/chat formatting is a subsequent checkpoint.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, existing Jackson streaming parser. Rust tokenizers 0.22.1 is development-only reference.

**Spec:** docs/superpowers/specs/2026-11-02-scala-torch-design.md, M3 tokenizer equivalence.

## Global Constraints
- No runtime Python or native tokenizer dependency.
- Tokenizer file max 32MiB; fromJson max 32Mi UTF-16 units (review ruling); vocabulary <= 1 million; text <= 1Mi UTF-16 units; per-piece bytes <= 16384 to bound reference BPE work.
- Malformed UTF-16 input rejected; decoding incomplete UTF-8 uses replacement characters.
- Unknown pipeline options that alter semantics are rejected; no arbitrary chat-template execution.

## Review Focus
- Unicode whitespace/numbers, multibyte Unicode and contractions: pinned Rust fixtures.
- Added token matching and skip-special decode: adjacent/embedded special fixtures.
- Merge ranking/overlap: synthetic rank and repeated-pair tests.
- Invalid JSON/config/vocab/merges: reject before encode.
- Resource limits and invalid IDs/surrogates: explicit rejection tests.

## Task 1
Files: models/ByteBpeTokenizer.scala, models/TokenizerJson.scala, models/TokenizerSuite.scala, tokenizer-reference Rust generator, tokenizer fixtures/provenance.
Interfaces: ByteBpeTokenizer.read(Path), fromJson(String), encode(String):Array[Long], decode(Array[Long],skipSpecialTokens:Boolean=true):String, vocabSize:Int.
- [x] Write synthetic exact-ID/roundtrip/invalid schema tests and opt-in pinned SmolLM2 Rust fixture test.
- [x] Run models/Test/testOnly scalatorch.models.TokenizerSuite; expect missing API RED.
- [x] Implement bounded reader and supported byte BPE pipeline.
- [x] Run targeted then full module tests with public model, tokenizer and CUDA gates enabled; expect all pass.
- [x] Review, document precise compatibility and limitations, commit.
