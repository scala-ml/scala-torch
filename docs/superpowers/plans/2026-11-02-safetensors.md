# Safetensors Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read published safetensors weights in Scala without Python conversion.

**Architecture:** A `tensorio` module (directory `io`) depends on core and Jackson streaming JSON. Opening an archive validates bounded UTF-8 header, duplicate-free objects, exact integer metadata, shapes/byte counts and complete contiguous non-overlapping ranges. Individual tensors materialize on demand through chunked positional channel reads. F16/BF16/F32 decode to Float32; F64/I64/BOOL use typed reads. No native code or object deserialization.

**Tech Stack:** Scala 3.3.8, sbt 2.0.10, Java 25, Jackson core 2.20.1, MUnit.

**Spec:** `docs/superpowers/specs/2026-11-02-scala-torch-design.md`, M3 loader gate. Primary format: https://github.com/safetensors/safetensors and its Rust tensor validation implementation.

## Global Constraints

- Runtime and tests require no Python. Upstream Rust may generate a checked-in independent fixture during development only.
- Unsupported dtypes reject explicitly; supported subset is F16/BF16/F32/F64/I64/BOOL.
- Raw Tensor public immutability and ownership rules remain intact.
- Archive file contents must not change while open; returned Tensors own independent data.
- Defaults: header <=100,000,000 bytes; <=100,000 tensors; rank <=64; materialized output array <=1GiB, configurable. Long metadata and file offsets; CPU arrays remain Int-indexed.

## Review Focus

- No Double conversion for shape/offset JSON integers.
- Duplicate keys, overflow, fractional/negative offsets, overlap, gaps, truncation and unindexed trailing data reject before payload reads.
- FP16 subnormals, signed zero, NaN and Inf and BF16 bits decode correctly.
- File handles close on failed open and use-after-close fails predictably.
- Independent upstream fixture and nn state loading must prove interoperability, not just a self-generated round-trip.

## Task 1: Reader

Files: `io/src/main/scala/scalatorch/io/{SafeTensors,SafeTensorsHeader}.scala`, `io/.../SafeTensorsSuite.scala`, build module and nn integration dependency.

Interfaces: `SafeTensors.open(path, limits=ReadLimits()): SafeTensors` (AutoCloseable), `.tensors: Map[String,TensorInfo]`, `.metadata: Map[String,String]`, `.readFloat`, `.readDouble`, `.readLong`, `.readBoolean`; `SafeDType`, `TensorInfo(dtype,shape,begin,end)`, `ReadLimits`.

- [ ] Write typed/half decoding, scalar/empty, malformed header/range/limit/lifetime tests; run missing API RED.
- [ ] Implement strict metadata validation and bounded chunk materialization with owned primitive arrays.
- [ ] Generate a small fixture with pinned upstream Rust safetensors, record version/hash and expected values, commit fixture plus generator provenance; ordinary tests need no Rust.
- [ ] Load fixture through Scala into nn state_dict and verify a known Linear output.
- [ ] Review, run full suite with actual GPU tests, document supported scope and commit.

## Remaining scope

Sharded indexes, public model config/tokenizers, attention/RoPE/KV cache and actual LLM generation/comparison remain required. A working format reader alone does not satisfy real-model inference.
