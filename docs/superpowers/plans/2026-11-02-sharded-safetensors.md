# Indexed Safetensors Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline.

**Goal:** Load standard indexed safetensors checkpoints through the existing Llama API with validated tensor ownership and bounded open files.

**Architecture:** SafeTensorSet owns one single archive or all shards referenced by model.safetensors.index.json. A bounded strict index parser reads weight_map and optional metadata.total_size. Every shard header is validated before payload materialization; actual tensor names must exactly match their index assignments. Typed reads delegate to the owning archive. Failure closes all acquired channels; close invalidates reads.

**Tech Stack:** Scala3.3.8, Java25, sbt2.0.10, existing Jackson and safetensors reader.

**Spec:** docs/superpowers/specs/2026-11-02-scala-torch-design.md, M3 direct public checkpoint loading.

## Global Constraints
- No Python/runtime conversion and no implicit downloads.
- Exactly one of model.safetensors and model.safetensors.index.json; ambiguous directories reject.
- Referenced names must be simple .safetensors basenames without path separators/colon; existing local file symlinks remain usable (e.g. Hub cache), not a filesystem sandbox.
- Index defaults16MiB, max128 shards; global tensor count uses ReadLimits.maxTensors. Per-shard header and per-tensor materialization limits remain active.
- Optional total_size must equal aggregate payload byte count; missing total_size accepted.
- No external modification during open lifetime; no offload or bounded total model-memory claim.

## Review Focus
- Missing/wrong/duplicate/unindexed tensor assignments fail before payload reads.
- Duplicate/malformed JSON, invalid sizes, traversal/absolute paths reject.
- Failed later-shard open and normal close release previously acquired channels.
- Typed reads and returned data survive archive close appropriately.
- Existing single-file and independently resharded public model produce the same logits.

## Task1
Files: io/SafeTensorSet.scala, io/SafeTensorsIndex.scala, io/ShardSuite.scala, models/LlamaModel.scala, models/ShardedLlamaSuite.scala, Rust reshard generator and docs.
Interfaces: ShardLimits(maxIndexBytes=16777216,maxShards=128); SafeTensorSet.open(directory,readLimits=ReadLimits(),shardLimits=ShardLimits()), tensors, readFloat/readDouble/readLong/readBoolean, close. LlamaModel.load gains final optional shardLimits.
- [x] Write valid routing/lifetime, invalid assignment/index/path/limits, cleanup and model equivalence tests; observe missing API RED.
- [x] Implement index parser and resource-owning set; integrate Llama loader; targeted GREEN.
- [x] Reshard pinned tiny public model with independent Rust safetensors; verify logits against existing Candle reference.
- [x] Full suites with all public/CUDA gates, review, document and commit.
