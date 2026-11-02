# Safetensors loading evidence

The `tensorio` module implements `scalatorch.io.SafeTensors` in Scala. Jackson core 2.20.1 handles streaming JSON parsing only; tensor indexing, validation, typed decoding, ownership and loading are Scala code. Java 25 supplies the FP16 conversion primitive.

Primary references: [format specification](https://github.com/safetensors/safetensors#format), [upstream Rust validation](https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs). The format uses a little-endian 64-bit header length, JSON metadata and packed tensor payloads; duplicate keys and gaps are disallowed.

## Supported API

`SafeTensors.open(path, limits)` validates metadata before returning an AutoCloseable index. `tensors` and `metadata` are immutable maps. `readFloat` accepts F32, F16 or BF16 and returns owned Float32 data. `readDouble`, `readLong` and `readBoolean` handle F64, I64 and BOOL respectively. Unsupported dtypes and typed conversions reject explicitly. No implicit F64 narrowing, pickle, Python execution, native glue or public weight writer is involved.

Reads use 64KiB positional chunks. Returned primitive arrays become Tensor-owned without a second full-size copy, and returned Tensors survive archive closure. Metadata remains inspectable after close; payload reads then reject. Concurrent reads/close on the same archive serialize; external modification of an open file is unsupported.

## Validation and limits

Opening rejects duplicate keys at every object level, malformed UTF-8/unpaired surrogate escapes, fractional or negative integers, Long overflow, unsupported types, missing/unknown tensor fields, shape or byte-count overflow, invalid ranges, overlaps, gaps, truncation, multiple JSON roots and unindexed trailing payload. Empty and rank-zero tensors are supported, including empty shapes with dimensions above 2^53 without loss of integer precision.

Default limits are 100,000,000 header bytes, 100,000 tensors, rank 64, and 1GiB per materialized output array. Parser nesting/string/number limits additionally bound header processing. These limits are configurable through ReadLimits; the array cap counts decoded output bytes (F16/BF16 expand to Float32), not total archive size or peak process memory. CPU arrays remain Int-indexed.

## Independent evidence

The [committed fixture provenance](../../io/src/test/resources/safetensors/PROVENANCE.md) records the official Rust safetensors 0.6.2 generator, dependency lock and binary SHA-256. Tests read its values using Scala and load its weights directly through nn state_dict into a derived Linear layer, whose known output is `[24,52]`.

Tests cover FP16/BF16 subnormals, signed zero, NaN/Inf; exact Int64 values beyond Double precision; all supported typed reads; scalar/empty shapes; F32/F16/BF16/F64 chunk boundaries; malformed metadata/ranges; converted-output size limits; and archive lifetime. Runtime and ordinary tests require neither Rust nor Python.

## Remaining work

This verifies format interoperability and loading generated upstream weights. It does not verify a published model's behavior. Sharded indexes, config/tokenizer readers, attention/RoPE/KV cache, public model revision/hash manifests and full inference comparisons remain required. mmap/zero-copy and performance measurements are also open.
