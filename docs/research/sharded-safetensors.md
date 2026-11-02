# Indexed safetensors checkpoints

`SafeTensorSet.open(directory)` reads either `model.safetensors` or `model.safetensors.index.json` plus its referenced shards. `LlamaModel.load` now uses this interface. A directory containing both forms is rejected as ambiguous. Tensor metadata and typed reads are available from the set; `close` closes all owned channels and rejects later reads. Returned tensors own their copied data.

The index follows the [Hugging Face serialization convention](https://huggingface.co/docs/huggingface_hub/package_reference/serialization): `weight_map` maps tensor names to shard filenames and `metadata.total_size`, when present, declares payload size. The loader checks duplicate fields, exact integer sizes, UTF-8/tensor-name validity, nonempty assignments, each shard's actual tensor-name set, and aggregate payload byte count. Unreferenced files in the directory are ignored; tensors inside referenced shards must all be assigned correctly. Unknown index metadata is ignored except total_size; unsupported top-level fields reject.

Defaults: index16MiB, at most128 referenced shards, global tensor count from ReadLimits (100000 by default). Each shard still uses the safetensors header/rank/dtype/offset validation and per-tensor allocation limits. All headers validate before model tensors materialize. This bounds open files and individual parsing/materialization, not total model RAM. All referenced channels remain open until close; failure at any acquisition/validation step closes previously acquired channels. There is no payload reopening between indexing and reads.

Shard references must be simple `.safetensors` basenames, without slash, backslash, colon or NUL. Existing local symlinks remain supported for cache layouts. This rejects index path traversal syntax but is not a hostile-filesystem sandbox; callers must not externally modify open files. Header or payload tampering during use is outside this contract.

## Evidence

Six I/O tests cover routing, close/data ownership, single-file selection and ambiguity, F32/F16/F64/I64/BOOL reads through the upstream Rust fixture, invalid names/assignments/total_size/index JSON/path syntax, resource limits and repeated later-shard failure. The Linux `/proc/self/fd` test confirms no retained descriptors after twelve failed opens.

A synthetic tied Llama model split across files retains exact logits, shared embedding/head identity and parameter gradients. The pinned public tiny Llama checkpoint was independently split with Rust safetensors0.6.2, preserving original tensor bytes. Its two-file Scala load passes all128000 existing Candle logit comparisons at atol=rtol1e-5 and generates next token16196. Original model provenance and Candle reference remain in [Llama model evidence](llama-model.md).

Reproduce the public split (development-only Rust tool, no Python):

```sh
cargo run --release --manifest-path io/src/test/fixture-generator/Cargo.toml --bin reshard -- /path/to/tiny-llama /path/to/tiny-llama-sharded
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Test / javaOptions += "-Dscalatorch.llama.sharded.dir=/path/to/tiny-llama-sharded";models / Test / testOnly scalatorch.models.ShardedLlamaSuite'
```

The sorted tensor names are assigned alternately to two shards. Aggregate payload bytes4129088. The test checks these SHA-256 values before loading:

| Artifact | SHA-256 |
|---|---|
| config.json | `5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382` |
| model-00001-of-00002.safetensors | `a5d05cc8c8a5107b4f14c77671fbf0924de9fe5eabfa475d91a3c32995bf1555` |
| model-00002-of-00002.safetensors | `71393387b01ff466e85a7bb3826e113fea16f51da432dde3e66be22094761730` |
| model.safetensors.index.json | `3393dbcd8d3936382c7ad054951eb3605b663dc3a968e47b1d6fa31cdf868247` |

This is independently resharded public-model evidence, not validation of an originally published multi-shard large model. A real large-model memory/performance run, download manager, quantized dtype families, GPU offload and platform-wide filesystem validation remain open. The existing Llama model-parameter limit still applies.

Full verification at this checkpoint: **116 passed, 0 failed, 0 skipped**, including all four public-reference gates and actual CUDA. Review found no blocking defects; the public shard gate explicitly requires hashed index/shard artifacts.
