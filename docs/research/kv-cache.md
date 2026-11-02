# Persistent CPU KV-cache evidence

`LlamaModel.newCache(batchSize,capacity,maxBytes)` creates an inference-only model-bound session. K/V buffers are preallocated per layer with capacity-aware strides. Each decode computes only the new token chunk's embedding/projections/MLP, rotates its Q/K at the committed absolute position, appends new K/V and attends over the visible prefix. Past keys remain rotated at their original positions.

Private mutable storage never escapes through public results: attention receives internal strided views and packs them internally; language-head logits use independent owned arrays. The implementation saves prior projection work but still copies prefix data inside reference attention. There is no zero-copy, measured speedup, FlashAttention or GPU-cache claim.

## State and limits

- Batch/capacity must be positive and capacity <= model context. Default cache byte budget is 256MiB, configurable. BigInt arithmetic checks aggregate bytes and per-array Int indexing before allocation.
- Each decode chunk must match batch and contain tokens. Token, capacity, logit and score limits apply. Logit limits count the new chunk rather than the accumulated prefix, allowing long prefixes to be processed in smaller chunks.
- Calls on one cache serialize and always run noGrad. Concurrent model parameter updates are unsupported; identity snapshots reject changed parameters until reset.
- Position commits only after all layers/head succeed. Tail writes on failure remain beyond the committed prefix and are overwritten before later use. Tests include a late head failure with unchanged position, but do not exhaustively inject every recoverable post-write failure.
- Reset discards prefix length and binds current parameter versions; buffers retain capacity. Close clears buffer/version references and is idempotent. Decode/reset/position after close reject; allocatedBytes reports zero. This refers to logical ownership, not synchronous JVM garbage collection.
- No eviction, sliding windows, serialization, GPU storage or differentiable cached training in this stage.

`generateIds` uses a fresh cache sized to prompt+requested output by default; uncached mode remains available as `useCache=false`. The last generated token need not be fed back when no further output is requested. EOS and equal-logit behavior remain unchanged.

## Verified evidence

- Batched synthetic GQA model, chunk sizes [1,1,1,1], [2,2], [1,3] and [4], all logits agree with full-prefix execution.
- Changed weights, invalid IDs, capacity/output limits, close/reset and late head errors preserve documented state rules. Earlier returned logits survive later appends, reset and weight changes unchanged.
- Huge batch/budget requests reject before narrowing/allocation.
- Public pinned tiny Llama, chunks [1,1,1,1], [2,2], [1,3], all new-position logits match the independent Candle full-prefix reference at absolute tolerance 1e-5. Four-token greedy generation agrees cached vs uncached.
- Existing generation tests verify immediate EOS, zero requested tokens and equal-logit lowest-ID selection with caching enabled by default.

The verified full suite has 94 passing tests, with public-model and actual GPU gates explicitly enabled. Artifact hashes and reference provenance are in the [model evidence](llama-model.md). Tokenizer behavior and useful trained-model text generation remain open.
