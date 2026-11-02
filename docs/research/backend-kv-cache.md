# Resident backend KV cache

Prepared model generation now defaults to cached decoding on CPU and CUDA. After prefill, only each new token passes through the model; weights and K/V remain resident. `useCache=false` retains full-prefix recomputation for comparisons.

```scala
Using.resource(prepared.newCache(batchSize=1, capacity=128)) { cache =>
  val prefill = cache.decode(promptTokens)
  val next = cache.decode(nextToken)
  assert(cache.position == promptLength + 1)
  cache.reset()
}
```

Each layer has private fixed-capacity key and value storage. CUDA appends using a device kernel and attention reads the buffers with capacity strides and a separate logical prefix length. It does not copy/compact the old prefix or download K/V. The CPU reference uses private capacity-strided arrays. `FloatBackend.attentionCache` exposes the lower-level attention resource, including differing key/value widths; its append position must be within an already written prefix, so gaps cannot expose uninitialized storage.

The model cache checks batch, positive chunk length, capacity, logits and score limits. It advances its committed position only after every layer, final download and temporary cleanup succeed. Failed calls may leave writes in the uncommitted tail; retries overwrite them and only the new valid prefix is visible. Tests inject a download failure after all layer writes and verify position zero plus successful retry. A real GPU test also rejects output allocation after append/scratch allocation, frees scratch, and succeeds after budget becomes available.

`reset` hides all old entries by setting the committed position to zero. It does not rebind changed model weights: prepared-weight changes still require re-preparation. Caches are thread-confined and closeable; closing a prepared model closes any forgotten child caches before releasing weights. Partial multi-layer allocation failure closes earlier allocations. Returned CPU logits own their data independently of the cache.

The default model cache budget is 256MiB across all layers, separately checked before allocation. CUDA also charges those buffers against the session's total device budget along with weights and scratch. For pinned SmolLM2 at capacity49, cache storage is **2,257,920 bytes**, in addition to **651,306,240 bytes** of prepared weights. Allocation usage is constant across successfully decoded chunks.

The [pinned pretrained model gate](backend-llama.md) compares all196608 cached last-token logits over four prefixes to the same Candle reference using unchanged atol/rtol2e-4. Maximum observed absolute error after the reduction/SiLU precision update is **5.7220458984375e-5**. Cached greedy IDs/EOS/text match the reference, and the native CUDA CLI prints **The capital of France is Paris.** Synthetic tests cover batching, GQA, chunk partitions, retained outputs, reset, capacity/stale/closed rejection, late failure/retry, allocation rollback, parent cleanup and cached/full generation parity. The low-level GPU gate checks unequal value widths and capacity strides.

This is a correctness implementation, not a throughput benchmark. Resident LoRA adapters now use the same cached path, with adapter updates invalidating both prepared weights and caches. GPU training, optimized GEMM/attention and diffusion/NUI migration remain open.
