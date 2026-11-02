# Attention and rotary position evidence

The autograd module implements rank-4 CPU Float32 causal attention and Llama-style rotary embeddings. Primary conventions were checked against [Transformers v4.57.1 Llama](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py): contiguous half-vector rotation and contiguous grouped-query head repetition. The Scala implementation uses its own reference loops and analytical pullbacks.

## Semantics

`query.causalAttention(key,value,queryStart,maxScoreElements)` takes `(batch,heads,tokens,width)` values. Query and key widths match; value width may differ. Q heads must be divisible by matching K/V heads. Query head `h` uses KV head `h / (Hq/Hkv)`. Key positions begin at zero; query token `t` can attend through key position `queryStart+t`. Nonempty queries require keys. Empty query/batch values preserve shape and produce zero input gradients.

Forward computes scaled dots, subtracts each visible row's maximum, and normalizes with Double accumulations/probabilities before producing Float32 output. Masked future positions are excluded from loops, so even NaN future keys/values do not contaminate results. Backward applies the softmax Jacobian analytically; contributions from all sharing query heads add to K/V gradients. Each pullback uses detached first-order tensors and retains parameter invalidation metadata.

The default probability workspace is limited to 16,777,216 Double elements; callers may change the cap, but Int indexing remains required. This is a probability-array limit, not a total peak-memory budget. Gradients and input copies consume additional memory. No fused kernel, dropout, custom/padding mask or sliding-window key origin is implemented.

`input.rope(positionStart,theta)` uses positive even head width and pairs features `j` and `j+D/2`, with angle `(positionStart+t) * theta^(-2*j/D)`. Backward applies the inverse rotation. Position addition uses Long before conversion to Double, and angle overflow rejects. Default theta is 10000. Advanced rope scaling and per-example position arrays are not implemented.

## Verified tests

- Exact known prefix means and value gradients for uniform scores.
- Masked future NaNs leave both output and q/k/v gradients unchanged.
- GQA forward/head grouping and shared value gradients; all q/k/v gradients checked by central differences with nonuniform seed and unequal value width.
- Decode offsets agree with full-prefix results.
- Batched noncontiguous values and nonuniform gradients route correctly.
- Large finite logits normalize without exponential overflow.
- Known RoPE forward values/inverse gradients, nonuniform gradients across multiple batches/heads, and positions near Int.MaxValue.
- Invalid shapes and score caps reject, and empty oversized unused dimensions avoid narrowing or allocation.
- Frozen K/V parameter changes after forward reject backward before publishing query gradients.

The attention checkpoint had 81 passing tests including actual GPU probes. These operation tests use numerical and analytical references, not an executed PyTorch model. Double intermediates deliberately differ from exact Float32 kernel rounding. Subsequent [Llama model assembly](llama-model.md) verifies a published checkpoint against Candle; tokenizer behavior and persistent KV-cache equivalence remain open.
