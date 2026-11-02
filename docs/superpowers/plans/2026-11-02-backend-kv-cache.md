# Resident backend KV caches

Continue prepared GPU inference integration inline.

1. Add a backend attention-cache resource with fixed-capacity private K/V storage. Validate batch/head/width/chunk/position and score limits before writes. A caller may retry or truncate at an already written prefix; gaps reject. CPU uses capacity-strided private arrays; CUDA appends through a device kernel and reads capacity-strided K/V directly in attention, without prefix copies/downloads.
2. Add a model-bound BackendLlamaCache with aggregate allocation budget, decode/reset/close and commit position only after all layers, download and cleanup succeed. Partial writes beyond committed position remain invisible and are overwritten on retry. Stale prepared weights still require re-preparation.
3. Wire offsets and per-layer cache resources through BackendLlama. Cached generation becomes default, with full-prefix mode retained for comparison. Weights/cache remain resident; final logits still download for sampling.
4. Validate CPU/GPU cached-vs-full logits for batches/GQA/chunks, fixed allocation counts, reset/close/stale/invalid inputs, late budget failure and retry, capacity/score limits, plus pinned SmolLM2 Candle logits/greedy output and native CLI.
5. Review and run all public/model/FFM CUDA gates. This does not add GPU adapters/training, optimized attention or diffusion/NUI migration.
