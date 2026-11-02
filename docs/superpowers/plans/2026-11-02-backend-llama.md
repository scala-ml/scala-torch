# Prepared backend Llama inference

Continue the approved GPU transformer integration inline.

1. Add `BackendLlama.prepare(model, backend)` with resident parameter snapshots, explicit close and stale-source rejection. Upload linear weights in their matmul orientation once; deduplicate parameter/orientation pairs. Scope intermediate handles per attention/MLP/block and release prior hidden states as layers advance. Failed preparation/forward must release owned allocations.
2. Compose the existing backend embedding, normalization, GEMM, rotary, causal GQA, activation and residual primitives. Expose backend-owned forward output, CPU logits, and full-prefix seeded generation. Preserve limits and explicit CPU Long token input. Initial prepared path supports the loader's bias-free, adapter-free Llama architecture; reject bias/LoRA before upload, never silently omit them. GPU adapters and cache are subsequent integration work.
3. Compare synthetic/tied/batched CPU-backend results with existing model inference; test stale/closed/invalid states, preparation and forward allocation rollback. Real CUDA tests compare pinned tiny-model full logits and pinned pretrained SmolLM2 reference last-token logits/greedy output.
4. Provide an offline backend text CLI that runs native tokenizer plus prepared CPU/CUDA inference. Enable native access in models execution/tests and explicitly enable CUDA model test gates.
5. Review and run all model/public/CUDA gates. This proves inference integration, not optimized performance, GPU training or complete diffusion/NUI replacement.
