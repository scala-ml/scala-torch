# Causal attention backend gradients

Execute inline under the approved GPU LLM training plan.

1. Add a jointly owned `(dq, dk, dv)` pullback to TrainingBackend and per-backward scratch memoization in BackendTape so three edges compute it once.
2. Establish failing CPU/CUDA reference and finite-difference tests for GQA, offsets and unequal value widths.
3. Implement Double CPU reductions and CUDA reduction ownership per batch/KV-head group. Each CUDA worker visits its query heads/tokens in CPU order and owns disjoint gradient regions; no atomic accumulation. Reuse bounded row probability/dp scratch, then convert Double gradient buffers to Float32 outputs. Validate logical score limits and actual workspace/output budgets.
4. Verify causal masking, empty/frozen inputs, repeated backward, tied graph edges, allocation failures and a composed rotary-attention training step against CPU. Gradients and activations stay on device.
5. Review and run all public model and CUDA gates. Cross entropy, full GPU LLM training, optimizer/checkpoint integration, performance and diffusion/NUI migration remain open.
