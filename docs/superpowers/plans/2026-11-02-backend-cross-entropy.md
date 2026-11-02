# Backend cross entropy

Execute inline under the approved GPU training plan.

1. Move loss reduction settings to core, preserving the autograd import. Add explicit backend loss and pullback operations and a tape edge.
2. Establish reference tests for all reductions, ignored/empty rows, noncontiguous targets, extreme finite logits and seeded gradients. Preserve finite-logit rejection even on ignored rows.
3. Implement CPU Double log-sum-exp and resident CUDA row statistics, loss reduction and gradients. Transfer only target metadata and a validation status flag; budget and release all private scratch.
4. Verify CPU/CUDA parity, finite differences, repeated backward, failed allocations and composed classifier training; review and run the full public-model/CUDA gates.
5. Full GPU Llama training, persistent optimizer/checkpoint integration and diffusion/NUI migration remain subsequent work.
