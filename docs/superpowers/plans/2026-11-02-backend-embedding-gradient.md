# Backend embedding gradients

Execute inline under the approved GPU LLM training plan.

1. Test repeated/scalar/batched/empty token gradients against existing CPU autograd and finite differences.
2. Add `TrainingBackend.embeddingBackward(tableShape, indices, gradient)` and tape composition with shared shape/ID validation. CPU scatter-add follows logical token order.
3. CUDA groups the explicitly host-side token IDs into per-row linked lists in original token order. Upload only integer metadata; one thread per gradient element reduces its row's occurrences in Float32. Avoid atomic races and scans across unrelated vocabulary rows/tokens. Preflight output plus metadata budget before host grouping.
4. Verify exact repeated-row ordering, untouched rows, invalid IDs/shapes, empty buffers and lifetime/budget cleanup. Train a small tied embedding/projection graph on CUDA with CPU gradient/update parity and stable allocations.
5. Review and run all public-model and CUDA gates. Attention/loss backward, full GPU LLM training, optimizer/checkpoint integration, optimized kernels and diffusion/NUI migration remain open.
