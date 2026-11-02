# Rotary backend gradient

Execute inline under the approved GPU training plan.

1. Establish failing CPU/CUDA tape tests for batched, strided RoPE gradients, offsets and finite differences.
2. Add `TrainingBackend.ropeBackward` and tape composition. Apply the transpose rotation by negating sine coefficients, reusing the existing kernel and validation. No activation/gradient downloads.
3. Verify empty handles, invalid arguments, budget rejection and a composed matrix/rotation gradient chain.
4. Review and run all public-model and CUDA gates. Embedding, attention and loss backward, complete GPU LLM training, performance and diffusion/NUI migration remain open.
