# Transformer backend gradients

Continue the approved GPU training plan inline using executing-plans.

1. Introduce `TrainingBackend[T] extends FloatBackend[T]` for explicit SiLU and RMSNorm pullbacks. CPU/CUDA implementations provide these capabilities; BackendTape requires this interface. Forward-only backends retain FloatBackend.
2. Test SiLU/RMSNorm input and weight gradients against CPU autograd and finite differences, including zero/extreme inputs, frozen parameters, empty batches and allocation cleanup. Establish failures before implementation.
3. Reuse the accurate CUDA SiLU sigmoid computation for its derivative. Implement RMSNorm input pullback with Double row reductions; compute per-row inverse norms once for weight-gradient reduction, without atomic writes or host transfers.
4. Add tape SiLU/RMSNorm composition and verify a gated transformer MLP training step on real GPU against CPU gradients and update results. Keep all forward/backward intermediates resident.
5. Review, run all public-model and three CUDA gates, document evidence and limits. Embedding, rotary, attention and loss backward, full GPU LLM training, optimizer/checkpoint integration, performance and all diffusion/NUI work remain open.
