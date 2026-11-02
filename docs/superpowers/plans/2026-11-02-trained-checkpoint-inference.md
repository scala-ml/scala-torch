# Inference from trained checkpoints

Execute inline under the approved JVM model/training plan.

1. Add a shared checked AdamW checkpoint weight-loading path that discovers saved optimizer settings, validates moment values/counters/checksum while streaming past them, and never retains optimizer arrays for inference.
2. Reuse envelope validation and expose bounded context inspection before constructing a matching model. Restore full or LoRA models from TrainRun metadata; verify the tokenizer fingerprint. Keep explicit template-based restore for metadata-free API/legacy checkpoints.
3. Extend BackendTextRun with --checkpoint while preserving existing prompt/count/chat arguments. Generate on CPU/CUDA with the restored resident inference model and no data/optimizer arguments.
4. Compare restored logits/tokens with the trained model, test corrupt files/moments/context/tokenizers and legacy restoration, then run actual saved SmolLM2 CLI checkpoints and the complete public/CUDA suite.
5. Further work includes richer Transformer-style APIs/model coverage, efficient kernels/memory, diffusion and full NUI migration.
