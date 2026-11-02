# Offline training CLI

Execute inline under the approved JVM LLM training plan.

1. Add bounded UTF-8 line/document loading with explicit EOS delimiters, deterministic packed next-token batches and ignored padding targets. Resume batch selection derives from the global saved step; fingerprint the actual token corpus.
2. Extend checkpoint binding with optional caller context while retaining existing empty-context behavior. CLI context binds corpus/tokenizer hashes, sequence/batch settings and adapter initialization/selection; mismatches reject before publication.
3. Add TrainRun cpu|cuda MODEL_DIRECTORY DATA CHECKPOINT TOTAL_STEPS with explicit resume, sequence/batch/rate/budget and full-model/LoRA options. Default to last-layer q/v LoRA. Validate inputs early, report each loss, save at boundaries and close all resources on failure.
4. Test batching, bounds/Unicode/padding, CLI validation, full/LoRA training, checkpoint resume and context mismatch. Run actual pretrained CPU/CUDA CLI commands, review and execute the complete public/CUDA suite.
5. Subsequent work includes trained-artifact inference ergonomics, data formats/shuffling and a Transformers-style upper layer. Diffusion and complete NUI migration remain part of the full objective.
