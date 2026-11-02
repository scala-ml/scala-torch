# Resident Llama training

Execute inline under the approved GPU LLM training plan.

1. Prepare independently owned resident weights, deduplicated by Parameter identity. Bind one leaf per parameter into each scoped step; preserve frozen base weights and tied embedding/head accumulation.
2. Compose the complete bias-free Llama graph, including GQA/RoPE, residuals, RMSNorm, gated MLP, LoRA and cross entropy, through BackendTape. Keep model and allocation limits.
3. Provide staged plain SGD with finite learning-rate, gradient and result checks on device. Publication follows successful computation of every replacement. Reject stale/reused steps and external host parameter changes; close all live steps with the parent. Downloads occur only for explicit inspection/export and small validation flags.
4. Verify all-parameter CPU/GPU gradients, full synthetic-model loss reduction, LoRA frozen-base invariants, limits/lifetime/allocation failures, and real pretrained tiny/SmolLM2 training where resources permit.
5. Review and run all public model and CUDA gates. Backend AdamW/checkpoint continuation, performance improvements, diffusion and full NUI migration remain subsequent work.
