# Resident Llama training

`BackendLlamaTraining.prepare(model, backend)` creates an independent resident weight snapshot over a `TrainingBackend`. `newStep()` binds one tape leaf per unique Parameter identity. The complete bias-free Llama forward graph includes embeddings, RoPE/GQA, residuals, RMSNorm, gated MLP, tied output heads and optional LoRA factors. Host Long tokens and labels remain explicit inputs.

```scala
import scalatorch.models.BackendLlamaTraining
import scalatorch.nn.Module
import scala.util.Using

// model is a LlamaModel; gpu is a live CudaSession; tokens/labels are Long tensors.
val trainedState = Using.resource(BackendLlamaTraining.prepare(model, gpu)) { prepared =>
  for _ <- 0 until 100 do
    Using.resource(prepared.newStep()) { step =>
      val loss = step.forward(tokens).crossEntropy(labels)
      loss.backward()
      step.sgd(0.01f)
    }
  prepared.stateDict() // Explicit download, including tied-name aliases.
}
Module.loadStateDict(model, trainedState)
```

The host model stays unchanged while training. Loading or updating its Parameters externally invalidates the prepared snapshot. Training state is thread-confined. The parent owns live steps and closes them before releasing resident weights. Each step owns all forward values and backward scratch through `BackendTape`; close it after each update. Gradients can be inspected through `step.parameter(p).grad`, and tied parameters return the same leaf. Frozen base leaves have no gradients.

Plain SGD computes `weight + gradient * -learningRate` in Float32. It requires a positive finite learning rate and a gradient for every trainable parameter. It validates all gradients, stages every replacement, validates replacement values, then publishes the new generation. Device allocation-budget rejection and nonfinite gradients/results leave resident weights unchanged. A successful update makes all existing step methods stale, preventing reuse of old gradients; their tape-owned buffers remain valid until close. Failed updates can be retried after resolving the cause. CUDA driver cleanup failures are not claimed to support transactional recovery.

`TrainingBackend.allFinite` performs resident validation and reads only a four-byte status flag. Model training never downloads activations or gradients for computation or weight updates. Loss validation also reads a small status flag. Explicit logging, `stateDict`, and test reference comparisons download selected values. The current implementation retains transposed weights and activations for each complete tape; it has no activation checkpointing or optimized training kernels.

Verification covers every parameter gradient against existing CPU autograd for tied and untied two-layer GQA models, LoRA/frozen-leaf behavior, host mutation and stale steps, thread/lifetime/model limits, nonfinite updates, and allocation failure followed by successful retry. Resident allocations return to the prepared baseline after every closed step and to zero after parent cleanup.

A complete synthetic Llama trains all parameters over500GPU SGD steps, with a learning-rate schedule of0.1 for250steps then0.03. Loss falls **2.185781 → 0.012040619**. At every step, an independent CPU autograd model uses the exact current resident weights and verifies loss, every parameter gradient and the resulting update. Independent CPU/GPU Float32 training trajectories initially agreed but diverged after19updates; recomputing CPU gradients at identical GPU weights reduced the observed relative error from1.44e-4 to1.38e-7. The final gate tests operations at identical weights rather than assuming long trajectories remain numerically identical.

Pinned pretrained SmolLM2-135M-Instruct trains its last-layer q/v rank4, alpha8 LoRA factors on CUDA. Over12SGD steps at learning rate1, synthetic-token loss falls **20.862274 → 10.11425**. Every adapter gradient matches CPU autograd at the existing2e-4 relative-plus-absolute tolerance. SGD updates match CPU Float32 arithmetic using the observed GPU gradients exactly; this separates update arithmetic from allowed gradient rounding differences. Frozen leaves receive no gradients, host weights remain unchanged, and the allocation baseline is **538,090,752 bytes**. The gate allows a2GiB device budget for retained forward/backward buffers. This is a synthetic adaptation test, not evidence of downstream language-quality improvement or optimized throughput.

The tests use `SCALATORCH_CUDA_TESTS=1` in the models/FFM test forks and `scalatorch.smollm.model.dir` pointing to the pinned local model. Disabled gates are skips. [Resident AdamW and checkpoint continuation](backend-adamw.md) are now implemented through `prepareAdamW`, `Step.adamW`, `saveCheckpoint` and `loadCheckpoint`. The [offline training CLI](training-cli.md) now runs text training and strict resume. Memory/performance improvements and complete diffusion/NUI migration remain open.
