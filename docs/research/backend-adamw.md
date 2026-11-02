# Resident AdamW and checkpoint continuation

`AdamWBackend[T]` extends `TrainingBackend[T]` with immutable, independently owned optimizer state. Both CPU and CUDA implementations use Float32 weights/gradients and Double first/second moments, matching the existing CPU `AdamW` formula and evaluation order. `AdamWConfig` validates learning rate, betas, epsilon and decoupled weight decay.

`adamW(weight, gradient, previous, config)` borrows its inputs and returns an `AdamWUpdate` containing a new weight handle and `AdamWState`. Callers own both outputs. State exposes shape/counter and an explicit `snapshot(): AdamWMoments` host transfer. `restoreAdamW` validates finite moments, nonnegative second moments, shape and positive counter before creating independently owned state. Closed or foreign state rejects; step-counter overflow rejects without mutation.

CUDA stores two interleaved Doubles per element in private budgeted storage. One update kernel computes moments and weights with explicit round-to-nearest Double operations. It checks finite inputs/results and publishes only a small status flag to the host. Old weights/moments never mutate. New weight/state/status allocations are preflighted; any rejected update releases every temporary allocation. Empty tensors retain valid independently owned state.

Primitive tests compare weights and both moments with the CPU optimizer over repeated updates, including gradients near1e20,1e-30 and Float32 subnormal values. Export/restore produces exactly matching next updates. Tests also cover IEEE invalid inputs, invalid restored state, counter overflow, closed state, exact allocation budgets and cleanup.

## Model training

```scala
import scalatorch.AdamWConfig
import scalatorch.models.BackendLlamaTraining
import scala.util.Using

Using.resource(BackendLlamaTraining.prepareAdamW(model, gpu, AdamWConfig(0.01))) { prepared =>
  for _ <- 0 until 15 do
    Using.resource(prepared.newStep()) { step =>
      step.forward(tokens).crossEntropy(labels).backward()
      step.adamW()
    }
  prepared.saveCheckpoint(path, step = 15)
}

Using.resource(BackendLlamaTraining.prepareAdamW(freshModel, gpu, AdamWConfig(0.01))) { resumed =>
  val savedStep = resumed.loadCheckpoint(path)
  Using.resource(resumed.newStep()) { step =>
    step.forward(tokens).crossEntropy(labels).backward()
    step.adamW()
  }
}
```

Optimizer configuration is fixed for a prepared AdamW model. SGD and AdamW modes are explicit and cannot be mixed through the same prepared instance. Each AdamW step stages all trainable weights and moments before publishing a new generation. Missing gradients, nonfinite results or device-budget failure leave weights, moments, counters and generation unchanged. Successful updates/restores invalidate existing step methods. Host Parameters stay unchanged. Driver cleanup failures after publication are not claimed to support rollback.

`optimizerState()` explicitly downloads moment slots and preserves tied-name aliases. Frozen parameters have no optimizer slots. A two-layer synthetic Llama reduces loss **2.1563542 → 0.025828669** over30CUDA AdamW updates. CPU autograd verifies every gradient at identical weights; CPU optimizer arithmetic using the observed device gradients verifies each weight update exactly. CPU-only backend training reaches0.025828589 along its own numerical trajectory.

Pinned SmolLM2-135M-Instruct trains last-layer q/v rank4 alpha8 LoRA factors at learning rate0.01 for30CUDA AdamW updates, reducing synthetic-token loss **20.862274 → 4.8634853**. Adapter gradients match CPU autograd at2e-4 relative-plus-absolute tolerance; updates match CPU optimizer arithmetic using observed GPU gradients. Frozen leaves remain untracked. Resident weights and moments occupy **538,213,632 bytes** between steps; tests allow a2GiB device budget for retained computation buffers. This synthetic adaptation gate does not measure conversation quality.

## Checkpoint format and failure behavior

`STBADM01` is a full-model checkpoint envelope:8-byte magic,32-byte architecture fingerprint,8-byte big-endian payload length, then an existing checksummed `STCKPT01` payload. The fingerprint binds Llama configuration, actual normalization epsilons and projection LoRA presence/alpha. The inner payload validates shapes, tied aliases, trainability, optimizer membership/settings, Double moment counters and SHA-256 checksum. Different initialization seeds may load the same architecture because all weights, including frozen base weights, are stored.

Saving uses temporary files and atomic replacement with no non-atomic fallback. Invalid/nonfinite snapshots and byte-budget rejection preserve the previous destination. Loading validates the envelope and payload into detached host Parameters, checks all weights are finite, then stages every backend weight and optimizer state. Only after all uploads succeed does it replace the resident generation. File errors, configuration mismatch, checksum failure, trailing/truncated data and failed device allocations preserve the previous resident state. The default file budget is4GiB; parsing allocations are bounded by the target model's shapes. Host staging and serialization remain explicit transfer costs.

CPU and CUDA tests verify7updates + save/load +13updates exactly match20uninterrupted updates in both weights and moments. The pretrained SmolLM2 test verifies15updates + save/load into a fresh preparation +15updates exactly match30uninterrupted updates. Tests exercise partial model-update and restore allocation failures, complete rollback, successful retry, malformed files and invalid save snapshots. Allocations return to the expected weight/moment baseline after each closed step and to zero after cleanup.

The caller-supplied checkpoint training step is distinct from optimizer slot counters; both persist. Current backend checkpoints store the complete base, so LoRA files are not adapter-only. The existing CPU `LoRACheckpoint` format remains separate. Training CLI, data pipelines, gradient accumulation policies, learning-rate schedules, optimizer-state offloading and performance/memory optimization remain future work, alongside diffusion and full NUI migration.
