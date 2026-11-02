# Scoped backend autodiff

`BackendTape[T]` records a first-order graph over `FloatBackend[T]`. The same code executes on CPU or CUDA. CUDA forward values, backward intermediates, gradients and explicit SGD arithmetic remain on device. Existing `Variable`, `Parameter` and optimizer APIs remain CPU-based.

```scala
import scalatorch.autograd.BackendTape
import scala.util.Using

// x, target and weights are already uploaded handles owned by the caller.
val nextWeights = Using.resource(new BackendTape(gpu)) { tape =>
  val w = tape.variable(weights)
  val residual = tape.variable(x, false).matmul(w) - tape.variable(target, false)
  val loss = (residual * residual).mean()
  loss.backward()
  val delta = gpu.scale(w.grad.get, -0.1f)
  try gpu.add(weights, delta)
  finally gpu.release(delta)
}
gpu.release(weights)
// Caller now owns nextWeights.
```

Supported differentiable operations are equal-shape addition/multiplication, scalar multiplication, rank-two GEMM, transpose, reshape, total sum and mean. `backward(seed)` accepts an explicitly owned backend seed for nonscalar outputs. Implicit backward requires a rank-zero result. Frozen leaves do not accumulate gradients. Shared edges accumulate correctly; repeated backward adds to leaf gradients until `zeroGrad`.

Importing a value creates an independently releasable reshape handle; the caller keeps its input ownership. Graph values and leaf gradients belong to the tape and must not be released by callers. They become invalid when it closes. Use a fresh tape each training iteration and close it before releasing the CUDA session. The tape is thread-confined. All graph outputs remain resident until tape close; backward scratch remains owned until backward finishes. This is not an activation-checkpointing or memory-optimized implementation.

Backward stages new gradients before publishing them. An allocation or computation failure preserves existing leaf gradients and releases backward scratch, permitting retry after freeing capacity. Native cleanup failures are still errors and do not promise transactional recovery from a failed device. A real-GPU test fills the session budget during repeated backward, verifies unchanged prior gradients and allocation count, then frees capacity and retries successfully.

Backend primitives `fill`, `sum` and `expandScalar` support the tape without host reads. `sum` returns a rank-zero Float32 result after Double accumulation; the CUDA reference uses one thread. Empty sums are zero; empty means reject. `expandScalar` requires rank zero. Tests cover IEEE nonfinite values, signed zero, empty shapes, views and allocation budgets.

CPU and real-GPU gradient tests compare matrix/layout chains against the existing autograd implementation and central finite differences. A resident GPU regression fits `y=2x+1` in 100 steps: loss **9.0 → 1.2079227e-13**, learned weights **[1.0000004, 1.9999998]**. The test checks the initial analytical gradient, learned weights, final loss, allocation baseline after every iteration and zero after cleanup. Downloads are limited to explicit test observations, not gradient computation or updates. This proves basic device training, not GPU LLM training.

The new layout gradient test exposed a pre-existing Numscala adapter defect: GEMM with both operands transposed requires packing workspace. The adapter now asks `gemmWorkspaceFloats` and calls `reserveFloats` before invocation. A direct two-transpose matrix regression reproduces the original failure and verifies the correction against the independent core GEMM.

GPU tests require `set autograd / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1"`; this is a third gate in addition to the model and FFM gates. Disabled GPU checks are skips, not passes. Transformer backward kernels, cross entropy, backend parameters/optimizers/checkpoints, broadcasting, higher-order gradients and optimized reductions remain open, along with diffusion and complete NUI migration.
