# Scoped backend autodiff

`BackendTape[T]` records a first-order graph over `TrainingBackend[T]`, which extends `FloatBackend[T]` with explicit pullbacks. The same code executes on CPU or CUDA. CUDA forward values, backward intermediates, gradients and explicit SGD arithmetic remain on device. Existing `Variable`, `Parameter` and general optimizer APIs remain CPU-based; [resident Llama training](backend-llama-training.md) now supplies scoped model steps, staged SGD/AdamW and checkpoint continuation. Forward-only backend implementations can continue to implement `FloatBackend` alone.

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

Supported differentiable operations are equal-shape addition/multiplication, scalar multiplication, rank-two GEMM, transpose, reshape, total sum/mean, SiLU, RMSNorm, split-half RoPE, embedding lookup and causal grouped-query attention. `backward(seed)` accepts an explicitly owned backend seed for nonscalar outputs. Implicit backward requires a rank-zero result. Frozen leaves do not accumulate gradients. Shared edges accumulate correctly; repeated backward adds to leaf gradients until `zeroGrad`.

Importing a value creates an independently releasable reshape handle; the caller keeps its input ownership. Graph values and leaf gradients belong to the tape and must not be released by callers. They become invalid when it closes. Use a fresh tape each training iteration and close it before releasing the CUDA session. The tape is thread-confined. All graph outputs remain resident until tape close; backward scratch remains owned until backward finishes. This is not an activation-checkpointing or memory-optimized implementation.

Backward stages new gradients before publishing them. An allocation or computation failure preserves existing leaf gradients and releases backward scratch, permitting retry after freeing capacity. Native cleanup failures are still errors and do not promise transactional recovery from a failed device. A real-GPU test fills the session budget during repeated backward, verifies unchanged prior gradients and allocation count, then frees capacity and retries successfully.

Backend primitives `fill`, `sum` and `expandScalar` support the tape without host reads. `sum` returns a rank-zero Float32 result after Double accumulation; the CUDA reference uses one thread. Empty sums are zero; empty means reject. `expandScalar` requires rank zero. Tests cover IEEE nonfinite values, signed zero, empty shapes, views and allocation budgets.

CPU and real-GPU gradient tests compare matrix/layout chains against the existing autograd implementation and central finite differences. A resident GPU regression fits `y=2x+1` in 100 steps: loss **9.0 → 1.2079227e-13**, learned weights **[1.0000004, 1.9999998]**. The test checks the initial analytical gradient, learned weights, final loss, allocation baseline after every iteration and zero after cleanup. Downloads are limited to explicit test observations, not gradient computation or updates. This proves basic device training, not GPU LLM training.

The new layout gradient test exposed a pre-existing Numscala adapter defect: GEMM with both operands transposed requires packing workspace. The adapter now asks `gemmWorkspaceFloats` and calls `reserveFloats` before invocation. A direct two-transpose matrix regression reproduces the original failure and verifies the correction against the independent core GEMM.

SiLU backward reuses the forward kernel's Double sigmoid calculation and applies `sigmoid(x) * (1 + x * (1 - sigmoid(x)))`. Exact sigmoid endpoints avoid infinity-times-zero artifacts; a zero derivative masks even NaN upstream values, matching existing CPU autograd. RMSNorm input gradients use Double row reductions. Weight gradients first compute one Double inverse norm per row, then reduce each column deterministically without atomic updates. The inverse scratch counts against the session budget and releases on success or rejected output allocation. No gradients or activations download during either pullback.

Transformer tests compare both input and scale gradients against existing autograd and central finite differences. Frozen inputs/weights, empty batches and extreme/nonfinite SiLU inputs are covered. A 20-step resident CUDA gated MLP (RMSNorm, gate/up projections, SiLU product and down projection) matches CPU gradients and updated weights at every step; loss falls **0.09506189 → 0.022409033**. The allocation count returns to the persistent input/weight baseline after each step and to zero after cleanup. Explicit test observations download values; training arithmetic stays on device. This is a transformer-MLP training gate, not full attention or LLM training.

RoPE backward applies the transpose rotation using negated sine coefficients, preserving the original offset/theta. CPU and CUDA share their respective forward rotation implementations; CUDA gradients remain resident and only position coefficients upload as in forward execution. Tests cover batched logical views, maximum Int position, extreme theta, central finite differences, and input/weight gradients through a projection-plus-rotation chain. Budget rejection and independently owned empty handles are also verified.

Embedding backward produces a dense table gradient. Repeated IDs accumulate in logical token order using Float32 addition, matching existing CPU scatter-add semantics. CUDA constructs per-row occurrence lists from the explicitly host-side IDs, uploads only integer metadata, and assigns one thread to each output element. It does not download gradients, use atomic writes, or scan unrelated token occurrences. Untouched rows remain zero even when another row receives NaN. Dense output and both metadata arrays are preflighted against the device budget before grouping. Empty IDs produce zero gradients without metadata allocation.

Embedding tests cover repeated/strided/scalar/empty IDs, reference gradients and finite differences, stable cancellation order, nonfinite row isolation, shape/ID limits and exact allocation budgets. A tied embedding/output-projection graph reuses one tape leaf for both paths and matches CPU gradients and updates over20CUDA steps; loss falls **0.13348606 → 0.0065176673**, with stable allocations between steps and zero after close. This verifies tied parameter accumulation but is not a language-model training result.

Attention backward returns three independently owned query/key/value gradients. The tape memoizes this joint computation only within one backward call, retaining separate edge handles; tied input edges accumulate all contributions, and a new seed recomputes the pullback. The current implementation computes all three gradients even when some inputs are frozen.

CUDA assigns one worker to each batch/KV-head group. That worker visits its query heads and tokens in CPU order, owns disjoint Double gradient buffers, and reuses per-group probability/dp row scratch. No atomic accumulation or activation/gradient download occurs. Probabilities use the same Float32 exponential approximation as CUDA forward, with Double dot products, normalization and gradient accumulation. Logical score limits and the actual workspace/output budget are checked separately. This serial-per-group reference kernel is not an optimized attention implementation.

Tests cover GQA, offsets up to maximum Int, unequal value widths, all-input finite differences, repeated backward, tied edges with different seeds, frozen/empty queries, masked future NaNs and exact allocation limits. A 20-step rotary/GQA training graph matches CPU gradients and updated tensors throughout, reducing loss **0.15299392 → 0.043585684**, with stable resident allocation between steps and zero after cleanup.

During integration, a parameter named `scale` in the new backward kernel read zero despite a host argument of0.5. Renaming only its declaration/load to `attention_scale` restored0.5 and passed the numerical tests; reverting the name reproduced zero and failure. The module also contains a kernel named `scale`. Both attention kernels now use distinct parameter names. This establishes a reproducible parameter-name interaction in the tested module, not a proven general compiler/driver defect. All diagnostic stores and downloads were removed before final verification.

`Value.crossEntropy(targets, reduction, ignoreIndex)` now supports last-axis classes, host Long targets, `None`/`Sum`/`Mean`, and ignored rows. `LossReduction` lives in core; the existing `scalatorch.autograd.LossReduction` source import remains available through aliases. All-ignored or empty means return NaN with zero gradients; sums return zero. Every logit must be finite, including ignored rows, matching `Variable`.

CUDA computes per-row Double maxima, log-sum-exp and unrounded losses in budgeted scratch. Reduced losses accumulate these Double losses before conversion to Float32. The exponential uses the range-reduced degree-12 Double polynomial already used by SiLU; logarithms normalize the mantissa and evaluate the odd atanh series through degree39. Backward computes probabilities in Double before multiplying by the seed, preserving small probabilities whose gradients become representable with large seeds. This is a reference implementation, not a throughput-optimized loss kernel or a universal correctly-rounded math guarantee.

Only target metadata uploads and a four-byte nonfinite-status flag downloads during loss execution. Logits, statistics, losses and gradients stay on device. Forward and backward each recompute statistics and validate logits; the tape currently retains no specialized saved-loss context. Exact workspace/output preflight and cleanup cover ignored/empty rows and rejected operations.

Tests compare all reductions and gradients with the existing CPU loss, including scalar/strided/empty targets, wide class axes, extreme logits and seeds, finite differences, repeated accumulation, frozen leaves, invalid IDs/nonfinite logits, and exact allocation limits. A tied embedding/classifier trains for40CUDA steps with CPU gradient/update parity, reducing loss **1.2666097 → 0.032580208**, with stable allocations after each step and zero after cleanup. Full GPU Llama training is now integrated through [resident model steps](backend-llama-training.md).

GPU tests require `set autograd / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1"`; this is a third gate in addition to the model and FFM gates. Disabled GPU checks are skips, not passes. Broadcasting, higher-order gradients and optimized reductions remain open, along with diffusion and complete NUI migration.
