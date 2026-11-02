# Prepared CPU/CUDA Llama inference

`BackendLlama.prepare(model, backend)` uploads model weights once and composes the explicit FloatBackend operations into a complete Llama forward. The same model runs with `CpuFloatBackend` or a scoped `CudaSession`. No Python, JNI shim, model interpreter or CPU activation fallback is involved in the CUDA path.

```scala
import scalatorch.ffm.CudaSession
import scalatorch.models.BackendLlama
import scala.util.Using

CudaSession.withDevice() { gpu =>
  Using.resource(BackendLlama.prepare(model, gpu)) { prepared =>
    val logits = prepared.logits(tokens) // explicit final CPU download
    val ids = prepared.generateIds(promptIds, 12)
  }
}
```

`forward` returns a caller-owned backend tensor; `logits` downloads it and releases that handle. Weights remain resident between calls. Linear weights are uploaded already transposed, deduplicated by parameter identity/orientation. A tied embedding/head needs two physical orientations with the current packed GEMM. SmolLM2's resident weight allocation is **651,306,240 bytes**. Intermediates release within attention/MLP/block scopes and prior hidden states release as layers advance. Tests verify the resident allocation count after each forward and zero after close, including failed preparation and output allocation.

Prepared objects are thread-confined and explicitly closeable. Source Parameter changes reject subsequent inference until re-preparation; concurrent mutation is unsupported. Input IDs, context/output limits and score workspace remain checked. Bias-free Llama projections support existing LoRA adapters; biases still reject before uploads. GPU training remains pending. Generation uses the [resident KV cache](backend-kv-cache.md) by default; `useCache=false` retains full-prefix recomputation. Logits download for CPU sampling through the existing seeded sampler.

LoRA factors are uploaded once in transposed form alongside base weights. Each adapted projection evaluates `base + ((input Aᵀ) Bᵀ) * (alpha/rank)`, preserving the CPU operation order rather than folding scale into a factor. All activations and factors remain on device during CUDA execution. Full forward, cached decoding and generation use the same projection path. Updating either factor invalidates the prepared model and its caches. Load a bound `LoRACheckpoint` into a matching `LlamaLoRA` attachment before preparing `adapted.model`; optimizer steps still run on CPU.

The public adapter gate trains the existing 7,680-parameter final-layer SmolLM2 attachment, restores its bound checkpoint, verifies one exact resumed CPU optimizer step, and compares every full/cached GPU logit against CPU at unchanged `2e-4 * (1 + abs(reference))`. Maximum absolute error is **1.10626220703125e-4**, and resident weights/factors occupy **651,336,960 bytes** (30,720 bytes above the base model). Synthetic tests cover nonzero attention/MLP adapters, batches, tied weights, stale factors and allocation-failure cleanup.

The initial trained-adapter comparison failed with maxabs0.0022087. Prepared CPU was exact. Diagnostic replacement of CUDA SiLU/attention with CPU calculations still left maxabs0.0015297 with serial GPU GEMM. After matching Numscala's four partial sums, those diagnostic full logits were exact; retaining real GPU attention with CPU SiLU passed. The final implementation uses device-only Double SiLU and the matching reduction order. All diagnostic CPU fallbacks were removed. Primitive regressions cover dense SiLU precision and GEMM reduction tails; see [backend numerical semantics](float-backends.md).

## Independent model checks

Artifacts are the same hash-pinned [tiny Llama](llama-model.md) and [trained SmolLM2](trained-text-generation.md) used by the existing CPU gates; GPU tests check their model/config/reference hashes directly.

- Tiny random Llama: all 128,000 Candle reference logits pass fixed atol/rtol `1e-5`; observed maximum absolute error **5.960464477539063e-8**, greedy next ID **16196**.
- SmolLM2-135M-Instruct: all 196,608 reference last-token logits across four prefixes pass unchanged atol/rtol `2e-4`; observed maximum absolute error **5.7220458984375e-5**. All eight greedy continuation IDs, EOS and decoded text match: **The capital of France is Paris.**
- Synthetic CPU-backend models cover batching, tied embeddings, empty inputs, greedy/seeded generation and stale/closed handles. Real CUDA tests check retained weight allocation and preparation/forward cleanup.

These are inference correctness measurements on the local GTX1650 SUPER. No optimized-GEMM, throughput or production performance claim follows from them.

## Upload ordering correction

The first integrated generation run raised CUDA700 after passing four reference prefixes. An unchanged diagnostic rerun passed, so the failure was intermittent. Review identified a missing ordering dependency: pageable `cuMemcpyHtoD` can return before device DMA completes on the default stream, while our kernels use a nonblocking compute stream. Integer metadata consumed before transfer completion can invalidate kernel indices. This behavior is specified in NVIDIA's [Driver API synchronization documentation](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).

All uploads now use a shared helper that synchronizes the default stream after HtoD before any compute kernel can consume the destination, including integer metadata, rotary coefficients and the original probe path. Cleanup now preserves the original exception rather than replacing it with a subsequent release failure. The fixed focused suite and independent CUDA CLI run pass. The ordering defect is established by the API contract; the intermittent CUDA700 was not reduced to a deterministic minimal race reproducer.

## Offline CLI

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Compile / run / javaOptions += "-Xmx3g";models / Compile / runMain scalatorch.models.BackendTextRun cuda /absolute/path/to/smollm2-135m "What is the capital of France?" 12 --smollm-chat'
```

Verified output is the Paris sentence above. Choose `cpu` for the same prepared inference path on CPU. The optional flag selects the explicit pinned SmolLM2 template, not general Jinja templates. Model CUDA test gates require `set models / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1"` in addition to the documented public-model properties; FFM tests have their own corresponding envVars setting. Without those gates, GPU checks skip and must not be reported as passing.

The pretrained LoRA integration suite uses a two-minute test timeout after the added concurrent GPU model checks pushed its previous 30-second allowance over the limit. Its training steps, numerical assertions and resume checks are unchanged; timeout configuration is not a performance acceptance criterion.
