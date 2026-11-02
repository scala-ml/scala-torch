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

Prepared objects are thread-confined and explicitly closeable. Source Parameter changes reject subsequent inference until re-preparation; concurrent mutation is unsupported. Input IDs, context/output limits and score workspace remain checked. This initial path supports the loader's bias-free, adapter-free Llama architecture; adapters/biases reject before uploads. It does not silently ignore them. GPU adapter inference, KV cache and GPU training remain pending. Generation recomputes the full prefix with resident weights, downloads logits and samples on CPU using the existing seeded sampler.

## Independent model checks

Artifacts are the same hash-pinned [tiny Llama](llama-model.md) and [trained SmolLM2](trained-text-generation.md) used by the existing CPU gates; GPU tests check their model/config/reference hashes directly.

- Tiny random Llama: all 128,000 Candle reference logits pass fixed atol/rtol `1e-5`; observed maximum absolute error **8.940696716308594e-8**, greedy next ID **16196**.
- SmolLM2-135M-Instruct: all 196,608 reference last-token logits across four prefixes pass unchanged atol/rtol `2e-4`; observed maximum absolute error **9.918212890625e-5**. All eight greedy continuation IDs, EOS and decoded text match: **The capital of France is Paris.**
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
