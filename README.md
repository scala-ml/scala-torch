# scala-torch

Deep learning in Scala 3, targeting Java 25 and Python-free LLMs, training, and image generation.

Requires **Java 25, Scala 3.3.8 LTS, sbt 2.0.10**. Versions are pinned in the build.

Implemented: CPU tensors (Float32/Float64/Int64/Bool), checked shapes and views, slicing, broadcasting, Float32 add/reduction/matrix multiplication, and a numscala GEMM adapter. The FFM module includes host memory scopes, a real CUDA memory-copy probe, and a driver-level Float32 GEMM probe with owned stream/event completion.

Not implemented yet: autograd, a persistent GPU Tensor backend, cuBLAS acceleration, model loading, LLM generation, or diffusion/NUI migration. The small PTX GEMM kernel verifies correctness and FFM execution; it is not intended for model performance.

```sh
sbt 'core / Test / testOnly scalatorch.*'
sbt 'cpu / Test / testOnly scalatorch.cpu.*'
sbt 'ffm / Test / testOnly scalatorch.ffm.*'
# Requires a Linux 64-bit CUDA driver and GPU; fails if unavailable:
sbt 'ffm / Compile / runMain scalatorch.ffm.CudaProbe'
# Force the actual GPU test, including when using an existing sbt server:
sbt ';set ffm / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1";ffm / Test / testOnly scalatorch.ffm.*'
```

```scala
import scalatorch.*
import scalatorch.cpu.NumscalaBackend

val x = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), Shape.from(Seq(2L, 3L)))
val gram = NumscalaBackend.matmul(x, x.transpose(0, 1))
assert(gram.toFloatArray.toSeq == Seq(14f, 32f, 32f, 77f))
```

`reshape` requires a contiguous layout; call `.contiguous` to copy a strided view first. Input and extracted arrays are copied. Operations currently use explicit CPU backends; no implicit device transfers or in-place public mutations.

- [Design and roadmap](docs/superpowers/specs/2026-11-02-scala-torch-design.md)
- [Initial implementation plan](docs/superpowers/plans/2026-11-02-tensor-foundation.md)
- [Reuse and migration audit](docs/research/reuse-audit.md)
- [Implementation evidence and remaining gates](docs/research/implementation-status.md)

The target is concise Scala model authoring and Python-free real-model execution, rather than full PyTorch API compatibility.
