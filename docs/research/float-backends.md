# Float32 inference execution

`FloatBackend[T]` selects a typed execution/storage backend without implicit CPU/GPU transfer. `CpuFloatBackend` operates on immutable `Tensor[Float]`; `CudaSession` implements the same interface for owned `CudaTensor` buffers. This explicit inference interface does not change existing Tensor or Variable operations and does not provide GPU autograd.

```scala
import scalatorch.FloatBackend

def gated[T](backend: FloatBackend[T], x: T, gateWeight: T, upWeight: T): T =
  val gate = backend.matmul(x, gateWeight)
  try
    val activation = backend.silu(gate)
    try
      val up = backend.matmul(x, upWeight)
      try backend.multiply(activation, up)
      finally backend.release(up)
    finally backend.release(activation)
  finally backend.release(gate)
```

Inputs remain unchanged, operation results are independently releasable, and the caller owns the returned value. CPU storage is immutable/GC-managed, so upload/download may retain the tensor and release is a no-op. CUDA upload/download are explicit copies; its operations use persistent device buffers, validate session/lifetime, allocate outputs within the session budget and synchronize before returning. No host copies occur inside GEMM, add, multiply, SiLU or RMSNorm.

| Operation | Current semantics |
| --- | --- |
| upload/download | Explicit CPU/backend boundary, logical CPU views are packed for CUDA |
| shape/release | Backend metadata and ownership lifecycle |
| matmul | Rank-2 Float32 matrices |
| add/multiply | Exact matching shapes; no implicit broadcasting |
| silu | Stable sign-based sigmoid; negative infinity maps to negative zero |
| rmsNorm | Positive final-axis width, matching rank-1 weight, finite positive Double epsilon |

CPU GEMM uses Numscala. CPU SiLU uses Double exponential, while CUDA uses `ex2.approx.f32` on the nonpositive exponent `-abs(x)*log2(e)` and Float32 arithmetic. Transcendental outputs are therefore compared with a fixed tolerance, not asserted bitwise equal. CUDA RMSNorm accumulates squares and computes the inverse RMS in Double, then applies scaling in Double before conversion; this avoids overflowing Float32 squares for finite large inputs. The reference kernel assigns one thread to each row and is not a performance implementation. PTX behavior follows NVIDIA's [instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).

Evidence includes hand-calculated CPU arithmetic/normalization, SiLU sign/infinity/NaN cases, CPU/CUDA comparisons across kernel blocks, strided inputs, maximum finite Float32 normalization, empty tensors/rows, invalid dimensions/epsilon and budget/lifetime rejection. A shared generic gated MLP/residual program matches at `2e-6 * max(1, abs(expected))`; its CUDA allocation count returns to zero after releasing all intermediates.

Remaining work: pretrained transformer integration and independent model references on CUDA; embedding, rotary and attention kernels; broadcasting/views; general Tensor/autograd dispatch; cuBLAS/optimized kernels, mixed precision and performance measurements. Existing full model inference and training still use the CPU implementation.
