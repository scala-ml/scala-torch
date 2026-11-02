# Float32 inference execution

`FloatBackend[T]` selects a typed execution/storage backend without implicit CPU/GPU transfer. `CpuFloatBackend` operates on immutable `Tensor[Float]`; `CudaSession` implements the same interface for owned `CudaTensor` buffers. Operations do not record gradients themselves. The separate [BackendTape](backend-autograd.md) now provides basic first-order CPU/CUDA autodiff without changing existing Tensor or Variable operations.

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
| scale | Elementwise Float32 scalar multiplication, preserving IEEE nonfinite behavior |
| fill / sum / expandScalar | Backend creation, total Double-accumulated scalar reduction, and rank-zero expansion |
| silu | Stable sign-based sigmoid; negative infinity maps to negative zero |
| rmsNorm | Positive final-axis width, matching rank-1 weight, finite positive Double epsilon |

CPU GEMM uses Numscala. CUDA GEMM now uses four Float32 partial sums, with incomplete groups accumulated into the first sum, matching Numscala's transposed-right linear projection reduction order. Other CPU layouts can use different reduction orders; general bitwise GEMM equivalence is not promised.

CPU SiLU uses Double exponential. CUDA now computes `exp(-abs(x))` with Double range reduction, split ln(2), and a degree-12 Taylor polynomial on `[-ln(2)/2, ln(2)/2]`. Two normal power-of-two factors preserve subnormal exponential results. Sigmoid and the final product use Double before Float32 conversion. This replaced the Float32 exponential approximation after trained adapter logits exposed accumulated error. A 20,010-value real-GPU regression checks agreement within two Float32 ULPs, including tiny values and extremes; it is fixture evidence, not a correctly-rounded guarantee for every input. NaN, infinities and signed zero remain covered separately.

CUDA RMSNorm accumulates squares and computes the inverse RMS in Double, then applies scaling in Double before conversion; this avoids overflowing Float32 squares for finite large inputs. The reference kernel assigns one thread to each row and is not a performance implementation. PTX behavior, including required rounding modifiers on integer-to-float conversions, follows NVIDIA's [instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).

Evidence includes hand-calculated CPU arithmetic/normalization, SiLU sign/infinity/NaN cases, CPU/CUDA comparisons across kernel blocks, strided inputs, maximum finite Float32 normalization, empty tensors/rows, invalid dimensions/epsilon and budget/lifetime rejection. A shared generic gated MLP/residual program matches at `2e-6 * max(1, abs(expected))`; its CUDA allocation count returns to zero after releasing all intermediates.

Remaining work: broadcasting and general strided CUDA views; general Tensor/autograd dispatch; cuBLAS/optimized kernels, mixed precision and performance measurements. Existing autograd inference/training remains CPU-based; [prepared model inference](backend-llama.md) now runs on CUDA.

## Embedding and layout operations

The shared interface now also includes `embedding(table, indices)`, `reshape(value, shape)` and `transpose(value, dim0, dim1)`.

Embedding requires a rank-2 table with positive width and CPU Long indices. IDs are checked against the table vocabulary before narrowing; scalar IDs produce one embedding vector, and arbitrary index shapes append the embedding width. CUDA transfers validated integer IDs and gathers directly from the persistent table without downloading it. The CPU implementation respects logical table/index layouts.

Reshape preserves logical element order and requires unchanged element count. CPU packs a noncontiguous input when needed. CUDA tensors are packed, so reshape creates another handle over the same device allocation without copying or charging the budget twice. Closing one handle does not invalidate another; the final close frees storage. Scope exit releases all remaining handles, including shared views.

Transpose accepts any valid pair of axes. CPU uses an immutable strided view. CUDA writes a packed device output using output dimensions and permuted input strides. Transposing an axis with itself retains a handle. Empty outputs skip coordinate arithmetic, avoiding overflow for otherwise enormous zero-sized shapes. The packed output can be reshaped and passed directly to existing GEMM/normalization kernels.

Temporary CUDA index/layout buffers count toward the session budget and are released after synchronized execution, including rejected output allocation. Tests cover both reshape close orders, nested handles, escaped views, repeated/scalar/empty/invalid IDs, all axis pairs on rank-2/3/4 tensors, huge empty dimensions and budget rollback. These operations provide transformer data movement; rotary/attention implementation is described below; [prepared pretrained GPU execution](backend-llama.md) is now verified.

## Rotary positions and causal GQA

`rope(value, positionStart=0, theta=10000)` accepts `(batch,heads,tokens,width)` with positive even width and uses the existing Llama split-half rotation convention. Double sine/cosine coefficients are calculated on the host from absolute positions, then uploaded as positional metadata. CUDA performs Double multiply/add/subtract and writes Float32 activations; activations remain on device. Invalid/nonfinite theta, negative position and overflowing angles reject. Empty input retains a handle without allocating coefficient storage.

`causalAttention(query,key,value,queryStart=0,maxScoreElements=16777216)` uses the same rank-4 layout. Query heads must be divisible by matching key/value heads; each contiguous group maps to one KV head. Query/key widths match; value width may differ. A query at local token t sees keys through `queryStart+t`, capped by available keys. No padding mask, dropout, bidirectional mode or GPU backward is implemented.

Both implementations accumulate dot products in Double, scale by `1/sqrt(headWidth)`, subtract the visible maximum and normalize. The CPU uses Double exponential; CUDA converts shifted scores to Float32 for `ex2.approx`, then uses Double denominator and value accumulation. Nonfinite visible scores propagate NaN; masked future entries are not read. CUDA scores occupy `batch*queryHeads*queryTokens*keyTokens*8` bytes of temporary storage, counted against the session budget in addition to the logical score limit. Scratch is released on success and rejected output allocation. The reference kernel assigns one thread per query row; it is not optimized attention.

Tests cover hand-calculated prefix means, exact CPU agreement with existing autograd forward semantics (including strided input), CPU/CUDA comparisons across batches/grouped heads, nonzero and maximum Int position offsets, differing value widths, huge finite dots, masked/visible NaNs, extreme theta, empty outputs and allocation limits. GPU attention comparisons use `3e-6 * max(1, abs(expected))` on these fixtures; this is test evidence, not a global approximation-error guarantee. [Prepared pretrained transformer execution](backend-llama.md) now passes the independent reference gates.
