# Native backend feasibility

## Driver-only CUDA computation

The `ffm` module calls the CUDA C Driver API through Java 25 `java.lang.foreign`. No JNI shim, Python process, C++ wrapper, or local CUDA compiler is involved in these probes.

`CudaDriver.withDevice` owns a primary-context reference, restores the previous current context, and registers acquired resources for reverse-order release. Host memory and library lookup share a confined arena. `CudaProbe.roundTrip` exercises synchronous host/device transfer.

`CudaProbe.gemm` loads a bundled PTX module and runs a packed row-major Float32 matrix multiply. Each invocation owns device buffers, module, stream, and completion event. It waits on the recorded event before copying results back; a final stream synchronization is registered after the dependent resources so cleanup drains work before releasing them, including when the body fails. CUDA status errors raise exceptions; cleanup failures are retained by `Using.Manager`.

Input dimensions and array lengths are checked before native access. Products use Long; outputs beyond Int indexing are rejected. Empty outputs do not launch a kernel; K=0 produces zeros on the GPU. The PTX kernel guards excess block threads and performs separate rounded multiply/add to compare directly with the CPU reference.

Official references used for ABI and execution semantics:

- [CUDA Driver execution API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
- [CUDA memory management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
- [PTX instruction set](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

## Scope and outstanding work

This is a correctness/ownership spike, not the production CUDA Tensor backend. Every probe invocation loads/allocates again; bindings are not yet cached. The one-output-per-thread GEMM is deliberately simple and does not replace cuBLAS, tiling, mixed precision, or performance measurements. Native CPU BLAS and cuBLAS loading remain unverified. Windows support is also unimplemented.

The public Tensor is still CPU-only. Persistent device storage, backend dispatch, asynchronous API lifetime rules, capability detection, failure injection, and leak accounting require their own implementation and tests. Real successful resource release is observed through successful probe completion; exhaustive cleanup under driver failure has not been proven.

## Reproduction

Java 25, Scala 3.3.8, sbt 2.0.10, Linux x86_64, GTX 1650 SUPER, NVIDIA driver reporting CUDA driver API version 13030.

```sh
sbt ';set ffm / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1";ffm / Test / testOnly scalatorch.ffm.*'
sbt 'ffm / Compile / runMain scalatorch.ffm.CudaProbe'
```

Tests compare (M,N,K)=(2,2,3), (5,17,7), and (3,4,0) to the CPU reference; the second case spans multiple blocks. The CLI verifies exact known values `[58,64,139,154]`. Both commands fail rather than silently fall back if the GPU probe cannot run.
