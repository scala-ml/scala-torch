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

This is a correctness/ownership spike, not the production CUDA Tensor backend. Every probe invocation loads/allocates again; bindings are not yet cached. The one-output-per-thread GEMM is deliberately simple and does not replace cuBLAS, tiling, mixed precision, or performance measurements. Native CPU BLAS and cuBLAS loading remain unverified.

The shared driver loader now supports 64-bit Windows (`nvcuda.dll`) and Linux
(`libcuda.so.1`). On Windows 11 / Oracle Java 25.0.5 / NVIDIA driver 610.88,
the existing real-device probe first reproduced the Linux-only rejection
(two failures), then passed all three cases after the loader correction.
The full FFM run passed all 35 tests with `SCALATORCH_CUDA_TESTS=1`, including
resident tensors, ownership, attention, gradients and AdamW. Linux was not rerun
for this correction; its library name and driver calls are unchanged.

With sbt 2 thin-client mode, environment variables come from the server process.
After changing the CUDA gate in PowerShell, restart the server before testing:

```powershell
sbt shutdown
$env:SCALATORCH_CUDA_TESTS = "1"
sbt "ffm/testOnly *"
```

Check actual test output for failures and skips: this Windows thin client returned
exit code zero even for the deliberately failing reproduction. Neither successful
CUDA primitives nor library tests establish practical CUDA diffusion or NUI's
Python removal.

The public Tensor is still CPU-only. Persistent device storage is now implemented as described below. Backend dispatch, asynchronous API lifetime rules, capability detection, failure injection, and leak accounting remain open. Real successful resource release is observed through successful probe completion; exhaustive cleanup under driver failure has not been proven.

## Reproduction

Java 25, Scala 3.3.8, sbt 2.0.10, Linux x86_64, GTX 1650 SUPER, NVIDIA driver reporting CUDA driver API version 13030.

```sh
sbt ';set ffm / Test / envVars += "SCALATORCH_CUDA_TESTS" -> "1";ffm / Test / testOnly scalatorch.ffm.*'
sbt 'ffm / Compile / runMain scalatorch.ffm.CudaProbe'
```

Tests compare (M,N,K)=(2,2,3), (5,17,7), and (3,4,0) to the CPU reference; the second case spans multiple blocks. The CLI verifies exact known values `[58,64,139,154]`. Both commands fail rather than silently fall back if the GPU probe cannot run.

## Persistent device storage

`CudaSession.withDevice` now owns a primary-context scope, one PTX module and one nonblocking stream across multiple operations. `CudaTensor` is an explicit packed Float32 device allocation. This is the resource layer for future general tensor/backend dispatch; the array-backed `Tensor[A]` and model autograd still execute on CPU.

```scala
import scalatorch.ffm.CudaSession
CudaSession.withDevice(maxBytes=256L*1024*1024) { gpu =>
  val x = gpu.upload(cpuInput)
  val w1 = gpu.upload(cpuWeight1)
  val w2 = gpu.upload(cpuWeight2)
  val hidden = x.matmul(w1)
  val output = hidden.matmul(w2)
  hidden.close()
  val cpuOutput = output.toCpu
  // Remaining tensors release automatically at scope exit.
}
```

GEMM does not download either operand or its result. Each operation synchronizes its stream before returning; asynchronous scheduling is not yet exposed. Uploads pack CPU logical views, downloads return independent CPU tensors, and host staging arenas close after each operation. The default 1GiB budget counts live device allocations (one byte for an empty allocation), not module/context overhead or transient host memory. Explicit tensor close frees budget for reuse and is idempotent on the owning thread.

All operations reject wrong-thread, closed or foreign-session handles before submitting native work. A scope may return a handle, but subsequent tensor operations reject it. Nested scopes retain independent ownership even on the same primary context. A changed current context rejects normal operations; cleanup rebinds the owned context before draining/freeing resources, then the driver scope restores the caller's prior context. See NVIDIA's [current-context semantics](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html).

Five real-GPU tests cover chained GEMM, strided upload, budget rejection/reuse, individual/scope/exception release, nested-session rejection, wrong-thread rejection, zero-inner GEMM, empty tensors and external context changes. The context-change regression first failed at shutdown and then passed after rebinding was added. Ordinary success and Scala exception cleanup are exercised; injected CUDA driver failures and exhaustive native leak accounting remain open. Bindings still resolve per call, and the existing simple PTX GEMM has no performance claim.
