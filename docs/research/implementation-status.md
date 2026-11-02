# Implementation evidence

Updated 2026-11-02. The design was approved by the user. Subsequent explicit requirements set Scala to **3.3.8 LTS**, sbt to **2.x** (pinned to 2.0.10), and communication to English. These supersede the initial 3.3.7 candidate in historical research notes.

## Verified checkpoint

Full verification: **18 passed, 0 failed, 0 skipped** (core 10, CPU 5, FFM 3), with the CUDA test explicitly enabled. Run the three test commands from README; enable CUDA with the documented `Test / envVars` command. Scala's runtime dependencies still emit an `Unsafe::objectFieldOffset` deprecation warning on Java 25; tests pass, but warning-free runtime compatibility is not claimed.

- Java 25.0.1, Scala 3.3.8, sbt 2.0.10 compile the core, CPU, and FFM modules.
- Storage tests cover defensive copies, scalar and empty shapes, negative dimensions, checked Long size overflow, and primitive Float32/Float64/Int64/Bool preservation.
- View tests cover transpose aliasing, contiguous conversion, reshape validation, scalar/singleton/empty broadcasting, stepped and reversed slices, and layout bounds/overflow.
- Read-only review identified unused stride overflow for enormous empty dimensions. A regression test reproduced it; empty layouts now use safe zero strides and the full suite passes. No other review findings remain for this checkpoint.
- CPU tests cover broadcast add, reductions (including empty inputs), rank-2 GEMM against known results, numscala GEMM on transpose/slice inputs, NaN propagation, and oversized empty reduction rejection.
- Native host scope tests cover memory lifetime, idempotent close, and exception cleanup.
- Real CUDA copy test passed on the GTX 1650 SUPER using `libcuda.so.1` and Java FFM: allocate, host-to-device copy, device-to-host copy, and release. No Python, JNI shim, or CUDA Toolkit is needed for this particular driver-only probe. This does **not** prove GPU GEMM or a CUDA Tensor backend.

Each feature suite was first run before its API existed and failed compilation, then passed after implementation. The oversized-empty reduction regression specifically failed with NegativeArraySizeException before the guard was added.

## Execution decisions

- Initialized this previously unversioned directory on `feat/tensor-foundation`; no pre-existing branch or implementation needed an isolated linked worktree.
- Kept `Ops` as the core reference implementation and `NumscalaBackend` explicit in the CPU module, avoiding a core-to-CPU dependency cycle.
- Continued independent Tensor work while leaving the full NUI feature audit unfinished. M0 is not complete.
- sbt 2 joins CLI task arguments differently from sbt 1. Use individual commands or a semicolon-delimited sequence. Verify reported test counts; a zero-test run is not evidence.
- A resident sbt server did not pick up the client's CUDA environment flag. The documented `Test / envVars` command explicitly enables the forked real-GPU test. Default CPU-only execution skips it and must report that skip.

## Remaining gates

M0: full source audit and feature ledger. M1: allocation/performance measurements, broader numerical fixtures, and all planned native probes. M2–M6: autograd, modules/optimizers, safetensors/tokenizers, LLM inference and tuning, CUDA GEMM/streams/events, diffusion models, and complete NUI replacement. Native BLAS/cuBLAS were not found in the inspected library locations; driver access alone is insufficient for those gates.

The full project goal remains active and unachieved.
