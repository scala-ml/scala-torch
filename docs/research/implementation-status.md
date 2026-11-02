# Implementation evidence

Updated 2026-11-02. The design was approved by the user. Subsequent explicit requirements set Scala to **3.3.8 LTS**, sbt to **2.x** (pinned to 2.0.10), and communication to English. These supersede the initial 3.3.7 candidate in historical research notes.

## Verified checkpoint

Full verification: **50 passed, 0 failed, 0 skipped** (core 10, CPU 5, autograd 10, nn 20, FFM 5), with CUDA tests explicitly enabled. Run the module test commands from README; enable CUDA with the documented `Test / envVars` command. Scala's runtime dependencies still emit an `Unsafe::objectFieldOffset` deprecation warning on Java 25; tests pass, but warning-free runtime compatibility is not claimed.

- Java 25.0.1, Scala 3.3.8, sbt 2.0.10 compile the core, CPU, and FFM modules.
- Storage tests cover defensive copies, scalar and empty shapes, negative dimensions, checked Long size overflow, and primitive Float32/Float64/Int64/Bool preservation.
- View tests cover transpose aliasing, contiguous conversion, reshape validation, scalar/singleton/empty broadcasting, stepped and reversed slices, and layout bounds/overflow.
- Read-only review identified unused stride overflow for enormous empty dimensions. A regression test reproduced it; empty layouts now use safe zero strides and the full suite passes. No other review findings remain for this checkpoint.
- CPU tests cover broadcast add, reductions (including empty inputs), rank-2 GEMM against known results, numscala GEMM on transpose/slice inputs, NaN propagation, and oversized empty reduction rejection.
- Native host scope tests cover memory lifetime, idempotent close, and exception cleanup.
- Reverse autograd tests cover shared edges, leaf accumulation and zeroGrad, explicit VJP seeds, broadcast/empty reductions, transpose/reshape pullbacks, noGrad nesting/exception restoration/thread isolation, detach, and iterative traversal of a 12,000-operation graph. Central finite differences validate input/weight/bias gradients through GEMM+bias+ReLU+mean. A regression model fits `y=2x+1` from zero weights to MSE below `1e-6` in 200 manually applied gradient steps, without Python. This is a training integration test, not a finished Module/optimizer API or MLP gate.
- Autograd review caught inactive ReLU branches producing NaN for nonfinite upstream gradients. A failing regression reproduced it; conditional branch selection fixes it, and the full suite passes.
- The nn module verifies Scala-derived nested/optional/vector parameter names from outside the library package, tied-weight deduplication, seeded Linear initialization, conventional (out,in) weights, SGD and AdamW known updates, changing-gradient Adam moments, and rollback on invalid gradients. A derived 2→8→1 ReLU MLP learns XOR with AdamW to MSE below `0.001` in 500 steps and predictions within `0.05` of each target.
- Parameter updates preserve gradients while invalidating saved graphs. Review found frozen-weight transforms dropping invalidation lineage; a failing Linear regression reproduced it, and ordinary frozen transformations now retain lineage. Explicit detach/noGrad snapshots intentionally remain independent. State dict snapshots are immutable; strict key/shape/tied-alias validation precedes all loads.
- Persistent [training checkpoints](checkpoint-format.md) save weights, SGD/AdamW configuration and membership, immutable Adam moment/counter snapshots, tying and a caller-supplied step. AdamW uninterrupted 40-step regression matches 17 steps + save/load + 23 steps exactly. Tests cover missing slots, reordered optimizers, tied/subset parameters, raw Float32 bits, corrupt/truncated/trailing/oversized metadata, mismatched configurations, and failed atomic replacement cleanup. Recomputed-valid-checksum files with invalid counters/nonfinite/negative moments reject without changing values, gradients, or optimizer state.
- Real CUDA copy and GEMM tests passed on the GTX 1650 SUPER using `libcuda.so.1` and Java FFM. GEMM loads bundled PTX, launches on an owned stream, records and waits for an event, copies results, and releases resources after draining work. No Python, JNI shim, or CUDA Toolkit is needed for these driver-only probes. This does **not** prove a production CUDA Tensor backend or cuBLAS performance. See [native backend evidence](native-backends.md).

Each feature suite was first run before its API existed and failed compilation, then passed after implementation. The oversized-empty reduction regression specifically failed with NegativeArraySizeException before the guard was added.

## Execution decisions

- Initialized this previously unversioned directory on `feat/tensor-foundation`; no pre-existing branch or implementation needed an isolated linked worktree.
- Kept `Ops` as the core reference implementation and `NumscalaBackend` explicit in the CPU module, avoiding a core-to-CPU dependency cycle.
- Continued independent Tensor work while leaving the full NUI feature audit unfinished. M0 is not complete.
- sbt 2 joins CLI task arguments differently from sbt 1. Use individual commands or a semicolon-delimited sequence. Verify reported test counts; a zero-test run is not evidence.
- A resident sbt server did not pick up the client's CUDA environment flag. The documented `Test / envVars` command explicitly enables the forked real-GPU test. Default CPU-only execution skips it and must report that skip.

## Remaining gates

M0: full source audit and feature ledger. M1: allocation/performance measurements, broader numerical fixtures, CPU BLAS probe and complete native failure-path verification. M2: Embedding/norm/additional operations; stochastic or mid-gradient-accumulation checkpoint resume needs additional state beyond the verified deterministic step-boundary restart. M3–M6: safetensors/tokenizers, LLM inference and tuning, persistent CUDA backend/cuBLAS, diffusion models, and complete NUI replacement. Native BLAS/cuBLAS were not found in the inspected library locations; the simple PTX probe does not substitute for the production performance backend.

The full project goal remains active and unachieved.
