# Persistent CUDA execution

Continue the approved native backend roadmap inline. Existing CPU Tensor storage is array-backed; preserve its contracts while building the device resource layer used by future backend dispatch.

1. Add a thread-confined `CudaSession.withDevice` scope and explicit owned `CudaTensor` Float32 allocations. Upload logical CPU views, retain device buffers across operations, explicitly download, free individual tensors, and release all remaining resources on scope exit. Enforce per-session byte budgets and reject closed, foreign-session and wrong-thread use before native calls.
2. Reuse one module and stream per session for packed rank-2 GEMM. Keep output on device for subsequent GEMMs. Initially synchronize each operation so freeing buffers has straightforward lifetime semantics. Use operation-local host arenas to prevent staging-memory accumulation.
3. Real GPU tests: chained GEMM vs CPU, strided upload, empty/zero-inner dimensions, budget rollback/reuse, close/escape/exception cleanup, session and thread isolation. Run full public-model and CUDA gates and request review.
4. Follow-on: integrate device storage with general tensor/backend dispatch, implement model kernels and CUDA execution, cuBLAS/performance evaluation, then diffusion/NUI gates. The scoped low-level API alone does not complete those requirements.

Use the existing Linux64 Java25 FFM driver path; no Python, JNI shim or toolkit dependency. Synchronous correctness first; no asynchronous scheduling or performance claims.
