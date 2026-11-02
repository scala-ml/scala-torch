# Float32 inference backends

Continue the approved CUDA/model roadmap inline.

1. Add core `FloatBackend[T]`: upload/download, shape, release, rank-2 matmul, equal-shape add/multiply, SiLU, last-axis RMSNorm. Operands/results belong to their backend; results are independently owned and inputs are unchanged. Broadcasting is not implicit in this initial inference interface.
2. Implement `CpuFloatBackend` using existing Numscala GEMM and stable scalar reference arithmetic. CPU release is a no-op because storage is immutable and GC-managed.
3. Implement the same interface in `CudaSession`, using persistent device inputs/results, one additional PTX module for arithmetic/activation/normalization, and shared bounded synchronous launch machinery. No host round trips inside operations. SiLU uses stable sign branches and approximate base-2 exponential; RMSNorm uses Double accumulation, division and square root to avoid Float32 square overflow.
4. Test independently calculated outputs, CPU/CUDA numerical agreement including extreme values, empty shapes, invalid shapes/epsilon, ownership/budget rejection, and a common backend-generic gated MLP/residual computation. Use tolerance for transcendental results, not bitwise CPU equivalence.
5. Review and run full public-model and real-GPU suites. Follow-on remains model integration, missing attention/embedding/rotary kernels, full Tensor/autograd device dispatch, optimized kernels and NUI migration.

Files: core FloatBackend.scala; cpu CpuFloatBackend.scala and suite; ffm CudaSession.scala, transformer.ptx and CudaOpsSuite.scala. ffm tests may depend on the CPU module; production ffm remains core-only.
