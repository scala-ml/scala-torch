# Backend embedding and layouts

Continue the approved transformer GPU roadmap inline.

1. Extend FloatBackend with embedding(table, CPU Long IDs), reshape and transpose. CPU implementation preserves logical element order and uses immutable views where possible.
2. CUDA reshape retains a shared allocation with reference-counted handles. Individual handles are independently releasable; allocation budget is charged once and freed when the last handle closes. Scope exit releases all live handles.
3. CUDA transpose materializes packed output with a generic coordinate kernel, preserving the existing packed-kernel contracts. Embedding validates IDs on CPU, transfers integer indices, and gathers from the persistent GPU table. Temporary integer metadata is included in the session budget and released after synchronized execution; no activation download.
4. Test rank/scalar/empty embedding, repeated and invalid IDs, logical layouts, all axis pairs, view ownership in both close orders, scope cleanup, budget failures and zero-sized dimensions. Run public/real-GPU suites and review.

Files: FloatBackend.scala, CpuFloatBackend.scala and tests; CudaSession.scala allocation handles/metadata helpers; transformer.ptx layout/gather kernels; CudaLayoutSuite.scala. Follow-on rotary/attention/model integration remains required.
