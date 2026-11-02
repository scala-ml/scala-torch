# Backend autodiff implementation plan

Execute inline with the executing-plans workflow under the approved Scala ML design.

1. Add explicit backend `fill`, scalar `sum`, and `expandScalar` operations. CPU is the reference; CUDA uses owned device outputs and no activation downloads. Test empty/scalar/view inputs, IEEE behavior and allocation limits.
2. Add `autograd.BackendTape[T]`, a thread-confined closeable first-order graph over FloatBackend. It owns independent handles for imported leaves, graph outputs and leaf gradients. Support equal-shape add/multiply, scalar scaling, matrix multiplication, transpose/reshape and sum/mean. Reverse traversal accumulates shared edges; explicit seeds support nonscalar outputs. Backward stages leaf gradients before publication, and releases scratch on success/failure. Existing CPU Variable API remains available.
3. Validate analytic shared-edge gradients, finite differences, repeated backward, ownership and misuse. Add real CUDA tests through an autograd test-only FFM dependency.
4. Train linear regression on device using explicit SGD updates and fresh scoped tapes per iteration. Compare CPU gradients and final learned weights/loss; verify allocation returns to the resident input/weight baseline every iteration and zero after close. No CPU activation/gradient fallback.
5. Review and run all public-model/CUDA gates. Document limits: no GPU transformer backward, optimizer state/checkpoint integration, broadcasting or higher-order gradients yet. Those remain required follow-on work toward GPU LLM training and diffusion/NUI migration.
