# Resident LoRA inference

Extend the approved backend Llama work to execute existing trained adapters on CPU and CUDA, with no implicit activation transfers or GPU autograd claim.

1. Establish failing CPU/CUDA adapter parity tests with nonzero factors, batched cache decoding, stale-weight rejection and allocation cleanup.
2. Add scalar Float32 multiplication to both backends and verify IEEE values, views, empty tensors and ownership.
3. Upload transposed adapter factors with prepared weights. Execute `(x Aᵀ Bᵀ) * (alpha/rank)` before adding the base projection, preserving CPU operation order. Release temporary handles on success and failure.
4. Compare a checkpoint-restored, trained public SmolLM2 adapter against CPU full and cached logits on the real GPU.
5. Review, run the complete public-model and CUDA gates, and document the remaining GPU training and diffusion/NUI work.

Integration investigation: the trained-adapter gate exposed accumulated numerical differences beyond the existing `2e-4` logit tolerance. Prepared CPU results are exact. Investigate CUDA dot-product reduction order and transcendental approximation in isolation, retain the tolerance, and add primitive regression coverage for any kernel correction. Diagnostic host fallbacks must not remain in the implementation.
