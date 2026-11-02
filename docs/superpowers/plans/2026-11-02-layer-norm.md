# Token LayerNorm

Continue inline under approved diffusion Transformer scope.

1. Add Variable.layerNorm(epsilon) for the final positive dimension by reshaping logical rows and reusing validated GroupNorm(groups=1). Preserve rank-one, higher-rank, noncontiguous and empty-leading-dimension behavior.
2. Add derived nn.LayerNorm with configured width, optional per-feature scale/bias and identity initialization. Enforce width even with affine disabled.
3. Test analytical token isolation, strided/rank-one/empty inputs, all input/scale/bias derivatives, noGrad and stale parameter graph rejection.
4. Run affected autograd/nn regression with CUDA autograd enabled, review and commit. General multidimensional normalized shapes, resident CUDA kernels, pretrained diffusion and full NUI integration remain follow-ons.
