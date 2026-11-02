# GroupNorm for diffusion

Continue inline within approved M5 diffusion/NUI scope.

1. Add CPU Float32 GroupNorm reference on (N,C,...) tensors: group channels per sample, normalize over each group's channels and spatial axes, biased variance (divide by group size), finite positive epsilon. Use Double means/centered variance; allow empty batches but reject empty channel/spatial dimensions.
2. Expose Variable.groupNorm(groups,epsilon) and a derived nn.GroupNorm with optional channel scale/bias and identity initialization. Existing broadcast autograd handles affine derivatives; a dedicated normalization pullback handles the input.
3. Test analytical normalization, group/sample isolation, noncontiguous data, singleton/large finite groups, invalid geometry, all-parameter finite differences and noGrad/empty-batch behavior. Missing APIs must fail first.
4. Run affected regression tests and request read-only review; document limits and commit. CPU reference is groundwork; resident CUDA normalization and complete pretrained diffusion/NUI remain required.
