# CPU GroupNorm evidence

`GroupNormOps.forward(input, groups, epsilon)` normalizes CPU Float32 tensors shaped (N,C,...). Channels are split into equal groups independently for each sample; each group includes its channels and spatial elements. Statistics use Double accumulation with centered squared deviations and biased variance (divide by group size), then Float32 output. This follows the [PyTorch GroupNorm normalization convention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GroupNorm.html); it is not a binary PyTorch comparison.

`Variable.groupNorm` adds the analytical first-order input pullback. `nn.GroupNorm.init(channels, groups, epsilon = 1e-5, affine = true)` creates unit scale and zero bias; optional affine fields compose through existing broadcast autograd. The module validates its fixed input-channel count even without affine parameters. Derived traversal includes weight/bias and excludes scalar configuration.

Positive channels and spatial dimensions, groups dividing channels, finite positive epsilon and Int indexing are validated before allocation. Empty batches are valid. Singleton groups normalize to zero for finite values. Noncontiguous tensors are read in logical order. There are no running statistics or separate training/evaluation branches.

Four new tests cover independent sample/group statistics, analytic values distinguishing biased from unbiased variance, noncontiguous input, singleton groups, extreme finite values, invalid geometry/epsilon, empty gradients, affine initialization and all input/scale/bias finite differences. NoGrad behavior is also checked. Existing versioned graph semantics apply.

This is a CPU reference component required by diffusion U-Net/VAE, including the GroupNorm usage in pinned NUI's cd_tuner tests. Resident CUDA normalization, optimized performance, pretrained model weight mapping, diffusion image generation and full NUI feature migration remain open.
