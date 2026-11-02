# Diffusion residual block

`models.DiffusionResnet2d` composes existing differentiable CPU operations:

1. GroupNorm → SiLU → 3×3 convolution.
2. Optional SiLU-activated timestep embedding → Linear projection → channel broadcast/add.
3. GroupNorm → SiLU → 3×3 convolution.
4. Add the input (or a 1×1 channel-changing shortcut), then divide by output scale.

This is the default additive conditioning order used by [diffusers ResnetBlock2D](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py). The implementation composes scala-torch's own primitives and has no Python runtime dependency. Weights are supplied explicitly; pretrained checkpoint mapping is not yet implemented.

The constructor validates convolution kernel/stride/padding/dilation/groups, normalization channels, shortcut requirements, timestep projection width and a representable positive reciprocal output scale. Calls require rank-four NCHW inputs and matching timestep batch/width. Supplying an embedding to an unconditioned block or omitting it from a conditioned block rejects explicitly. `Module` derivation traverses nested weights including optional time projection and shortcut.

Tests check all input, timestep and parameter gradients against finite differences on a two-batch channel-changing block. A zero residual analytically preserves the scaled shortcut and its gradient. A separate identity-center-kernel example verifies timestep addition precedes the second normalization/activation. Invalid construction/conditioning and noGrad are covered.

This component supports zero dropout, SiLU and additive timestep conditioning only. Scale-shift/adaptive normalization, other activations, integrated up/downsampling and stochastic dropout require explicit implementations before loading configurations that request them. GPU execution, independent pretrained block comparison, full U-Net/VAE and complete NUI migration remain open.
