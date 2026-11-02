# Final-axis LayerNorm

`Variable.layerNorm(epsilon)` normalizes each logical final-axis row independently with biased variance. It reshapes the tensor into (rows,width), reuses the validated GroupNorm(groups=1) reference and pullback, and restores the original shape. This avoids another normalization kernel and preserves existing Double statistics and autograd graph/version semantics.

`nn.LayerNorm.init(width,epsilon=1e-5,affine=true)` initializes unit scale and zero bias. Explicit optional scale/bias fields allow either affine component to be omitted. Width remains enforced with affine disabled. Parameter traversal is derived, and per-feature affine parameters broadcast across leading dimensions.

Tests cover analytical values distinguishing token normalization from batch/channel normalization; noncontiguous and rank-one input; central differences for every input/scale/bias element; noGrad; empty leading dimensions and zero affine gradients; malformed dimensions/epsilon; and stale parameter graph rejection.

This is CPU Float32 final-axis normalization for Transformer token features. General multidimensional normalized shapes, backend-resident GPU LayerNorm, pretrained Transformer/diffusion fixtures and complete U-Net/VAE/NUI execution remain open.
