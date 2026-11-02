# Image-space diffusion Transformer

`DiffusionSpatialTransformer(norm,projIn,blocks,projOut)` connects NCHW image features to the token Transformer. It applies GroupNorm and a1x1 input convolution, reshapes/transposes to (batch,height*width,inner), applies a nonempty Transformer stack with shared cross-attention context, then restores NCHW, projects back through a1x1 convolution and adds the original image.

Constructor checks enforce projection geometry/channel restoration, stack feature widths and a common context width. Calls validate image rank/channels and context batch/width. All parameters remain ordinary derived nested Modules.

Three new tests cover a rectangular image with identity token blocks and analytical normalized image values, every image/context/parameter derivative of a two-batch spatial Transformer against central differences, zero-output residual identity/gradient and invalid stack/projection geometry. The spatial plus token Transformer/residual/timestep/attention component suite passes15 tests with no skips.

This additive image wrapper does not change existing kernels. The full public/CUDA affected regression passed at the preceding token Transformer milestone; it was not repeated for this wrapper. Pretrained config/weight mapping and independent spatial-block comparison are still needed. Linear-projection variants, masks/adaptive norms, GPU execution, full U-Net/VAE and all NUI application migration remain open.
