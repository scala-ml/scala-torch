# Image-space diffusion Transformer

Continue inline under approved M5.

1. Add DiffusionSpatialTransformer with GroupNorm, 1x1 input projection, nonempty token Transformer stack, 1x1 output projection and image residual. Validate widths and shape-preserving projection geometry.
2. Convert NCHW to (B,H*W,inner) and back without mixing batch/channel/spatial positions; use existing differentiable reshape/transpose.
3. Test rectangular images and analytical normalized identity-stack behavior, full image/context/parameter finite differences, zero-output residual identity and invalid geometry/stack widths.
4. Run targeted spatial/token/residual suites and model compile checks, review and commit. This additive composition does not change existing kernels; full public/CUDA model regression already passed at the prior milestone. Pretrained fixture/config mapping and full U-Net/VAE/NUI remain open.
