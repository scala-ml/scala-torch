# VAE encoder and decoder graphs

`VaeEncoder` compresses NCHW images into raw posterior moments: the first half of the output channels represents the mean, the second half the log variance. It composes an input convolution, unconditioned residual down blocks, a residual-attention-residual middle block, output GroupNorm/SiLU/convolution and an optional channel-preserving 1x1 quant projection.

`VaeDecoder` reconstructs raw NCHW image values from unscaled latent tensors using an optional 1x1 post-quant projection, input convolution, middle block, residual up blocks and output GroupNorm/SiLU/convolution. Each upsampling stage doubles spatial dimensions with nearest resize followed by convolution. Both graphs are ordinary derived Modules with Float32 CPU autodiff.

`VaeAttention2d` applies image GroupNorm, flattens spatial coordinates into tokens, performs projected dense self-attention, restores the image layout and adds the original image residual. The middle attention is optional. This attention has no context Transformer stack or cross-attention dependency.

Graph constructors reject timestep conditioning, channel mismatches, empty stage stacks, misplaced samplers, non-preserving boundary/quant convolutions and invalid moment channel counts. Before execution, each graph checks every intermediate image tensor against `maxFeatureElements` (default 134217728) and Int indexing, including residual hidden channels, bottom/right padding and resized upsampling buffers. Attention scores have their own existing limit. These are per-tensor limits, not a bound on total training graph or peak heap use.

Stride-two downsampling with bottom/right padding uses floor division. An odd 7x11 image in the tested two-stage encoder therefore produces 3x5 moments; decoding a corresponding latent produces 6x10. This is not the odd-size skip restoration used by the conditional U-Net. Tiny inputs that would produce a zero spatial dimension reject before execution. Empty batches remain supported, and impossible upsampling dimensions reject during preflight.

Five tests cover:
- Analytical rectangular image-attention layout and per-channel means across two batches.
- A complete image-to-moments-to-image computation, finite gradients for every parameter and image entry, and three joint directional derivatives perturbing the full graph.
- Frozen inference, odd dimensions, malformed topology and score/feature limits.
- Padding and resized intermediate allocations larger than input/final output, plus empty-batch and overflow-sized upsampling boundaries.
- Rejection of timestep-conditioned residuals and malformed quant/sampling structures; execution without quant projections or middle attention.

The affected diffusion suite, including public CLIP/U-Net checkpoints, passes **47 tests, 0 failures, 0 skips**. Read-only review found no correctness blockers. No existing core/autograd kernels changed; the latest all-module public/CUDA run remains the prior 312-test release-preparation run.

## Pretrained interoperability

The VAE files from `hf-internal-testing/tiny-sd-pipe`, revision `39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`, have been downloaded under ignored `.superpowers/models/tiny-sd/vae`:

- `config.json`: 480 bytes, SHA256 `e5d2dc3a363761f93349b5751db4526015420d4b76c8cbde4d0201d4bfcb04c1`.
- `diffusion_pytorch_model.safetensors`: 2646532 bytes, SHA256 `7999b516094ac4239121d9f62a37e6be99620e7016220011bb99cf8b1a391aad`.
- Config uses channels32/64, one residual layer per encoder stage, groups32, latent width4, image width3 and pipeline scaling factor0.18215.

These files are now loaded and independently compared in the [pretrained VAE gate](pretrained-vae.md). Six encoder/decoder cases agree within3e-6; the expanded affected public diffusion suite passes57 tests. Posterior log-variance clamping, sampling/KL, image normalization, diffusion latent scaling, pixel conversion, schedulers and prompt processing belong to subsequent integration; this API does not silently implement them. Complete image generation and Python-free NUI remain unfinished.

The verified Maven preparation commit `10878ca` remains available. This new graph is a subsequent implementation slice and does not claim that any Central artifact has been uploaded.
