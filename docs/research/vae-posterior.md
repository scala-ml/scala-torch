# VAE posterior and diffusion latents

`VaeModel.encode(image)` produces a `VaePosterior`. The existing
`encodeMoments` and `decode` APIs retain their raw, unscaled behavior.
The posterior splits NCHW moments in half along channels, clamps log variance
to[-30,20], and exposes `mean`, `mode`, `logVariance`, `variance` and `stddev`.
`sample(noise)` applies differentiable mean + stddev * noise with caller-supplied
constant noise of exactly the latent shape. `kl()` returns one value per image,
summing the divergence to a standard normal over channels and spatial axes.

These formulas follow [Diffusers v0.35.1 DiagonalGaussianDistribution](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/vae.py#L633).
General posterior-to-posterior KL, negative log likelihood and a separate
zero-variance distribution mode are not implemented. `mode` returns the mean
of the ordinary distribution.

`sample(seed: Long)` uses `java.util.Random.nextGaussian`, converted to Float32.
Identical seeds reproduce the JVM sequence. This does not reproduce PyTorch's
seed sequence. Exact replay or comparison across implementations supplies the
noise tensor directly. No mutable global generator is used.

`model.toDiffusionLatents(rawLatents)` multiplies a sample or mean by the
checkpoint scaling factor. `model.decodeDiffusionLatents(latents)` applies the
inverse factor before raw decoding. Both APIs require matching NCHW latent
channels and positive spatial dimensions. These operations retain gradients.
For a normalized image `Variable`, the image-to-latent-to-image path is explicit:

```scala
val posterior = model.encode(image)
val latents = model.toDiffusionLatents(posterior.sample(42L))
val reconstruction = model.decodeDiffusionLatents(latents)
```

Images are still raw model values; image normalization/pixel conversion belong
to pipeline integration. Shifted/mean-normalized latent variants remain rejected
by the config parser.

CPU Variable `exp()` and `clamp(min,max)` provide the required differentiable
operations. Clamp passes the incoming gradient at both bounds and inside the
interval, and produces zero gradient outside it, even for nonfinite upstream
values. NaN bounds and reversed bounds reject. Float32 exponential overflow and
underflow retain IEEE behavior. Neither operation adds a CUDA backend kernel.

Tests cover analytical posterior values, per-image KL, reparameterization and
KL gradients, finite differences through clipped tails, empty batches, seeded
reproducibility, malformed/noise shapes, frozen/noGrad execution, explicit
scaling gradients and public VAE decode round trips with an exactly representable
scale. Primitive tests also cover logical transposes and clamp boundary gradients.

Full public/CUDA regression: **337 passed, 0 failed, 0 skipped**. Read-only review
found no correctness blockers. Schedulers, prompt tokenization,
complete image generation and Python-free NUI remain unfinished.
