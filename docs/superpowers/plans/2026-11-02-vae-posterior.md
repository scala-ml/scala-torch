# VAE posterior and latent integration

Continue inline within the approved Python-free diffusion scope. Published VAE
parity is committed as9f08252. This step keeps the differentiable model API
usable for both inference and later VAE training.

1. Add CPU Variable exp and clamp with explicit inclusive-boundary derivatives,
   logical-view/empty/noGrad behavior and gradient checks. These are prerequisites
   for log-variance stabilization and reparameterization; no backend claim.
2. Build NCHW diagonal Gaussian posterior from channel-split moments, clamp
   log variance to[-30,20], expose mean/mode, variance/stddev, sampling with
   caller-supplied noise and seeded JVM noise, and per-image KL to standard normal.
   Keep gradients through mean/log variance and treat supplied noise as constant.
3. Add explicit model encode/posterior and diffusion-latent scaling/decode APIs,
   preserving raw encodeMoments/decode semantics. Test analytical values,
   reparameterization/KL gradients, seeded reproducibility, invalid shapes and
   integration with the pinned public VAE.
4. Run affected autograd/public diffusion regressions, read-only review,
   document behavior/limits and commit. Scheduler/tokenizer/image-pipeline work
   follows; NUI and the overall goal remain incomplete.

Reference semantics: Diffusers v0.35.1 models/autoencoders/vae.py,
DiagonalGaussianDistribution. Seed sequences are JVM-defined and do not claim
PyTorch generator compatibility; exact cross-implementation checks supply noise.

## Outcome

Completed all four steps. Missing-API tests failed before implementation;
primitive tests4passed and posterior/loader integration tests16passed.
Full public/CUDA regression337passed,0failed,0skipped (19/15/74/17/43/134/35).
Read-only review found no correctness blockers. Scheduler/prompt/image pipeline
and full NUI migration are subsequent work, not completed by this slice.
