# Euler scheduling and token-conditioned denoising

`DiffusionNoiseSchedule` constructs immutable training coefficients from explicit
betas or linear/scaled-linear beta schedules. Cumulative alpha and sigma use
Double arithmetic. `euler` selects linspace, leading or trailing timesteps and
linearly interpolates training sigmas, then appends a terminal zero. Leading
spacing uses sqrt(maxSigma²+1) as initial noise scale; the other two use maxSigma.
An explicit `EulerSchedule` accepts model timesteps and sigmas for future custom
schedule integration. It does not infer or verify their model-specific mapping.

`EulerScheduler` is a stateless CPU Float32 computation built from Variables:

- `scaleModelInput(index,sample)` divides by sqrt(sigma²+1).
- `addNoise(index,clean,noise)` produces clean + sigma * noise.
- `step(index,prediction,sample)` returns the next sample and predicted clean
  sample using epsilon, sample or velocity prediction.

The caller passes a step index and separately supplies `schedule.timesteps(index)`
to the U-Net. No hidden cursor is advanced, so retries and branches do not share
mutable scheduler state. These operations retain gradients; frozen inputs remain
frozen. Exact shapes and per-tensor element budgets are checked before arithmetic.
This is not a peak-memory budget or a CUDA scheduler implementation.

Arithmetic follows [Diffusers v0.35.1 EulerDiscreteScheduler](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_euler_discrete.py).
The Double training coefficient calculation does not promise bit equality with
Torch's Float32 training schedule. Tests establish analytical scalar/vector
results, timestep rounding and interpolation, full constant-epsilon trajectories,
gradients, empty tensors and rejection boundaries. No independent end-to-end
scheduler/pipeline numerical parity claim is made.

The public integration test pins CLIP, U-Net and VAE config/weight hashes for
`hf-internal-testing/tiny-sd-pipe` revision
`39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`. It passes token IDs through CLIP,
performs four Euler updates of a 1x4x3x5 latent using the public U-Net, then decodes
to a finite 1x3x6x10 image tensor through the VAE with explicit inverse latent
scaling. Identical initial values/tokens reproduce exact output; changing tokens
changes output. Initial values are deterministic sine data for the regression,
not a standard-normal generation prior. This tests wiring, not image quality.

Read-only review found no correctness blockers. Affected public diffusion
regression: **68 passed, 0 failed, 0 skipped**. Shared kernels did not change;
the preceding all-module public/CUDA337-test result remains the latest full run.
Churn, Euler ancestral, checkpoint scheduler-config parsing, classifier-free
guidance, alternate sigma schedules and the remaining NUI sampler inventory
(including its default DPM++2M/Karras) remain open. Natural-language tokenization,
image file output, complete image workflows and Python-free NUI remain unfinished.
