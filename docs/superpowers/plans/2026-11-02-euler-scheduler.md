# Euler diffusion schedule and update

Continue inline under approved diffusion/NUI migration. NUI currently defaults
to DPM++2M/Karras and also exposes Euler/Euler ancestral. Start with deterministic
Euler as the common noise-space/input-scaling foundation; this does not replace
or remove the remaining sampler inventory.

1. Add immutable beta-derived training noise schedule with explicit betas and
   linear/scaled-linear factories, bounded sizes and finite representability.
   Build Euler schedules with linspace/leading/trailing timesteps and linear
   sigma interpolation, or accept explicit timestep/sigma pairs with terminal0.
2. Implement stateless indexed CPU Euler input scaling, noise addition and step
   for epsilon/sample/v prediction, returning both next and denoised samples.
   Validate shapes/indices/budgets before operations. Use Variable composition
   to retain gradients and allow repeated/forked schedules without hidden cursors.
3. Analytical tests cover schedule values, timestep rounding/offsets, all three
   prediction conventions, complete constant-noise denoising, gradient and
   invalid/empty/budget cases. Connect published CLIP/U-Net/VAE to a small
   token-conditioned denoising loop, preserving frozen inference.
4. Verify affected suites, review, document current precision/limits and commit.
   Churn/ancestral updates, alternate sigma schedules, DPM++2M/other inventory,
   checkpoint scheduler config parsing, text tokenization, pixels and NUI follow.

Primary arithmetic reference: Diffusers v0.35.1 scheduling_euler_discrete.py.
Training schedule coefficients use Double accumulation; exact Torch Float32
schedule bit patterns are not claimed. Runtime tensors remain CPU Float32.

## Outcome

Implemented all four steps. Targeted scheduler/public-loader tests17passed;
expanded public diffusion regression68passed,0failed,0skipped. Read-only review
found no correctness blockers. Independent end-to-end scheduler parity is not
claimed; analytical equations and public model integration are the current gates.
Remaining samplers, natural-language/pixel integration and full NUI remain open.
