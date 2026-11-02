# Published VAE interoperability

Continue inline under approved diffusion/NUI scope, after graph commit6098acc. Release metadata/credentials remain pending; do not infer answers.

1. Parse bounded strict AutoencoderKL config, including channel/stage/group counts, optional quant and middle-attention branches, and latent scaling metadata. Unsupported transforms/options must reject rather than be ignored.
2. Load all encoder/decoder/quant weights from the original Diffusers safetensors file. Validate total parameters, exact names/shapes/dtypes, missing/unused entries, and freeze by default. Support the two known attention naming conventions and singleton convolution projection weights explicitly.
3. Expose raw moments and unscaled decode through a derived VAE model. Keep posterior sampling/clamping and diffusion scaling outside this forward API.
4. Test synthetic malformed/optional/legacy archives and the pinned public124-tensor VAE. Compare decoder outputs and encoder mean/log-variance against independent Candle0.9.1. Retain source recipe, dependency lock and enforced hashes. Rust is test-fixture generation only, never a runtime/build dependency of Scala artifacts.
5. Verify affected public diffusion tests, independent review, documentation and commit. Then proceed to posterior sampling, schedulers and prompt-driven image generation; full NUI remains unfinished.

## Verification outcome

All five steps completed. The public VAE maps124 tensors and six independent
Candle cases agree within3e-6 at the unchanged2e-5 threshold. Candle batch
inconsistency is documented; fixture generation uses fresh independent rows.
Synthetic archive rejection, F16/BF16 conversion, optional branches and legacy
projections pass. Expanded public diffusion suite:57 passed,0failed,0skipped.
Read-only review found no correctness blockers. No shared kernels changed.
Posterior/scheduler/prompt integration and full NUI migration remain open.
