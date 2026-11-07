# NUI Euler Uniform schedule correction

NUI's current Euler sampler injects log-interpolated Uniform sigmas, regardless
of checkpoint timestep spacing. Add a separate native schedule constructor so
the reusable Diffusers-style scheduler remains unchanged. Use the NUI schedule
for both the decoded and latent-return Euler passes, including img2img and
Hires. Compare 1, 3, 4 and 9 steps against fixtures produced by NUI/Diffusers,
and check the actual worker output with a leading-spacing checkpoint. Keep
Euler a and DPM++ behavior unchanged.
