# Native Euler a sampler

Implement NUI's Euler ancestral sampler with explicit per-step Gaussian noise,
independent numerical fixtures and deterministic per-request seeds. Reuse the
current Euler model-input scaling and image/mask preparation, with the
ancestral sigma-down/sigma-up update. Connect txt2img, img2img, latent Hires and
img2img Hires Fix, including progress, cancellation and metadata. Support the
NUI Uniform schedule and reproduce its local Diffusers Euler a fallback for
Karras to checkpoint timestep spacing with an explicit warning; reject unsupported prediction modes. Verify
CPU/CUDA small models and a real packaged NUI request without Python.
