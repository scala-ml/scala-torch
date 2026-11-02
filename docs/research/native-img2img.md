# Native img2img pipeline foundation

EulerImagePipeline.generate/generateDpm append optional DiffusionImageSource.
Its image must match request dimensions and VAE input channels in NCHW Float32,
with finite values in[-1,1]. Strength must be finite0..1. The active schedule is
the suffix of max(1,floor(totalSteps*strength)) steps, matching existing NUI
python/sdfx/params.py img2img_steps (including the minimum-one-step behavior).
Progress counts only those active steps. No input-file resizing is implicit.

VAE encoder currently runs on CPU. The diagonal posterior is sampled, then scaled
once into diffusion latents. Optional posteriorNoise and existing diffusion noise
allow exact replay; absent noise uses one java.util.Random seeded from request,
posterior draws before diffusion draws. This does not reproduce Torch RNG.
Euler starts at clean+sigma*noise. VP DPM starts at alpha*clean+sigmaVP*noise,
where alpha=1/hypot(1,sigmaRatio),sigmaVP=alpha*sigmaRatio. DPM history starts empty
at the suffix. U-Net/VAE decoder placement remains selectable, including CUDA
weight offload. Existing VAE decode-failure recovery still wraps the exact input.

Behavior references inspected locally: NUI img2img_steps, installed Diffusers
pipeline_stable_diffusion_img2img.py get_timesteps/prepare_latents, and
scheduling_dpmsolver_multistep.py add_noise. Native implementation uses existing
Scala tensors, VAE posterior and schedulers, with no Python runtime calls.

This is library plumbing, not complete NUI img2img support. Remaining work:
worker image path/format/bounds validation and preprocessing; NUI request routing
and metadata/replay; CUDA VAE encode for practical full-size performance; real
SD1.5/SDXL acceptance and inpainting. The complete migration goal remains open.
## RGB resizing foundation

DiffusionImageResize implements Lanczos3 with normalized22-bit coefficients,
per-axis RGB clipping, stretch/center-crop/edge-fill geometry. Twenty-one RGB
fixtures from NUI images.resize_image with Pillow13.0.0 match byte-for-byte.
The7 geometries cover up/down/mixed scaling, portrait/landscape, one-pixel axes,
and unchanged dimensions, across modes0/1/2. Invalid mode/size/budget reject.
Input and intermediate pixel counts are bounded. EXIF/alpha handling is not part
of this component and remains needed before worker img2img can accept image files.

Two mismatches were diagnosed against the independent oracle: edge fill's
zero-size box uses bicubic extrapolation (9/8 edge,-1/8 adjacent), not simple
edge repetition; Pillow13 may execute vertical resampling first when its pixel
reduction exceeds twice horizontal reduction. Preserve this order because RGB
clipping between passes changes values. Behavior source checked:
https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
Implementation was written independently from the filter formula and oracle;
no upstream code or new dependency was copied into the native runtime.

Reproduce fixtures using scripts/reference_image_resize.py with NUI repo path
and Pillow13.0.0. NUI python/sdfx/images.py SHA256 at generation:
c06acef02c883518f658c49c34bff19284f75801a433f7faedece9de7a0e44d0.
The Python script is an optional reference generator; checked-in RGB fixtures
allow Scala-only tests. This does not assert parity with older Pillow versions.