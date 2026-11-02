# Native img2img pipeline foundation

EulerImagePipeline.generate/generateDpm append optional DiffusionImageSource.
Its image must match request dimensions and VAE input channels in NCHW Float32,
with finite values in[-1,1]. Strength must be finite0..1. The active schedule is
the suffix of max(1,floor(totalSteps*strength)) steps, matching existing NUI
python/sdfx/params.py img2img_steps (including the minimum-one-step behavior).
Progress counts only those active steps. No input-file resizing is implicit.

VAE encoder follows VaeExecution (CPU or CUDA). The diagonal posterior is sampled on CPU, then scaled
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

The worker now accepts img2img, but complete NUI img2img support is not established.
Remaining work: NUI cross-repository request/metadata/replay acceptance;
practical full-size performance; real
SD1.5/SDXL acceptance and inpainting. The complete migration goal remains open.
## RGB resizing foundation

DiffusionImageResize implements Lanczos3 with normalized22-bit coefficients,
per-axis RGB clipping, stretch/center-crop/edge-fill geometry. Twenty-one RGB
fixtures from NUI images.resize_image with Pillow13.0.0 match byte-for-byte.
The7 geometries cover up/down/mixed scaling, portrait/landscape, one-pixel axes,
and unchanged dimensions, across modes0/1/2. Invalid mode/size/budget reject.
Input and intermediate pixel counts are bounded. EXIF/alpha handling runs in
DiffusionImageInput before resizing.

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

## Worker image input

NativeDiffusionBackend accepts the NUI img2img envelope and nested Img2ImgParams.
It decodes the source once per request, applies EXIF orientation and RGB conversion,
resizes with mode0/1/2, then normalizes RGB to Float32 NCHW using channel/255*2-1.
Each iteration samples the VAE posterior and diffusion noise with its own seed.
The shared generation path retains previews, active-step progress, staging,
publication cancellation, decode recovery, and model keyword processing.
Active inpainting settings and masks reject explicitly; txt2img still rejects
active img2img parameters. The default encoder remains CPU.

DiffusionImageInput uses JDK ImageIO, with a 256MiB encoded-file limit and
16,777,216-pixel decoded limit checked before raster decoding. EXIF parsing is
bounded to 1MiB metadata and supports JPEG APP1, PNG eXIf and TIFF headers.
RGBA/palette alpha composites over white. L/LA use direct gray samples; NUI's
load_rgb discards LA alpha. Twenty-seven checked-in fixtures cover orientations
1..8 in PNG/JPEG/TIFF and L/LA/palette transparency. RGB bytes match NUI load_rgb
with Pillow13.0.0 exactly for these fixtures; this does not prove every codec,
color profile, bit depth or older Pillow version is identical. Malformed files,
out-of-bounds EXIF and allocation limits are tested. The optional oracle script
scripts/reference_image_input.py takes the NUI repo path; normal tests and
runtime do not invoke Python. Source provenance is the same images.py hash above.

## CUDA encoder

CudaVaeEncoder returns raw mean/log-variance moments using the shared diffusion
runtime: conv, residual blocks, asymmetric zero-padded downsampling, middle
attention and optional quant projection. CPU and CUDA use the same allocation
preflight. GPU activation storage stays resident through encoding; host/device
transfer happens at input and raw moments. The encoder session closes before
U-Net sampling, and VAE decoding opens its own session afterwards.
SDFX_SCALATORCH_VAE_DEVICE, VAE_MAX_BYTES and VAE_OFFLOAD_WEIGHTS now govern both
encoder and decoder. Defaults remain CPU. Posterior sampling, CLIP and scheduler
remain CPU, so this is hybrid execution with no Python runtime dependency.

Independent Candle encoder fixtures cover even/odd rectangular inputs and batch2.
Resident and offloaded CUDA moments have max absolute error9.54e-7 (limit2e-5).
Tests also verify cancellation cleanup/reuse, invalid channels, allocation failure,
omitted middle attention/quant projection, and CPU/CUDA end-to-end img2img parity.
The existing VAE decoder and pretrained VAE suites passed with this change.
NUI NativeScalaEngineSuite passed with CUDA U-Net and VAE, offload enabled,
16MiB per session: actual typed64x64 txt2img and img2img requests in6.354s.
That timing includes the tiny-model integration suite and concurrent SD1.5 work;
it is not a clean performance benchmark or normal-model acceptance.

NativeImg2ImgSuite exercises real tiny checkpoint weights through the worker API:
Euler/DPM++ generation, seeded replay, differing resize modes, strength-based
progress, invalid paths/strength/masks, and publication-denial cleanup. This is
not real SD1.5/SDXL image-quality or full desktop acceptance.
