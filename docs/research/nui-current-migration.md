# Current local NUI: Python-removal target

## SD1 single-file progress (2026-11-03)

The native worker now reads the common SD1 LDM `.safetensors` layout directly.
The public SD1.5 pruned checkpoint (SHA-256
`6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa`)
loaded through NUI AppServices/Generator with no Python installation; the native
worker also generated a 256x256, 12-step image depicting the requested apple.
The extracted native-only package's MCP route generated a 64x64 image from the
same checkpoint using its bundled Java, worker and tokenizer. The NUI native
branch has separate file and folder selectors and Python remains selectable in
the normal dual-provider build.

The tested single-file path is SD1 only. SD2, SDXL and inpainting single-file
checkpoints are rejected; their Diffusers-directory paths remain separate.
The checkpoint's tokenizer is not embedded in the checkpoint and must be
supplied with the app or configured separately. The worker implements a subset
of NUI requests and hooks: in particular its hello currently advertises no
features, while the Python engine advertises many extension capabilities.
This prevents feature-gated plugins from being treated as available and is a
separate gap from the model loading path. Matching each advertised feature to
its actual behavior and porting missing requests remains necessary before
Python removal can be considered complete.

On 2026-11-03 the user explicitly identified `/home/mizushima/dev/stablediffusion-nui`
as the current application and confirmed that removing its Python dependency is
the immediate goal. The broader Scala LLM/framework work remains useful, but
library milestones alone do not satisfy this application goal. Historical issue
batches/releases in NUI handoff files are context, not replacement objectives.

The inspected local checkout is clean at
`d9f13ea72409ea543948bacb5807536ba6dc4f83`. Read `AGENTS.md`, `docs/HANDOFF.md`,
`HANDSOFF.md` and the current `HANDOFF-CODEX.md` before modifying the application.
Existing partial features and unmerged branches must not be called complete or
silently imported. No NUI product source was modified during this inventory update.

## Refreshed static inventory

The [current snapshot](nui-current/inventory-summary.json) supersedes the older
8823aa4 snapshot as the target baseline; historical tables remain intact.

| Surface | Previous snapshot | Current local target |
|---|---:|---:|
| Worker requests | 47 | 48 (42 queued,6 reader/control) |
| Standard plugins | 117 | 121 (35 scripts,16 extras,70 AlwaysOn) |
| Parameter fields | 97 | 100 |
| Python package modules | 135 | 138 |
| Python source lines | 34628 | 36086 |
| Scala request references | 254 | 263 |

Added request: `cn_animation`. Added registered plugins: `clip_io`, `cn_animation`,
`compile`, `corridor_crawler`. Added Python modules: `clip_io.py`, `cn_animation.py`,
`compile_accel.py`. `PreprocessRequest` gained `standalone`, `processor_res` and
`pixel_perfect`. These extend the migration requirements; they are not optional
because the older inventory did not contain them.

The current request/plugin/module/parameter/reference CSVs are under
[the current inventory directory](nui-current/requests.csv). Extraction used static
AST/text inspection, without importing the Python worker. Unique request/plugin
IDs, complete package file enumeration and every recorded module hash were
independently checked against the local tree. All48 handlers and121 registered
plugin classes/IDs resolved. All migration/parity rows remain open. Static counts
are not behavioral verification or evidence of native implementation.

## Concrete application boundary

`engine/.../AppServices.scala` still constructs the supervisor with
`PythonEnv.engineLaunch(home, settings, dryRun)`. EngineProcess launches that
command and exchanges protocol-v1 JSON Lines. The JVM replacement must preserve
request IDs, event/error/result schemas, cancellation ownership, queue admission,
interrupt protection, lifecycle/recovery and publication behavior. Existing
JavaFX/MCP consumers should use the same application services.

## Verified native progress (2026-11-03)

The feature branch now exercises the worker through NUI `AppServices`,
`Generator`, `EngineClient` and the child-JVM launch boundary. Tiny SD1 coverage
includes text/image generation, masks, standard linear LoRA and safetensors
textual inversion. The scala-torch worker also has tiny SDXL coverage for
dual-encoder conditioning, paired `clip_l`/`clip_g` textual inversion and unload
reset. NUI `core`, `ui`, `plugins` and `mcp` still have no scala-torch model or
tensor dependency; `engine` owns backend selection and launching, while model
JARs are supplied only on the child JVM classpath.

The SDXL NUI Generator integration ran with saved CUDA placement on the local
GTX 1650 SUPER (4 GiB): paired textual inversion changed output and remained
active after its catalog path was removed, and masked replay passed in 6.274s.
These checks prove selected request-to-worker paths, not full feature parity or
Python removal. Native textual inversion supports safetensors and the common
restricted ZIP `.pt` layouts (tensor, `emb_params`, `string_to_param["*"]`, and
SDXL `clip_l`/`clip_g` tensors); unsupported globals, views and older `.pt`
archives remain explicit warnings. Tiny SD1 native latent Hires now runs two passes and returns a final-size
PNG through NUI `Generator`; its six latent interpolation modes have PyTorch
golden-vector coverage. Tiny SDXL Hires also preserves original-size/crop
conditioning. Hires remains scoped to NUI's `engine` adapter and the external
scala-torch worker: core/UI store only generic settings and protocol values, and
NUI's SBT graph/assembly contains no PyTorch or scala-torch model/tensor jars.
Embedding inspection/edit/merge, resource-hash parity, SD2, broad
ControlNet/plugin coverage, normal-model quality/performance, packaging and
Python-absent JavaFX/MCP acceptance remain open. Txt2img Hires supports six
latent upscalers. The separate img2img Hires Fix supports built-in
Lanczos/Nearest pixel resamplers and a CUDA RRDBNet x4+ route for the default
R-ESRGAN model. The loader accepts bounded PyTorch ZIP `.pth`/`.pt` and
`.safetensors` state dictionaries; NUI `engine` resolves the display label to
the official `RealESRGAN_x4plus` filename and passes its path only to the
child worker. A standalone scala-torch CUDA smoke with the official `.pth`
checkpoint converted a synthetic 16x16 image to 64x64 in 1.73s on the local
GTX 1650 SUPER (4 GiB). Real checkpoint loading and CUDA output are verified;
larger-tile seams, perceptual quality and the full NUI request path remain
unverified across models and image sizes. Native-only NUI has since completed
both standalone Extras and txt2img Hires Fix using RealESRGAN x4plus / Anime
6B. Other upscaler architectures remain unsupported. Refinement LoRA
switching, ControlNet replay, trained nine-channel quality, per-pass hooks and
masked latent img2img/inpaint quality remain open. Unmasked img2img can use the
same latent Hires second pass by returning the base pass's denoised latents;
it avoids an intermediate VAE decode/encode. A direct CUDA worker smoke with
the pinned 9 MB Narsil tiny checkpoint requested 64x64 img2img at two base
steps/0.5 strength (one active step), followed by 96x96 latent Hires at two
second-pass steps. It returned one 96x96 PNG in 2.35s. This verifies direct
worker protocol execution and CUDA placement, not normal-model quality.

A fresh native-only packaged NUI run also exercised the masked `latent_upscale`
script with the real SD1.5 checkpoint on CUDA: 512x512 source, 640x640 output,
binary elliptical mask, 0.5 denoise and four requested steps. It produced a
640x640 PNG without Python; sampled device memory was 1765-2079 MiB, not peak
telemetry. The no-blur sample showed a visible mask boundary. This closes only
the packaged runtime-path gap; mask-edge behavior and image quality/parity
remain unverified.

The same packaged request was repeated with mask blur 8 and full-resolution
masked crop/paste (padding 32). The output preserved the 512x512 canvas. A
pixel comparison found zero changes in the 159,371 pixels at least roughly
37-47px outside the ellipse; changes were confined to the mask and its blur
boundary. This verifies unmasked-region preservation for one real SD1.5 case,
not Python output parity or broad mask quality.

The existing NUI `NativeHiresGeneratorSuite` was also rerun against the same
pinned model using the relative worker-distribution classpath. NUI
`AppServices`/`Generator` launched the CPU child JVM, completed txt2img latent
Hires, and passed the no-Python handshake and no-venv assertions in 42.895s.
The suite now also sends the generated image back through NUI `Generator` as
unmasked img2img with latent Hires enabled; `Generator.prepare` retained the
Hires settings and the final PNG was 128x128. This NUI CPU integration passed
in 75.849s with the no-Python assertions. Direct CUDA worker smoke covers the
same img2img Hires path on CUDA; normal-model quality is still unverified.

Native VAE tiling is now implemented for both encode and decode. A direct CUDA
worker smoke on the pinned tiny SD1 model used 256x256 img2img, activating 3x3
64px latent tiles for VAE encode and decode; it returned one 256x256 PNG in
10.07s with CUDA U-Net/VAE offload, no worker warnings, and `python:""` in the
handshake. The output is intentionally noise-like because the tiny random
checkpoint is for execution coverage, not image quality. Artifact:
`stablediffusion-nui/.superpowers/native-img2img-hires-smoke/vae-tiling-cuda-fixed/native-b6b4e1fa-da62-4b86-93f6-93b8c76cef9b.png`
(SHA256 `C810C006029C4E136DE2B95B6895615A16E96B2991A65716C114C16CB1B3EB51`).

NUI's assembled MCP stdio server also ran a real `txt2img` call from a separate
home with no `python/` directory. It loaded the same tiny SD1 checkpoint through
saved model settings, reported `scala-torch-0.1.0 device=cuda:0`, returned a
64x64 PNG at seed 42, and exited successfully after stdin closed. The artifact
is `stablediffusion-nui/.superpowers/native-mcp-no-python/outputs/txt2img/2026-11-03/00000-42.png`.
This proves MCP protocol-to-Generator-to-CUDA-worker plumbing, not normal-model
quality, package Java-runtime relocation, or complete MCP feature parity.

The Windows `-PreferScalaTorch` packaging path then built and ZIP-extracted a
relocatable app image. The 299,537,019-byte package hash is
`a1d0603b2050db5cd144a3a8c25391f667e61dc5f3afc8554772f9b116b81a53`. From the
extracted `SDNUI` directory, MCP stdio generated a 64x64 CUDA image at seed 44
with `SDFX_SCALATORCH_CLASSPATH_FILE` and `SDFX_SCALATORCH_JAVA` unset; logs show
both worker classpath and Java resolved inside the extracted package. The
process exited 0, and a separate `SDNUI.exe` run reached the packaged UI's
`Ready · cuda:0` state. The package includes Python source files but no
`python.exe`; these runs did not start a Python process. The test image is
`.superpowers/native-package-win/extracted/SDNUI/outputs/txt2img/2026-11-03/00000-44.png`.
This validates package relocation and startup on the local GTX 1650 SUPER with
the tiny fixture, not standard-checkpoint image quality or full parity.

## Default PNDM scheduler and refreshed Python-free package (2026-11-03)

The local Diffusers SD1.5 model stores a `PNDMScheduler` config, while the native
worker creates the sampler selected by NUI's request and previously accepted
only Euler source metadata. `EulerConfig` now reads the shared beta settings from
known PNDM configs, validates its PNDM-only boolean metadata, and keeps rejecting
other scheduler classes. The regression test failed on the actual PNDM shape
before the change and `EulerConfigSuite` passes all six tests after it.

The Windows app was repackaged with the updated worker distribution and ZIP
extracted to `.superpowers/native-package-win/extracted/pndm-fix/`; ZIP SHA256 is
`2ea2b814c4c0a812aea0e38604e6e96f96375ad8316cbb522c6d4bb6456a90da`. The
extracted package contains no `python.exe`. With an empty app home and the local
SD1.5 model, packaged MCP stdio returned 26 tools and logged
`engine ready: scala-torch-0.1.0`; its default scheduler path was the model's
`scheduler/scheduler_config.json` (PNDM), and no Python home was created. The
packaged `SDNUI.exe` then reached `Ready · cuda:0` from a second empty home with
the PNDM scheduler environment override unset. Screenshot:
`stablediffusion-nui/.superpowers/native-package-win/extracted/pndm-fix/packaged-pndm-ui.png`.
These prove startup and scheduler-config parsing through the current package;
they do not include a generation request through this exact package or complete
NUI feature parity.

## Full SDXL quality investigation

The official SDXL base checkpoint was tested using Float32 parameters and CUDA
offload on the local 4 GiB GPU. Both with and without the official offset LoRA,
256x256, eight-step images were strongly distorted. The no-LoRA result was then
compared with a separate CPU Diffusers/Transformers execution using the same
weights, prompt, Java-seeded initial noise, guidance, 256x256 size and DPM++ 2M
Karras schedule. The reference image was similarly distorted; RGB pixel error
versus the native PNG averaged 0.38 out of 255 (maximum 49 around sharp edges).
This shows that this particular poor output is reproduced independently; it does
not establish general SDXL image quality. At 512x512 and 16 steps, the same
prompt and seed produced a recognizable red apple on a table in both runs.
Native generation took 580.4 seconds; its RGB pixel error versus the CPU reference
averaged 0.030 out of 255, with a maximum of 4. The native image is
`.superpowers/runs/sdxl-baseline-512x16/native-c9971ec3-d73e-4dc3-ac27-35c14c59cc0b.png`.
This is a direct worker result, not a packaged NUI acceptance run.

The native-only NUI Windows package was then rebuilt and extracted. With Python
environment variables removed and PATH limited to Windows system directories,
its MCP `txt2img` completed the same 512x512, 16-step request on CUDA using the
bundled worker and Java runtime. The saved PNG was pixel-identical to the direct
worker image; the response included seed, model and generation metadata. The
ZIP SHA-256 was `548341182b23a0b93adb68eb20d6ca96f2e3728a0d2e8327115d9b10abbf0e61`;
its 828 entries contained no Python payload. Other NUI workflows remain open.

After this acceptance run, CUDA upload was changed to copy contiguous tensor
storage directly to native staging memory rather than first making a full-size
JVM array. A 16 MiB contiguous offset view used 12,520 bytes of JVM allocations
in the warmed test. A separate grouped, batched convolution test also exposed
and fixed missing geometry metadata in the CUDA workspace budget calculation.
These changes passed the CUDA session/convolution suites and the full SDXL
component references. A later 512x512/16-step packaged MCP run using the same
model, prompt and seed produced the same PNG byte-for-byte. Using the settings
file and output image timestamps as consistent endpoints, one run took 601.4s
before the change and one took 525.9s after it. These single runs include startup
and model loading and are not a controlled performance benchmark.

The native worker now also supports standalone `upscale` without loading a
diffusion checkpoint. NUI `Generator` successfully exercised Nearest and the
official Real-ESRGAN x4+ model on CUDA, including a fivefold request requiring
two model passes. The model-file alias is resolved in NUI's shared catalog. The
native route supports None, Lanczos, Nearest and that Real-ESRGAN model; other
model upscalers remain explicitly unsupported. Optional hardware metadata now
includes the best available CPU/GPU/RAM profile, elapsed time and remark, with
an `unknown` fallback when probing fails. The NUI Generator tests verify this
metadata in PNG output for both Nearest and the official CUDA Real-ESRGAN model.
The same option now works for txt2img and img2img, including sequential batch
items. The native-only Windows ZIP SHA-256
`7883d97397b53fef0875b366213b7f11539d958603d137cbd3c330defc121076` was
extracted and generated a 64x64/one-step image from official SDXL through MCP
with hardware information enabled; the response and saved PNG included the
profile, time and remark. Python-free NUI Generator tests separately cover two
images and img2img on tiny SDXL, and the worker suite covers CPU and CUDA.

Native Euler ancestral (`Euler a`) now supports deterministic request-local
noise, txt2img, img2img, latent Hires and img2img Hires Fix. Independent
Diffusers fixtures cover epsilon and v-prediction steps; tiny SDXL tests cover
CPU/CUDA replay, cancellation and both Hires paths. Uniform is the default;
Karras follows the current NUI Python fallback to checkpoint timestep spacing
and emits a warning. PyTorch and Java RNG streams differ. The native-only ZIP
`18e6b0d38d6f745b56617eb75bc0115cc66c74cdff18da07600f7f6295a89169`
completed an official SDXL 64x64/two-step Euler a MCP smoke after extraction;
this confirms execution, not full-size quality or complete sampler parity.

Additional full-checkpoint fixtures compare FP32 NUI prompt conditioning,
U-Net output and VAE decoding against independent CPU references. The measured
maximum absolute errors were 0.000184 for prompt context, 0.000014 for pooled
text, 0.0000031 for U-Net and 0.000016 for VAE. The optional
`FullSdxlReferenceSuite` requires the checkpoint, fixture directory and CUDA;
`python scripts/reference_full_sdxl.py MODEL_ROOT NUI_ROOT OUTPUT_DIR` generates
the CPU fixtures. Set `scalatorch.full.sdxl.dir` and
`scalatorch.full.sdxl.reference` as test JVM properties and set
`SCALATORCH_CUDA_TESTS=1` to run the suite. Python is used only for the external
comparison and is not part of the native worker.

NUI Euler Uniform now uses log-interpolated training sigmas regardless of the
checkpoint timestep spacing in both image and latent-return worker paths.
Fixtures generated from NUI and Diffusers cover Uniform and Karras schedules
at 1, 3, 4 and 9 steps. The native-only Windows package ran official SDXL at
64x64 and one Euler Uniform step without Python in the runtime environment;
against an independent CPU reference using the same initial noise, the saved
RGB image differed by at most 1/255 and averaged 0.0031/255. This is an
execution and numerical smoke, not a full-size quality assessment.

Euler Exponential is now available for native image generation and both Hires
paths. Its default schedule matches independent NUI/Diffusers reference
fixtures at 1, 3, 4 and 9 steps; a tiny SDXL worker test checks pixel equality
against the standalone pipeline at three steps and executes latent Hires.
Exponential also works for DPM++ 2M in image and latent-return passes. For
Euler a, NUI/Diffusers instead emits a warning and uses checkpoint timestep
spacing; the native worker follows that fallback. DPM++ 2M now retains a
separate FP32-compatible training schedule for exponential interpolation,
without changing the general double-precision schedule. For the tested SD
scaled-linear profile, model timesteps exactly match NUI/Diffusers fixtures at
1, 3, 4, 7 and 9 steps. Fixtures also cover a linear profile and trained betas;
there, model timesteps are within one step, with sigma values within 0.0002.
FP32 square-root and logarithm implementation differences still affect some
integer boundaries, so general exact numerical parity remains open.

The native worker now implements `image_tool` for NUI's `cas_upscale`: bounded
input decoding and EXIF orientation, Lanczos resize, contrast-adaptive
sharpening and atomic output publication. Five independent NUI/Pillow reference
cases, including RGBA input and fractional scale, matched within one RGB level.
The cross-repository NUI EngineClient test launched the worker without Python
and ran CAS before loading a checkpoint. The native worker also implements
Loopback Scaler's `loopback_scaler` finishing step, including sharpness,
brightness, color, contrast, repeated detail/smooth/blur and contour. Its 26
NUI/Pillow fixture cases, including RGBA and 2x2 inputs, agree within 3 RGB
levels; repeated-filter rounding is not pixel-exact. The worker advertises
`loopback_scaler`, allowing NUI's feature-gated plugin to be selected. A NUI
AppServices and JobQueue integration ran two img2img loops, non-identity
finishing and final PNG publication on the tiny SD1 checkpoint, without
Python; the second seed and saved metadata were checked. JavaFX interaction,
full checkpoint quality, nondefault settings and complete NUI feature parity
remain unverified. Other `image_tool` variants remain unsupported.

The native `tokenize` request and `tokenizer` feature now inspect the active
pipeline tokenizer, including both SDXL streams, token IDs/pieces, UTF-8 bytes
and grouped text, and added textual-inversion token IDs. NUI/Transformers
fixtures cover English, Japanese, specials, empty text and explicit IDs for tiny
SD1/SDXL; model load/unload and invalid IDs are checked. An NUI EngineClient
typed request ran against the native worker without Python. Native generation
and inspection now use an explicit Transformers slow-tokenizer fallback
profile for environments without optional ftfy, while the public fast profile
remains unchanged. Independent SD1/SDXL fixtures cover adjacent CJK characters,
control removal and context-sensitive Greek lowercasing; the NUI typed client
also checked Japanese IDs. Optional ftfy normalization, exhaustive Unicode
parity and JavaFX interaction remain open.

The native worker now accepts model-free `preprocess` requests for none,
inpaint, invert, Canny, luminance/intensity recoloring, tile resampling,
tile color correction and Gaussian blur. Standalone Extras requests retain
NUI's resolution rules, warnings, `parameters` PNG text and original `Source
parameters`; NUI Generator exercised preview and Extras without Python.
Independent Pillow/OpenCV fixtures cover 24 combinations. The Gaussian blur
now uses Pillow-compatible extended box passes, and Canny handles image borders.
The previous 17-pixel discrepancy in 64px Canny is now eliminated for the
tested synthetic input: native cubic resize uses replicated boundaries and
preserves intermediate precision, matching the Windows OpenCV 4.11 reference
for 11 tested image sizes and patterns. Normal, standalone and pixel-perfect
Canny fixtures now match exactly; other tested preprocessors match within one
RGB level. NUI Generator also exercised the 64px Canny Extras request. These
are bounded fixtures, not proof of parity for all images or OpenCV builds;
detector-based preprocessors remain open. The reference interpolation behavior
was checked against [OpenCV 4.11 resize.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/resize.cpp).

The native I/O module now writes safetensors with F32/F16/BF16/F64/I64/BOOL
payloads, metadata, scalar/empty/strided tensors, cancellation and staged
publication. The embedding layer writes validated SD1 `clip_l` and paired
SDXL `clip_l`/`clip_g` tensors with optional step metadata. Native round-trip
tests and independent upstream safetensors/PyTorch reads verified the files;
Python was used only as an external comparison tool. Other NUI embedding
editor and merge operations remain required.

For future embedding editing, the native I/O layer can stage a copy of an
existing safetensors file after checking its SHA-256, changing only selected
coordinates while preserving the original header, metadata and other payload
bytes. A fixture produced by NUI's existing editor matches byte for byte. The
model layer maps SD1 and SDXL encoder names to this staging operation. Staging
requires filesystem hard-link support and accepts only dtypes understood by the
native reader, including for untouched tensors.

The native embedding editor now supports opening an active or selected
SD1/SDXL embedding and updating guidance. It compares active model vectors to
the file, fingerprints the source before and after reading, and checks a
draft's model, file identity and vector dimensions when guidance is updated.
Tiny SDXL worker and NUI tiny SD1 child-worker tests exercise these paths.
For supported safetensors, an explicitly confirmed save stages the edit,
revalidates the source, retains a verified backup, and reloads the edited rows
into the model. Tests cover backup bytes, stale drafts and live reload. The same
flow now accepts the native reader's restricted current-format PyTorch `.pt`
and `.bin` embeddings, preserving supported tensor dtypes, shapes and primitive
metadata; unsupported archive values are rejected. Tiny SDXL and NUI tiny SD1
exercise backup and reload, and an independent PyTorch weights-only read checks
an edited Float16 archive. Loaded file metadata is not tracked from the original
activation time. Generation requests can carry an
unsaved editor draft: the worker validates it, uses independent CLIP embedding
rows for the request and restores the prior model in a finally block. Tiny SDXL
and NUI tiny SD1 tests verify changed output and normal output recovery; the
SDXL test also covers a generation error. Embeddings first activated during a
preview remain available afterward, while the edited rows are restored.

The native embedding inspector now implements its `tokenize` operation for
both CLIP streams, including the 128-token display limit and warnings. A NUI
EngineClient test starts the distributed child JVM and checks the response on
tiny SD1 without Python. The inspector also reads model token vectors and
compatible SD1/SDXL embedding files and returns cosine-ranked vocabulary rows;
the SDXL file path has unit coverage and the model-token path is exercised
through NUI. Loaded embedding file metadata remains unverified; PyTorch file
metadata is not yet surfaced. The inspector list reports successfully loaded
embeddings and their encoder shapes separately from catalog names, and clears
loaded entries on model reload or unload. Loaded file metadata is marked
unverified. Inspector vector export writes UTF-8 JSON for model tokens and
supported embedding files, with SDXL paired streams, cancellation before
publication and explicit overwrite handling. No-overwrite export requires
filesystem hard-link support. The inspector supports weighted SD1/SDXL mixing,
concatenation and combination, saves new `.safetensors` embeddings, and
activates them in both live encoders. Tiny SDXL and NUI tiny SD1 tests exercise
this route. Existing active embeddings can be replaced, including a changed
vector count; tokenizer IDs and other embeddings are remapped together before
publication. Tiny SDXL coverage checks preservation and cancellation, and NUI
tiny SD1 coverage checks replacement through the child worker. The restricted
native PyTorch ZIP writer enables new SD1/SDXL `.pt` mixer outputs; independent
PyTorch `weights_only` loading and native round trips verified its output.
The eight fixed mixer presets run as native operations, including batch output
and live activation; arbitrary Python eval expressions remain unsupported.
Batch inputs are resolved once, output collisions and token names are checked
before publication, and a later publication failure reports already saved paths.
Tiny SDXL and NUI tiny SD1 cover batch output and activation. No-overwrite
publication requires hard links.
Editor staging accepts Double coordinates
to avoid narrowing user edits before encoding the source dtype.

The standard Hires setting is now ignored on img2img requests, matching NUI's
`mode == "txt2img"` condition. The separate img2img Hires Fix hook remains
available. Masked img2img with the ordinary Hires flag produces the same pixels
as without it in synthetic nine-channel coverage and an NUI client test using
the four-channel tiny SD1 model. This is not evidence of trained inpaint-model
quality or parity for all image-processing options.

For txt2img latent Hires, the second pass accepts independent prompt, negative
prompt, CFG, sampler and scheduler settings, falling back to base settings when
omitted. Embeddings referenced only by the Hires prompt are loaded as well.
Parser coverage, a tiny SDXL worker test and a tiny SD1 NUI child-worker test
exercise these overrides. Hires can inherit base LoRAs, replace them with an
explicit set, or disable them with an explicit empty set. Tiny SD1 and SDXL
worker tests and a tiny SD1 NUI child-worker test cover this selection; the
SD1 worker test also checks later requests do not retain the selection. Base
prompt LoRA and LyCORIS tags now resolve through the supplied LoRA catalog;
model-keyword batches use the first prompt's tags together with manual LoRAs.
Tiny SD1 generation checks tag output against an explicitly supplied adapter and
checks that pre-resolved keyword LoRAs are not applied twice. Other extension
types and scheduled/block-weight LoRAs remain unsupported.

Ordinary linear and convolution LoRA factors now accept the `dyn` dynamic-rank
field, including Python-style negative slicing and zero rank; the adapter cache
distinguishes ranks. Loader tests pass, including CUDA convolution regression.
Tiny SD1 worker image generation exercises ranks 0, 1, 2 and -1 on CUDA, and NUI
Generator passes rank 0/1 to the native child worker without Python. The broader
CPU worker suite also passed (six tests, three SDXL tests skipped). DoRA and
block weights remain unsupported; full trained-model dynamic-rank parity remains
unverified.
Named `dyn=` and fourth positional rank are also accepted in native prompt
LoRA/LyCORIS tags, including Hires and keyword tag parsing. Tiny CUDA worker
and NUI Generator tests confirm tagged and explicitly supplied ranks agree.
The `LoraStepWindow` policy matches 600 NUI Python reference cases, including
ties-to-even fractional boundaries and the exclusive stop. Standard `start` and
`stop` fields and prompt tags now filter only U-Net LoRA evaluation per active
step. Tiny SD1 CUDA generation checks inactive, full and partial windows and
equivalent structured/tagged input; scope restoration is tested. Tiny SD1 CUDA
img2img checks window boundaries against its remaining steps, and two uses of
one LoRA file with different windows/ranks remain independent. NUI Generator
also passes structured and tagged windows to the native CUDA worker. A tiny SD1
CUDA Hires test exercises an independent second-pass LoRA selection, including
inactive, full and partial windows; fractional 0.5 and step 2 agree for four
active steps. Tiny SD1 Euler, Euler ancestral and DPM++ 2M txt2img/img2img
trajectories match an independent Python Diffusers reference at every step, for
five windows on CPU and CUDA (absolute tolerance 0.002). Hires trajectory and full-model image parity remain
unverified. `step=START-STOP` prompt tags now match the NUI start/stop parsing
and pass tiny CUDA worker and NUI Generator tests. Block-weight preset/layout
expansion and the SD1/SDXL Diffusers scale plans now match 28 Python reference
cases, including user presets and short/long numeric lists. Worker generation now
applies resolved block weights, including structured input, prompt tags and user
presets; tiny SD1 CUDA worker and NUI Generator tests cover ALL/NONE and selective
presets. A tiny SDXL CUDA worker also verifies a mid-attention LoRA with ALL,
MIDD, NONE and INS against its unweighted and unadapted outputs. NUI Generator
also exercised the tiny SDXL CUDA worker with structured MIDD and tagged NONE
without Python. Trained-model image parity and full SDXL block-weight coverage
remain unverified.
For tiny SD1, five combinations of block weights, dynamic rank and step windows
now match independent Python/PyTorch latent trajectories at each Euler step on
CPU and CUDA (absolute tolerance 0.002). A CUDA worker test also checks combined
structured settings against equivalent prompt tags, and NUI Generator repeats
that combined-settings comparison through its native CUDA child worker. This is
not full-model parity.
The real SD1.5 single-file checkpoint and trained LCM SD1.5 LoRA (278 factor
pairs, including 86 non-attention modules) also completed NUI Generator CUDA
smokes at 64x64 without Python. The `ALL` and `NONE` block settings produced
different images; the run did not compare those images to Python. Diffusers'
fallback can leave some non-attention layers at its default scale, so `NONE`
should not be interpreted as a guarantee that every module is disabled.
Per-target scale lookup now reproduces the installed Diffusers attention-key
expansion and fallback for 12 representative module paths across those cases.
The Python fallback cannot index certain short module names such as `conv_in`;
the native lookup represents those as unresolved instead of silently guessing.

A trained LCM SD1.5 LoRA's `proj_in` convolution layer was independently
compared against Python NUI `truncate_state_dict` and PyTorch for dynamic ranks
0, 1, 8, 64 and -1. CPU/CUDA maximum errors were below 1.6e-9 for this layer.
A tiny CUDA image-generation test also confirms that two references to the
same LoRA file with different ranks retain independent cached factors. This
does not establish full-checkpoint or complete-image parity.

Selected model-keyword embeddings and LoRAs now participate in native generation, including LoRA
keyword lookup from the local catalog and user mappings. Tiny SDXL generation
and an SD1 NUI child-worker test compare these selections with explicit inputs;
the SDXL test also supplies more than 32 catalog aliases. Raw LoRA and LyCORIS tags in standard Hires and img2img Hires Fix
override prompts are resolved for that pass when no explicit pass-specific
adapter list is supplied; an explicit list, including an empty one, takes
precedence. Tiny SD1 generation compares both forms and checks the empty case.
Keyword-generated prompts are included when discovering textual inversions,
as are img2img Hires Fix overrides. Replay metadata preserves the batch's active
LoRA tags and each image's original keyword prompt. When Hires Fix has no
prompt override, it uses the keyword preview and its LoRAs; tiny SD1 tests
compare the result to an explicitly configured second pass.
The NUI catalog-download action now works in the native worker after the UI's
confirmation. It checks the pinned upstream size and SHA-256 before publishing,
does not overwrite an existing catalog, and removes incomplete downloads.
Tests cover failure and publication races, a live download and lookup in a
temporary directory, and a live download through the NUI child-worker protocol.
Native txt2img now accepts a variation seed and strength, interpolating the two
initial noise tensors over their channel dimension as in NUI. Random variation
seeds are returned in image metadata; img2img ignores variation with a warning.
The interpolation has PyTorch reference values and tiny SD1/SDXL replay tests,
including an SD1 request through the NUI client. The default native Gaussian
generator still uses Java's random stream. An explicit `extra.rng=cpu` selects
a PyTorch-style MT19937/Box–Muller stream; independent PyTorch CPU reference
fixtures cover multiple seeds, sizes and consecutive draws with a 2e-6 tolerance.
Floating-point implementation and the rest of the pipeline can still differ,
so matching seeds do not establish image parity with Python. Explicit GPU/NV
random streams remain unsupported on CUDA; on a CPU engine, `rng=gpu` falls
back to the CPU stream as in the Python engine.
The CPU stream was also compared end to end with an independent PyTorch 2.7.1
and Diffusers 0.33.1 run of the pinned tiny SD1 model at 16x16 and three steps.
Maximum final-latent errors were 1.91e-5 for Euler, 2.29e-5 for Euler ancestral,
2.68e-5 for DPM++ 2M, and 1.39e-4 for Euler with CFG 7.5 and variation strength
0.45. The decoded Euler image differed by at most 4.48e-7 in [0,1] float RGB.
The NUI typed client launched the native child worker and produced a PNG within
one 8-bit channel value of that reference. Fixtures are generated only for tests
by `scripts/reference_tiny_torch_cpu.py`; Python is not needed at runtime.
An independent 64x64 Euler reference also matched within one 8-bit channel value
when generated through NUI `AppServices` and `Generator` with a native child
worker and a fresh app home without a Python environment. The same integration
checks img2img and CPU fallback for `rng=gpu`. Tiny SD1 img2img latent maximum
errors against independent PyTorch/Diffusers were 4.30e-6 for Euler, 5.25e-6
for Euler ancestral and 6.68e-6 for DPM++ 2M; decoded Euler RGB differed by at
most 3.28e-7. Run the opt-in NUI
`NativeCpuGeneratorReferenceSuite` with `SCALATORCH_CPU_REFERENCE_64` pointing
to a 64px fixture generated by the same script.
These bounded tiny-model cases do not prove parity for other models, schedulers,
inpainting or CUDA RNG.

Independent PyTorch/Diffusers inpaint references revealed and fixed three native
pipeline errors: four-channel blending selected the wrong mask pixel, full
inpaint strength did not start from pure noise, and the four-channel path did
not consume the unused masked-image posterior sample before stochastic steps.
For tiny SD1, four- and synthetic nine-channel U-Nets, strengths 0.75 and 1.0,
and Euler, Euler ancestral and DPM++ 2M, all 12 CPU reference cases now pass with
maximum latent error 3.06e-5. NUI AppServices/Generator also produced a 64x64
four-channel inpaint PNG within one 8-bit channel value of the independent
reference. The nine-channel model has synthetic widened input weights; trained
inpaint checkpoint quality, other masks/preprocessing and CUDA RNG remain open.

Partly transparent RGBA/LA input masks now use their alpha channel as in NUI's
Python loader; fully opaque masks use luminance. Palette and grayscale input
fixtures match the Pillow loader. On the local GTX 1650 SUPER, the NUI
AppServices/Generator integration generated two 64x64 tiny-SD1 inpaint images
on `cuda:0` with identical pixels for equivalent transparent and grayscale
masks, preserved the unmasked source pixels and created no Python environment.
The same transparent-mask comparison also passed on CPU against the independent
PyTorch image fixture. This does not establish real-checkpoint inpaint quality.

Full-resolution inpaint preprocessing and paste-back were compared byte for
byte against NUI/Pillow for 16 combinations of central, edge, narrow and empty
masks, blur radii 0/2 and padding 0/3. The nonempty cases cover crop coordinates,
resized model inputs and restored output; the empty-mask cases cover fallback
inputs. Separate fixtures cover six mixed resize geometries in each of the three
NUI resize modes. These checks do not exercise a model or full-resolution UI flow.

The NUI AppServices/Generator CUDA integration additionally ran full-resolution
inpainting on an 80x64 canvas with a 64x64 tiny-SD1 model input, blurred mask
and padding. The entire restored output was within two 8-bit channel values of
an independent Diffusers/Pillow reference; no Python environment was created by
NUI. This exercises the Generator and native worker, not the JavaFX UI or a
trained inpainting checkpoint.

The `image_tool` worker now implements the `color_correction` AdaIN and Wavelet
methods, including bicubic reference resizing and strength. Six independent NUI
reference cases match within one 8-bit channel value, and NUI EngineClient ran
both methods through the native child before loading a model, with `python:""`.
The default A1111 method and `all_methods` comparison outputs are also implemented.
The JavaFX Extras flow has been exercised with the native worker as described below.

For the A1111 method, native per-channel histogram quantile matching now agrees
byte-for-byte with four independent scikit-image LAB-stage fixtures, including
single pixels and unequal source/reference dimensions. The independent fixture
also records OpenCV LAB and the final blended output.

An experimental fixed-point RGB-to-LAB implementation follows OpenCV's lookup
table formulation. Against OpenCV 4.11, both RGB-to-LAB and LAB-to-RGB now
match byte-for-byte over all 16,777,216 possible 24-bit inputs in each direction.
The exhaustive fixtures are generated locally and not shipped in the runtime.
The OpenCV license is included alongside the implementation.

The native opaque luminosity blend now matches 1,298 independent blendmodes
reference pixels exactly, including black, white, primaries and random colors;
the upstream MIT notice is included. A numerical experiment on 65,536 colors
showed that a direct floating-point sRGB-to-LAB implementation differs from
OpenCV's byte LAB conversion by up to two levels, so it has not been used. The
fixed-point conversions, histogram and blend now implement A1111: three reference
strengths match the Python image within one 8-bit level. NUI EngineClient ran all
three methods through the native worker before loading a model, with `python:""`.
All-method requests return the selected method plus two separately computed
comparison images, matching standalone native results, and reject publication
without leaving artifacts when cancellation wins. EngineClient exercised the
comparison outputs through the native child. `RealColorCorrectionCheck` now
also ran the actual JavaFX Extras button against the native CPU worker with an
explicit reference and with the original-image fallback. Both runs verified the
three selectable saved outputs, metadata, input reset, invalid-reference error,
retry, stale-result suppression and absence of a Python environment. These used
small fixture images. A further run used a real 512x512 native-generated image
and a 256x256 generated reference: all three standalone outputs at strength 0.4
matched the Python reference byte-for-byte, with individual worker times of
approximately 0.50, 0.18 and 0.27 seconds on this host for A1111, AdaIN and
Wavelet respectively. The JavaFX button and three comparison outputs also
passed using that image pair, without creating Python. Broader image coverage and
package acceptance remain open.

The `color_enhance` image tool now runs in the native worker. Its OpenCV float
RGB-to-LAB grid is packaged as a lookup table; output conversion uses the
standard CIELAB transform. Twelve color/grayscale fixtures across strengths and
percentiles match NUI within two 8-bit channel levels with matching reported
gain. A separate 512x512 generated-image fixture at the default settings had
maximum error 1 and mean absolute channel error 0.000092 against NUI. NUI's
EngineClient and `ColorEnhancePlugin.transform` exercised the tool through a
Python-free worker; the plugin integration compared the 512px result against the
reference and checked its input temporary was removed. A denied publication
also left no output files. The rebuilt native-only ZIP SHA-256
`1dfd0ded64262cd94481feaa89c985e812031a10338e8608e2d153e2681d501b`
contains the 215,622-byte LAB grid and no Python directory. From an extracted
copy, the bundled worker and bundled Java passed the NUI plugin's 512px
reference comparison; the test harness itself ran from the development checkout.
The packaged EXE separately passed MCP initialize/tool listing on a system-only
PATH. Packaged EXE JavaFX interaction remains unverified for this tool.

The native-only package rebuilt after color-correction support has ZIP SHA-256
`1350a1922609f2c476bb25b737578fd79435bc199eb5ead3f824bb055061a3ac`.
From its extracted copy, the actual `SDNUI.exe --mcp` answered initialization
and tool listing with a Windows-only PATH and no Python directory. Separately,
the JavaFX color-correction integration used the extracted package's Java 25
and worker manifest for the 512px image pair and passed all three comparisons,
failure recovery and no-Python assertions. The JavaFX test harness itself ran
from the development checkout, so it is not an end-to-end packaged EXE UI test.
The extracted EXE subsequently opened its own `Stable Diffusion NUI` window,
but Windows Computer Use refused state capture with `Computer Use requires app
approval but elicitations are unavailable`; no packaged EXE UI interaction was
performed. The package script's default-JAR fix is now covered by
`scripts/dev/check_package_default_jar.ps1` under Windows PowerShell.

Native generation now accepts the NUI `Normal`, `Simple` and `SGM Uniform`
schedules with Euler, Euler a and DPM++ 2M. Euler uses NUI's custom sigmas;
Euler a and DPM++ use the same linspace/trailing fallbacks as NUI's Diffusers
adapter. Independent fixtures cover 1, 3, 4 and 9 steps, and nine tiny-SDXL
worker generation combinations matched the corresponding native pipeline.
EngineClient confirmed the three names appear in the native scheduler list.
On GTX 1650 SUPER, NUI Generator also ran all nine sampler/schedule pairs with
the tiny SD1 model, both at 64x64 and with a 128x128 Hires second pass. Changing
the second-pass schedule produced distinct outputs where the schedules differ;
the test asserted `python:""` and no Python environment. The official SD1.5
checkpoint then generated three 512x512 four-step Euler images through NUI
Generator on the same GPU, with CPU noise and weight offload. All three
schedules completed in 303.6 seconds total, at least two outputs differed, and
the three fox images were visually inspected. This is a generation smoke, not
an independent PyTorch image-parity test; JavaFX selection remains open.
An independent tiny SD1 CPU latent comparison now covers all nine combinations
for txt2img and img2img at CFG 1, and again at CFG 7.5 with a negative prompt.
All 36 cases passed with maximum absolute latent error 0.000145 against NUI's
Diffusers scheduler and noise-seeding behavior. This does not establish image
parity for the full checkpoint or CUDA execution.
Polyexponential is available through the native worker: Euler uses NUI's
default exponential-equivalent sigma curve, while Euler a and DPM++ 2M follow
NUI's respective fallback behavior and emit its warnings. A PyTorch CPU tiny
SD1 latent comparison covers all three samplers, txt2img/img2img and CFG 1/7.5;
its maximum absolute error across the expanded 48 cases is 0.000145. The
worker distribution was rebuilt and NUI AppServices/Generator exercised all
three samplers with Polyexponential on CUDA, for 64px base generation and
128px Hires, with a no-Python handshake. This is a tiny-model integration
test; full-checkpoint image parity and packaged EXE interaction remain open.
Work has begun on the separate img2img Latent Upscale script. Its full-step
schedule arithmetic and 10,000-step limit now have boundary tests and are
connected to worker requests. Euler, Euler a and DPM++ 2M support up to 10,000
schedule entries. The pipeline now supports a source-sized
VAE posterior followed by a latent transformation and continues the same noise
stream after the source-sized posterior draw. Tiny SD1 CPU comparisons against
PyTorch cover a 16-to-24-pixel nearest-exact transform for Euler, Euler a and
DPM++ 2M; the existing four/nine-channel inpaint reference tests also pass.
The area interpolation mode matches a PyTorch reference fixture. A tiny-SD1
worker request verified output size and full denoising steps, alongside existing
img2img and Hires regression tests. NUI AppServices/Generator subsequently ran
an unmasked 64-to-128-pixel nearest-exact request on CUDA using the rebuilt
worker and asserted no Python environment. A non-divisible source regression
also verifies that the ordinary pixel resize mode is ignored. The worker now
advertises the `latent_upscale` feature. NUI's actual script/queue path ran all
seven interpolation choices, all 18 combinations of three samplers and six
schedule choices, and masked full-resolution generation on tiny SD1 CUDA, with
no Python. Masked worker
requests now encode source and masked source before drawing diffusion noise;
the four-channel tiny SD1 latent outputs match PyTorch for Euler, Euler a and
DPM++ 2M at 0.75 and 1.0 strength, and a worker smoke checks dimensions and step count.
Source images not divisible by the VAE scale are now resized before encoding,
with nearest-neighbor mask resizing, and an odd-sized worker smoke passes. A
synthetic nine-channel tiny SD1 model also matches the PyTorch latent reference
for the same three samplers and strengths. The rebuilt worker also completed a
masked 64-to-128-pixel tiny SD1 request through NUI Generator on CUDA without
Python. Full-resolution masked Latent Upscale now has a tiny SD1 CPU final-image
comparison against NUI/PyTorch within two byte levels per channel. NUI Generator
also ran the full-resolution route on CUDA and checked original output size and
an unchanged pixel outside the mask. Masked SDXL, trained nine-channel checkpoints,
and independent CUDA numerical parity are not verified. Scheduler counts beyond
the training-table length are now supported. Seven Euler schedules at
1001, 2000 and 10000 steps match selected NUI reference timesteps and sigmas.
A 0.0005-strength, one-step request (2000 internal schedule entries) completed
through the native worker and NUI Generator on CUDA without Python.
Euler a now advances its random stream on repeated timesteps, as Diffusers does,
and accepts the trailing schedule's negative endpoint. Tiny-SD1 CPU img2img
latents match PyTorch for 2000- and 10000-step schedules with repeated timesteps;
a 2000-entry Euler a Latent Upscale request also ran through NUI on CUDA.
DPM++ 2M Karras, Exponential and Polyexponential at 2000/10000 entries match
tiny SD1 PyTorch CPU latent fixtures. Discrete Normal/Simple schedules can repeat
noise levels: Diffusers produced nonfinite results for some such cases, whereas
the native solver keeps the sample unchanged for zero-width intervals, uses a
first-order update after them, and emits a warning. Finite reference cases match
within the test tolerance; DPM++ Simple and Karras with 2000 entries also ran
through NUI Generator on CUDA without Python.
The real SD1.5 checkpoint also ran the Latent Upscale plugin through NUI's job
queue with a previously generated 512px fox image, three denoising steps and
CPU RNG, on the local 4 GiB GTX 1650 SUPER. It completed in 93.7s without a
Python environment and the 512px result was visually inspected at
`stablediffusion-nui/.superpowers/real-latent-upscale/sd15-latent-upscale-16987177509458500779/fox.png`.
This is a real-model smoke test, not independent image parity or an EXE UI test.
For an unmasked request on a nine-channel model, the pipeline now encodes the
all-zero masked source before drawing diffusion noise, matching NUI's implicit
all-white mask. Synthetic nine-channel CPU fixtures match PyTorch for three
samplers at strengths 0.75 and 1.0. The native worker also generated all six
combinations with a synthetic nine-channel model; its 0.75-strength images for
all three samplers match the decoded PyTorch latent reference within one byte
level per channel for strengths 0.75 and 1.0. NUI AppServices/Generator also
ran a 64-to-128-pixel unmasked request on CUDA using a synthetic nine-channel
tiny model and asserted a no-Python handshake. The synthetic model was prepared
with a separate Python fixture script; runtime generation did not use Python.
Trained-model and independent CUDA numerical parity remain open.
The native worker also resolves NUI's aliases for its three supported samplers,
case-insensitive canonical names, and case-insensitive supported schedules
(including `sgm_uniform`). Tiny SDXL worker tests compare alias-generated images
against canonical requests, including automatic Karras selection. This does not
add support for the remaining sampler or schedule families.

Inpaint Difference is partially ported: the native worker computes thresholded
RGB differences, contour boundaries and 3x3 dilation/erosion, then uses the generated
mask directly for img2img without temporary image files. Fifty-five same-size and four
resized mask fixtures match NUI's Python implementation. The contour boundary
rule was also checked against OpenCV for all 65,536 binary 4x4 images. Tiny SD1
worker generation matches explicitly supplied masks; NUI Generator also verified
the non-contour comparison on CUDA without Python. A model-free preview request
now creates owned mask and overlay PNGs, with effective-mode Pillow blur and
source-mode Gaussian blur. Forty-eight Python-reference previews agree within two
levels per RGB channel. NUI's preview service has exercised contour preview and
artifact cleanup through the native worker without Python. The worker advertises
`inpaint_difference`: NUI's plugin queue has generated with and without contours
on CUDA, matching corresponding explicitly supplied masks without Python.
An SD1.5 512x512 CUDA plugin-queue run with an effective preview and six steps
also produced a visibly repaired mask region without Python (149.5 s); a three-step
run at strength 0.75 retained nearly all of the artificial magenta source, so it
is not evidence of successful visual repair. Source-mode OpenCV byte-exact parity,
real-model desktop UI verification, edge blending quality and broader image
coverage remain open.

The magenta boundary in that sample is consistent with NUI's blurred-mask
composite: at (184,150), Pillow's radius-4 mask is 139/255 and the original
magenta contributes at least 116 to both red and blue. Repeated binary dilation
and erosion now use separable sliding windows, with cost linear in image pixels
instead of proportional to the requested iteration count; maximum radius and
thin-image border handling are covered by reference tests.

GPU temperature guard now runs as a native worker sampling callback and is
advertised to NUI. Its hysteresis, maximum wait, polling interval, cancellation
and invalid settings have model-free tests. NUI's always-on plugin has run with
an actual NVIDIA sensor and a tiny CUDA generation without Python. ROCm and
OpenHardwareMonitor provider parsing is checked against selected-device,
missing-value and invalid-reading fixtures but remains unverified on their
hardware. The NUI queue was also tested with an artificially low threshold:
the actual NVIDIA reading triggered a pause and a one-second timeout, after
which generation completed. This does not test a physical over-temperature or
cooling cycle. Sensor subprocess output is drained concurrently and bounded so
a large monitor response cannot deadlock while waiting for process exit.

RescaleCFG now applies Diffusers' per-sample standard-deviation rescaling after
classifier-free guidance, on both CPU and CUDA. SD1 img2img/inpaint and image
refinement passes deliberately ignore it, as the installed Diffusers SD1 image
pipelines accept but do not use the keyword; SDXL image pipelines apply it.
Independent tiny-SD Diffusers final-latent references cover txt2img and img2img
with Euler, Euler ancestral and DPM++ 2M (maximum error 0.000119). A two-sample PyTorch reference
and a tiny SDXL CPU/CUDA predictor comparison passed; the largest absolute CPU
versus CUDA difference in the rescaled output was 1.24e-5. NUI's always-on
RescaleCFG plugin also changed a tiny SD1 CUDA output through the real job queue
without Python. SDXL text and image input were compared against independent
Diffusers U-Net/scheduler final latents for Euler and DPM++ 2M, with maximum
error 6.2e-6. NUI Generator on CUDA also confirmed that rescale changes tiny
SDXL outputs for both masked and unmasked image input, without Python.
Full-model image quality, broader sampler/hook combinations and desktop UI
acceptance remain unverified.

CFG Schedule is now advertised and applied using the full scheduler timetable,
including the correct offset when img2img starts partway through a schedule.
Four curve types, three samplers and text/image inputs match 24 independent NUI
Python/Diffusers latent trajectories (maximum absolute difference 0.000145),
including the supported RescaleCFG interaction. Tiny SDXL CPU and CUDA predictor
results with both hooks differ by at most 6.6e-6. NUI's actual plugin queue on
CUDA confirms scheduled output, RescaleCFG interaction, and the ignored-with-
warning behavior at CFG <= 1 without Python. Normal-model visual acceptance,
other hooks and desktop UI remain unverified.

ReSharpen is now a request-local latent callback, applied after inpaint latent
blending and before step progress, and advertised to NUI. Thirty independent
Python/Diffusers reference trajectories cover all five scaling curves, three
samplers, and text/image input (maximum latent error 0.000123). A tiny SDXL
three-step CPU/CUDA comparison and the NUI plugin queue with CUDA both passed;
the queue verified a changed image and no Python environment. Masked inpainting,
including both four- and nine-channel tiny models, now matches twelve further
Python/Diffusers references across three samplers and strengths 0.75/1.0 (maximum
latent error 0.000159). NUI's CUDA plugin queue also ran two-pass Hires, verified
the effect and reproduced identical pixels on a repeated job. This checks
determinism, not Hires pixel parity with Python. Normal-model visual effects,
other latent-edit combinations and desktop UI remain unverified.

Latent Mirroring is now exposed to NUI and runs before ReSharpen, matching the
Python callback order. Forty tiny-SD Python/Diffusers reference trajectories
cover all five UI styles, both modes, text/image input, Euler and DPM++ 2M with
ReSharpen. Non-square 90-degree rotation follows NUI's 180-degree fallback.
NUI's CUDA plugin queue generated a changed image without Python. The current
Python implementation treats "Horizontal+Vertical Mirroring" as horizontal
because its horizontal prefix check wins; native matches that behavior. Masked
inpaint, SDXL and normal-model visual effects still need separate verification.

Diffusion Color Grading now implements SD1/SDXL latent color recentering and
final-step normalization after mirroring and ReSharpen. SD1 and SDXL intermediate
and final latent vectors agree with NUI's Python callback; four tiny-SD Euler/DPM
text/image trajectories also match with both preceding latent hooks enabled.
Tiny SDXL CPU/CUDA comparison and the NUI CUDA plugin queue passed without
Python. Larger latent sizes, masked inputs, broader option combinations and
normal-model visual effects remain unverified.

Anti-burn now keeps request-local step history, averages the requested recent
latents and stops sampling at the same callback boundary as NUI. Twenty-four
tiny-SD Python/Diffusers comparisons cover three samplers, text/image input,
zero and nonzero early-stop settings, and composition after mirroring,
ReSharpen and latent color grading. NUI's CUDA plugin queue verified that a
four-step job executes two steps with skip=2 and returns an image without
Python. SDXL, masked input, Hires, cancellation during sampling and real-model
visual quality remain unverified.

Vectorscope CC now runs as the first latent callback, including Straight,
Cross, N.Random and U.Random methods and all five scaling curves. Independent
NUI vectors cover SD1 and SDXL, repeated steps, batch-of-two random streams and
the Hires-specific hook seed. Twelve tiny-SD trajectories match the Python
callbacks together with mirroring, ReSharpen, color grading and Anti-burn. NUI's
CUDA plugin queue verifies a changed image and deterministic replay without
Python. Full Hires/batched end-to-end parity, masked input and real-model image
quality remain unverified. Additional independent tiny-SD references now cover
two-image batches for both random methods and all three samplers, plus an Euler
latent-Hires second pass with two images and the NUI second-pass seed; final
latents agree within 0.002. NUI's CUDA plugin queue also ran and replayed a
two-image Hires batch deterministically without Python. These checks do not
establish broader Hires/seed-plan or normal-model coverage.

Detail Daemon now applies its U-Net-only noise/timestep adjustment separately
to conditional and unconditional branches, leaving ControlNet inputs unchanged.
The envelope and sigma conversions match Python reference fixtures; 24 tiny-SD
Euler/DPM text/image trajectories cover all three modes with CFG on/off. DPM's
integer timestep rounding is reproduced. A tiny SDXL CPU/CUDA predictor comparison
with CFG Schedule and RescaleCFG passed, and NUI's CUDA plugin queue produced a
changed image without Python. A nonzero tiny ControlNet and Detail Daemon were
also compared against Diffusers at the predictor level on CPU and CUDA, covering
all three modes (maximum absolute error 0.000035); the test checks that ControlNet
actually changes the prediction. A separate factor-8 tiny VAE fixture now also
exercises native worker txt2img with ControlNet and Detail Daemon on CPU and
CUDA: both features change the resulting images. NUI's catalog and generation
queue also exercised this tiny ControlNet together with the Detail Daemon plugin
on CUDA without a Python runtime, and each affected the generated image. These
image tests do not establish pixel parity with Diffusers. Trained-model output,
other schedules, Hires and interactive UI acceptance remain unverified.

ControlNet step gating now follows Diffusers' full step interval: a step is
active only when both its start and end fall inside the requested range. Soft
control mode uses the source policy's fixed exponent `12 - index`, including
the 10-residual SDXL case. Boundary and weight checks plus the tiny CPU/CUDA
prediction and image-generation suites pass; full SDXL ControlNet model
generation remains unverified.

Tiny worker image tests now verify guidance windows in txt2img and truncated
img2img on CPU and CUDA. Source-compatible Canny also enables soft ControlNet
injection on the Hires pass, matching NUI's Python policy; request preparation,
resize propagation and the soft scale vector are tested, while Hires image
parity and trained ControlNet models remain unverified.

On the GTX 1650 SUPER (4096 MiB), the opt-in `NativeSd15VramSuite` ran a real
SD1.5 Diffusers model at 512x512, Euler, one step, CPU-offloaded U-Net and VAE
weights, fp32, on CUDA. Loading took 6.55 s and generation 40.70 s. Sampling
`nvidia-smi` approximately every 0.5 s observed a whole-device peak of 1930 MiB,
including other applications; brief peaks between samples can be missed. This
is a memory/performance smoke, not a meaningful image-quality or multi-step
benchmark. The test requires `-Dscalatorch.sd15.dir=<model>` and
`SCALATORCH_CUDA_TESTS=1`.
With the same configuration and two steps, loading took 5.92 s and generation
59.15 s, with a sampled whole-device peak of 2054 MiB. Progress events arrived
26.42 s and 45.66 s after generation began; the remaining 13.49 s includes VAE
decoding and output handling. These are wall-clock intervals, not kernel-level
profiles, and the first interval also includes prompt conditioning and setup.
An opt-in synchronous CUDA wall-clock profile (`SCALATORCH_CUDA_PROFILE=1`)
on the same real 512px one-step run observed U-Net attention scores/softmax/
output at 4.08/1.82/3.02 s and uploads including allocation at 3.65 s. The
separate VAE session spent 5.13 s in group normalization. Total generation was
39.87 s with a sampled whole-device peak of 2054 MiB. Measurements include
launch and synchronization overhead and instrumentation, and identify
optimization candidates rather than demonstrating a speedup.
GroupNorm now keeps its original serial Float64 reduction order and applies
the resulting statistics in parallel for large groups, falling back to the
original kernel if the small scratch allocation does not fit. Large-group CUDA
outputs match CPU Float32 bit-for-bit in the affine fixture, and the complete
GroupNorm suite passed. On the same real 512px one-step profile, whole generation
fell from 39.87 s to 36.35 s and sampled whole-device peak was 2004 MiB. VAE
statistics alone took 2.11 s; its separate parallel-apply time was below the
top-12 reporting cutoff, so this is not a precise comparison of combined kernel
times. These single-run timings are indicative, not a controlled benchmark.
Diffusion CUDA attention now permits up to 64 MiB rather than 16 MiB of score
scratch, still bounded by the session's remaining allocation budget. On the
same real-model one-step profile the U-Net's score/output/prepare launches fell
from 734 each to 214 each, and generation measured 35.44 s with a 1940 MiB
sampled whole-device peak. These are individual runs and do not establish a
reliable speedup distribution or a strict VRAM peak. Six sliced-attention CUDA
tests, including low-budget fallback and bitwise comparisons, and the independent
ControlNet/Detail Daemon CPU/CUDA prediction comparison passed after the change.
The GEGLU CUDA series now stops once the decreasing positive terms no longer
change its Float64 sum. A 20,001-value erf reference comparison, the tiny U-Net
Transformer and ControlNet/Detail Daemon tests pass. In a single comparable
real-model run GEGLU dropped from 1.18 s to 0.84 s, full generation from 35.44 s
to 35.13 s, with a sampled 1877 MiB whole-device peak; this is not a statistical
benchmark or an end-to-end image parity test.
After rebuilding both FFM and model JARs, NUI's child JVM passed its CUDA
ControlNet/Detail Daemon queue test and the real SD1.5 Latent Upscale queue test
without a Python environment. The latter completed in 83.691 s using the
previously generated fox as input; its 512x512 output is preserved at
`stablediffusion-nui/.superpowers/real-latent-upscale/sd15-latent-upscale-17635554731149707099/fox.png`.
The output was visually inspected, but this does not establish numerical parity
or interactive JavaFX acceptance.
The latest native-only Windows ZIP was rebuilt with the CUDA optimizations and
extracted independently (SHA-256
`a7e222d8754c201f0d1736f9447388f1401807ebb006bfb5c8a354fc3c1afa66`). Its
`SDNUI.exe --mcp` used the extracted worker JRE and JARs, loaded the local SD1.5
model on CUDA and completed 512x512 one-step txt2img. The extracted package has
no `python.exe`. The output is a one-step smoke image, not a quality assessment;
this exercise does not verify interactive JavaFX workflows.
The extracted FFM JAR's SHA-256 matches the rebuilt worker distribution and the
package contains no `.py`, `.pyc`, or Python executable files. The large-group
CUDA GroupNorm suite also passes when the session budget fits only input and
output, verifying fallback without scratch or leaked allocations. An attempted
packaged EXE UI launch through Computer Use was rejected by its automatic app
approval because elicitations were unavailable; no interactive UI result was
obtained.

Work toward the actual application in this order:

1. Complete the reusable JVM image-generation path using CLIP/U-Net/VAE, scheduler,
   guidance, checkpoint configuration and image IO. Verify tokenizer/prompt
   semantics against the current NUI, including 75-token chunks, weighting,
   embeddings and slow-tokenizer text repair; fast-profile BPE is only a component.
2. Add the JVM worker/launcher behind the current protocol boundary and exercise
   real application services/MCP with it. Preserve the Python implementation as
   a comparison oracle during development; do not claim Python removal while it
   remains necessary to execute supported features.
3. Port every current request and built-in capability in the refreshed tables,
   including model management, image/video tools and diagnostics. Resolve the
   existing Python `run_code` extension and legacy Torch-format migration policy
   explicitly; do not silently drop those behaviors.
4. Replace interpreter/venv/pip bootstrap, packaging and maintenance dependencies.
   Validate from a Python-absent environment, through JavaFX and external MCP
   stdio/HTTP clients, with real SD1.5/SDXL and image inspection as NUI requires.
   Preserve settings, language support, metadata/replay and package behavior.

This is the completion criterion. Tiny models, standalone library tests, or a
single native text-to-image path cannot establish that the NUI target is finished.

On 2026-11-04 the worker distribution and native-only NUI assembly were rebuilt
from the current worktrees after the LoRA rank, block-weight and step-window
changes. The resulting Windows ZIP has SHA-256
`7872cf8601de44f9774fac22062c208dcfdcef07dd00967a6ac1bd8cc4cbdfd8`.
After extraction, `SDNUI.exe --mcp` completed a real SD1.5 CUDA 512x512
one-step DPM++ 2M request and exited successfully; no Python executable,
`.py` or `.pyc` files were present in the extracted package. A single
observation during generation showed 1350 MiB whole-device usage; this is not
a peak measurement. This smoke request did not apply LoRA and does not verify
interactive JavaFX operation or the remaining Python-dependent capabilities.

VRAM calibration identity now includes U-Net and VAE weight-offload settings
and VAE tiling, in addition to device placement and the attention limit. A
profile measured under different settings is rejected as incompatible, so old
profiles must be recalibrated. The identity compatibility regression and worker
protocol suite pass; this does not change the estimator's sampled-peak limits.

The CUDA VRAM calibration now records JVM OutOfMemoryError as an OOM row and
skips larger sizes in that mode/batch group, preserving the profile rather than
aborting the entire benchmark. An opt-in CUDA test injected the exception and
verified the persisted profile and all four mode groups; this was a simulated
failure, not a real exhausted-heap or GPU-OOM stress test. Under an actually
exhausted JVM, saving a profile may itself fail.

An independently extracted native-only NUI package completed SD1.5 CUDA
512x512, eight DPM++ 2M/Karras steps, with a trained LCM LoRA tag at 0.7.
The resulting barn image is in NUI `.superpowers/native-lora-512-20261104`;
the MCP output preserved the LoRA tag, but this run did not compare against a
no-LoRA image or a Python reference. Device-wide nvidia-smi sampling at roughly
0.5-second intervals saw a maximum of 1990 MiB, including desktop usage; brief
peaks can be missed. This package predates the later VRAM-estimator fixes and
Windows BOM settings fix.

The same SD1.5 512x512 seed and 8-step settings were run without the LCM LoRA
on the preceding package. The tagged and untagged images differ at 262140 of
262144 pixels (mean absolute RGB-channel difference 55.53/255), supporting that
the adapter path affects generation; this does not establish Python parity.
The native-only Windows package was then rebuilt with the subsequent VRAM and
UTF-8 BOM settings fixes (SHA-256
`cd6e25279f298b6b23872c8be8f2f51a592bb3371a4691e8ec4505582c0326d8`).
Its extracted EXE loaded BOM-prefixed model settings and completed a real
512x512 one-step SD1.5 CUDA MCP request. The extracted package contained no
Python executable, `.py` or `.pyc` files. This does not verify its JavaFX UI or
all remaining features.

Ordinary LoRA weights can now be read from restricted current-ZIP PyTorch
`.pt`, `.pth`, `.bin`, and `.ckpt` tensor dictionaries without running Python.
The reader also accepts the two additional metadata records written by recent
PyTorch versions and enforces a cumulative decoded-tensor limit. A fixture
created by PyTorch passed the native loader test. Tiny CUDA worker generation
and NUI Generator structured/tagged requests produced identical images for
matching `.pt` and safetensors factors, including dynamic rank. This supports
only the reader's restricted tensor/pickle subset, not arbitrary pickle code,
legacy non-ZIP formats, DoRA, or all real-world checkpoint layouts. The current
Windows ZIP predates this change.

The checkpoint VAE restore path now parses all tensor descriptors but only
materializes VAE values, while retaining SD1 metadata checks and streaming
validation of all ZIP entries. A standalone Java 25 reader probe on the official
SD1.5 `.ckpt`, in separate processes with `-Xms256m -Xmx10g`, measured calling
thread allocations of 4,363,075,288 bytes for the existing full reader and
371,844,528 bytes for the selective reader. Each run returned 1,145 entries.
These single-run allocation totals are not whole-process peak memory. The
restricted archive suite covers selective allocation, shared memoized references,
aggregate limits and cancellation. The existing Windows ZIP predates this change.

An isolated Java 25 `SingleFileSd1Model.loadVae` probe on the same official
checkpoint compared the previous packaged worker and the current worker, in
separate processes with `-Xms256m -Xmx10g`. One run each sampled process working
set every 100 ms: 4,732,878,848 bytes before and 753,954,816 bytes after.
Separately, 10 ms in-process heap samples peaked at 5,805,545,872 and
1,138,753,032 bytes. Both probes reported a loaded VAE. These are sampled,
single-run isolated-loader observations, not NUI process peaks or minimum RAM.

The restricted PyTorch LoRA reader now also accepts a `state_dict` wrapper
alongside primitive training metadata, including binary floating-point values.
A real PyTorch-written wrapped `.ckpt` with epoch and learning-rate metadata
passed the loader and NUI Generator CUDA tests; both structured and tagged
inputs matched the corresponding safetensors image on a tiny model. Unsupported
pickle objects and old non-ZIP archives remain rejected; the current packaged ZIP
predates these additions.

NUI's tiny CUDA Generator test now discovers the PyTorch `.ckpt` LoRA by
refreshing its actual model catalog rather than injecting a catalog entry. Both
structured and tagged requests match safetensors output. A rebuilt native-only
Windows ZIP (SHA-256 `facdb9e3f298a386ff706ab71250f37c433eac2557fb517560c42356b54ecdc7`)
was extracted and its bundled EXE completed a tagged wrapped-PyTorch-LoRA
64x64 two-step CUDA MCP smoke. The extracted payload contains no Python
executables or `.py`/`.pyc` files. The package smoke used a tiny model, not a
trained LoRA or JavaFX interaction.

A real 834-tensor trained LCM SD1.5 LoRA was converted to a current PyTorch ZIP
fixture for compatibility testing only. From the independently extracted
native-only Windows EXE, two sequential SD1.5 CUDA generations with identical
seed, prompt, weight and two DPM++ steps produced pixel-identical outputs for
the original safetensors and the wrapped PyTorch archive at both 64x64 and
512x512. The 512 comparison had 0 differing pixels out of 262144; sampled
whole-GPU usage peaked at 1866 MiB (roughly 0.5-second sampling, started after
the process launch). This checks one converted trained adapter, not arbitrary
PyTorch archives or Python inference parity; two-step images are smoke tests.

Safetensors archive filenames now accept case-insensitive extensions, matching
native catalog discovery on Windows. A LoRA loader regression and CUDA tests
pass, and NUI's catalog-to-Generator test now uses `test.SAFETENSORS` and
completes native CUDA generation. The currently published ZIP predates this fix.

Textual inversion `.bin` paths now reach the same restricted PyTorch reader as
`.pt` during generation. Previously the catalog and inspector accepted the file
but generation skipped it with a warning. A tiny SD1 native CPU generation test
confirmed no warning and changed output. The full backend suite was run before
the fix and reproduced the failure; it was not rerun in full afterward. CUDA,
NUI UI and the existing packaged ZIP remain unverified for this change.

The `.bin` textual-inversion path now passes tiny SD1 generation on both CPU and
CUDA, and NUI's typed client generated an image with a native-created `.bin`
embedding, verified its effect and confirmed it in the loaded embedding list.
The updated native-only Windows ZIP has SHA-256
`379cab0e236a75030c26363975b6cab90c4f8ad0a8a910b7e525d7f9d9c62f14`.
Its extracted EXE passed a tiny-model CUDA LoRA smoke and contains no `.py`,
`.pyc`, or Python executable. The packaged smoke did not exercise `.bin`
textual inversion or JavaFX interaction.

The extracted native-only ZIP was also exercised with a real `.bin` textual
inversion fixture on tiny SD1 CUDA. With identical seed and prompt, adding the
embedding changed 4084 of 4096 pixels at 64x64. An initial PowerShell harness
stopped draining stdout after the response and blocked the child on its output
pipe; a corrected external streaming harness drained stdout and stderr and
confirmed the EXE exited with code 0. Python was used only as the external
verification harness, not by the packaged NUI or worker. This is not a JavaFX
interactive test or a trained-model embedding parity test.

Standard txt2img Hires pixel upscalers now follow the pixel path: the base pass
is VAE-decoded, resized and center-cropped in image space, then re-encoded for
the second pass. `Nearest`, `Lanczos` and `None` are supported; previously
Lanczos incorrectly interpolated latent tensors and Nearest was rejected.
Tiny SD1 CUDA tests cover scale 1 and 2, distinct filters, and the NUI Generator
child-worker route. They do not establish trained-model or end-to-end Python
numerical parity. VRAM calibration identity now distinguishes the changed
pipeline; old profiles must be recalibrated. The current Windows ZIP predates
this change.

A real SD1.5 CUDA test now exercises NUI Generator's corrected pixel Hires:
256x256 base, Lanczos to 512x512, then a second pass with two nominal steps and
strength 0.5. It completed without Python in 91.8 seconds including loading;
the 512x512 image was inspected and is visibly coarse at these smoke-test step
counts. Sampling device-wide GPU usage every roughly 0.5 seconds observed a
1990 MiB maximum, including desktop usage; brief peaks may be missed. This
confirms an execution path on the 4GB GPU, not image quality or parity with
Python. The current ZIP predates the corrected pixel Hires implementation.

Historical measurements with mismatched CFG values (superseded below): independent Diffusers/Pillow tiny-SD1 two-pass pixel Hires references covered
Nearest and Lanczos from 64x64 to 128x128 using CPU-seeded noise. Both native
CPU and CUDA final PNGs matched each reference with maximum RGB-channel error
at most 3/255. The expected PNGs and their Python-only reference generator are
checked in; inference itself uses no Python. The stored-resource CUDA test was
rerun successfully. This checks two specific Euler cases, not every scheduler,
trained model, intermediate stage, or general perceptual quality.

Historical measurement with mismatched CFG values (superseded below): the tiny SD1 64-to-128 pixel-Hires references also covered Euler ancestral
with Uniform and DPM++ 2M with Karras, each with Nearest and Lanczos on CUDA.
Euler and Euler ancestral differ from the Diffusers/Pillow PNGs by at most
3/255 per channel; DPM++ differs by at most 5/255, with mean difference at most
0.533/255. DPM++ base PNG differs by max 2/255, and an isolated img2img pass
fed the identical Python-generated resized source differs by max 4/255 and mean
0.521/255. The cause of the residual remains unproven. These are bounded
regression comparisons, not bit-exact parity or trained-model evidence. CPU
reference coverage remains Euler only.

Correction to the preceding pixel-Hires parity measurements: the Python
reference used CFG 7.5 while the native test omitted the field and thus used
its default CFG 7. After explicitly setting 7.5 on both sides, tiny SD1 CUDA
output matches Diffusers/Pillow within 1/255 per channel across Euler, Euler
ancestral and DPM++ 2M with Nearest and Lanczos. The largest average difference
is 0.00269/255. The DPM++ identical-source second pass also matches within
1/255 on both CPU and CUDA. Its CUDA latent trace differs by at most 0.000018
at each of the two steps. Saved fixtures and tightened regression tests pass.
These checks cover the specified tiny model and schedule only; they do not
establish bit-exact or trained-model parity.

Pixel Hires center cropping now follows NUI's eight-pixel alignment rather
than the VAE scale, which can differ for nonstandard models. A tiny SD1
64x64-to-128x72 CUDA reference against Diffusers/Pillow failed before the fix
(max channel error 22/255 for Nearest) and passes afterward within 1/255 for
Nearest and Lanczos. NUI Generator also completed the 128x72 request on CUDA.
This does not establish parity for every aspect ratio or trained model; the
current Windows ZIP predates the fix.

Nine-channel txt2img now supplies the same gray input and full inpaint mask as
NUI, including for pixel Hires, where the resized image is encoded again with
a full mask. A CUDA test using a synthetic nine-channel model checks a 64-to-128
Nearest Hires run against a Diffusers reference within 1/255 per channel; the
second-pass latent trace differs by at most 0.000018. The reference explicitly
uses contiguous NCHW masked-image input: on this degenerate fixture PyTorch CPU
channels-last and NCHW convolutions produce noticeably different zero-image VAE
outputs. This is not proof of bit-exact parity with the ordinary Python pipeline
or of trained-model parity. Latent Hires for nine-channel models remains
unsupported, and the packaged NUI build predates this change.
The same full-mask conditioning is now used for nine-channel img2img without
an explicit mask. Its identical-source CUDA test initially differed by up to
47/255 and now matches the contiguous-layout reference within 1/255.

Nine-channel latent Hires now follows NUI's decode-first fallback: resize the
base latent at NUI's 8-pixel scale, crop, decode to pixels, and rerun full-mask
inpainting. It reports the same fallback warning. On a synthetic nine-channel
CUDA fixture, Nearest and bilinear 64-to-128 results and a Nearest 128x72 crop
match NCHW-contiguous Diffusers references within 1/255. NUI AppServices also
completed the Nearest case using the native worker with no Python environment.
This supersedes the earlier note that latent Hires is unsupported. References
use a nonstandard VAE scale and contiguous masked input, and do not establish
trained-model or packaged-application parity.

Align Your Steps (AYS) is now available as an Euler schedule, using the SD1
and SDXL ratios and NUI's interpolation, including its one-step behavior.
Euler a and DPM++ 2M use NUI's respective fallback schedules and warnings.
Reference tests cover both sigma tables at 1, 4, 11 and 20 steps, Euler image
parity on a tiny model, and both fallback paths. NUI AppServices generated a
512x512 SD1.5 image with Euler/AYS, 8 steps and seed 42 without Python; the
model-load-and-generation test took 168.771 seconds. Whole-device nvidia-smi
samples every ~0.5 seconds peaked at 1868 MiB, which is a sampled peak, not a
guaranteed maximum. The image `.superpowers/ays-sd15-fox-8step.png` was viewed.
No real SDXL AYS image or current packaged distribution was tested.

KL Optimal is now available for Euler, preserving NUI's nonzero terminal
sigma. Euler a and DPM++ 2M use NUI's existing sampler fallback and warnings.
Tiny SD1 CUDA tests match Diffusers references for txt2img and img2img within
1/255, and check the sigma/timestep vectors at four step counts. NUI
AppServices generated the reference txt2img without a Python environment.
No real-model KL image, SDXL image, inpainting or packaged build was verified.

The DDIM scheduler choice is now supported for Euler, Euler a and DPM++ 2M.
Euler uses NUI's custom discrete sigma list, while the other two use their
Diffusers leading-spacing equivalents. The requested step count is retained
separately because NUI's list may contain more steps than requested; img2img
strength slicing must use the requested count. Tiny SD1 CUDA tests cover all
three samplers, offsets 0/1, requested counts 1, 4, 7, 11 and 20, and a
7-step img2img request with 0.6 strength. Images match the reference within
1/255; NUI AppServices runs both txt2img and img2img with no Python runtime.
This is the scheduler choice, not the separate DDIM sampler, which remains
unsupported. No real model, SDXL, inpainting or current package was tested.

The DDIM sampler is now implemented separately from the previously supported
DDIM schedule. It supports epsilon/sample/v-prediction, checkpoint clipping,
NUI's schedule fallbacks, and seeded eta noise. Tiny SD1 CUDA tests compare
DDIM's step trajectories, txt2img, img2img, 4/9-channel full-mask inpainting,
and pixel/9-channel latent Hires with Diffusers. NUI AppServices also ran
stochastic DDIM with no Python runtime. A real SD1.5 512x512 eta=1, four-step
CUDA request completed in 94.863 seconds including model load; nvidia-smi
sampled whole-device memory every ~0.5 seconds, with a maximum of 1993 MiB.
The viewed `.superpowers/ddim-sd15-fox-4step.png` is visibly rough; this is a
smoke test, not a quality claim. Real SDXL, partial-mask DDIM, eta with Hires,
interactive JavaFX and the updated Windows package remain unverified.

DDIM partial-mask CUDA references now cover four/nine-channel U-Nets with
eta=0/1, matching within 1/255. Tiny SDXL DDIM txt2img and img2img with
eta=0/1 match Diffusers within 1/255 when NUI's own dual-CLIP embeddings are
used; NUI AppServices also ran both eta=1 paths without Python. Full SDXL
256x256 two-step DDIM eta=1 completed via NUI on CUDA in 96.151 seconds
including model loading. Whole-device nvidia-smi samples every ~0.5 seconds
peaked at 1513 MiB; this is not a guaranteed peak. The inspected output
`.superpowers/ddim-sdxl-apple-2step.png` is visibly distorted and establishes
only a launch/smoke result, not quality or real-model numerical parity.

The native worker now handles NUI `merge_models` without loading a model or
starting Python. Four merge modes, SD1/SDXL block weights and presets, VAE
source selection, channel-mismatched input convolution, and fp16/bf16/fp32
output are compared against NUI Python fixtures. The safetensors writer now
materializes entries lazily and preserves raw integer tensors; staged writes
are published only after the worker grants publication. NUI AppServices and
the model-merge plugin both exercised the native worker.
A 4.27 GB real SD1.5 safetensors checkpoint was merged with itself to a
2.13 GB fp16 checkpoint without Python, then reloaded in a separate NUI CUDA
worker and used for one-step 64x64 generation. Independent Python verification
compared all 1,145 tensors to the expected fp16 conversion with no mismatch;
Python was used only for the check. The tiny generated image is a smoke test.
Other checkpoint dtype variants, larger independent trained-model merges,
model conversion, package/UI acceptance and cancellation under real-model
load remain unverified. The existing packaged build predates this work.

The queued NUI plugin test exposed that model-tool output paths were treated
as temporary files and deleted after the job. On NUI branch
`feat/scala-torch-native-engine`, merge and conversion destinations now remain
durable while previews remain temporary; the native queued merge and the NUI
generator/job-queue regression tests pass. The native model catalog still
advertises only the worker's configured checkpoint, so selecting multiple
checkpoints for a merge through the normal NUI model list is not yet supported.

Investigation of model switching: `list_models` intentionally returns only the
constructor's canonical model root, and both `load_model` branches reject any
other path. The root also determines scheduler configuration, VAE restoration,
ControlNet compatibility, tokenizer paths, generation validation, hashing and
keyword lookup. Expanding the catalog alone would therefore expose selectable
models that cannot load. A switching implementation must bind these decisions
to the selected model together, preserve the previously published model if a
new load fails or is cancelled, and publish its status atomically: worker
requests are serialized, but `system_status` reads on the reader thread. Test
the complete NUI catalog -> selection -> load -> generation path, a failed and
cancelled switch, SD1 single-file and Diffusers directories, and memory use on
the 4 GB GPU. Scheduler and tokenizer selection for each model must also be
verified; simply mutating the current root would leave stale configuration.

An experimental `NativeSwitchingBackend` now creates a candidate backend per
load and publishes it only after the candidate has loaded successfully. Its
unit test checks failure and publication denial preserve the old backend. A
CPU integration test loaded two tiny Diffusers models in one process and ran
one-step image generation with each. The worker entry point now uses the wrapper.
Default scheduler paths follow the selected model; an explicitly configured
non-default path is retained. The worker now discovers safetensors checkpoints
and Diffusers directories under configured model directories, without exposing
the latter's component files as separate models. A NUI AppServices test used
the real native worker to discover two tiny Diffusers models, switch between
them, and generate a 64x64 one-step image with each without Python. Discovery
does not guarantee support for every checkpoint architecture. Real SD1.5
switching was exercised through NUI AppServices on a GTX 1650 SUPER (4096 MiB):
the original 4.27 GB fp32 checkpoint and its 2.13 GB fp16 merge were loaded in
one Python-free worker, with 64x64 one-step CUDA generation after each load.
Of 37 samples at roughly 0.75-second intervals, whole-device GPU memory
peaked at 1332 MiB (including a 1156 MiB baseline), worker working set at
11050192896 bytes and worker private bytes at 16905191424 bytes. These sampled
maxima are not guaranteed true peaks. The models have identical underlying
weights, so this establishes switching and execution, not a comparison of
distinct learned outputs. Retaining the previous model during loading can need
substantial host memory; switching independent models and interactive UI
acceptance remain unverified. The earlier NUI Generator regression also passed
with the switched worker (one SDXL test was skipped).

The native worker now accepts `convert_model` for safetensors checkpoints and
safetensors VAE inputs, covering precision selection, EMA selection, junk-key
removal, CLIP position IDs and VAE replacement. Publication is staged and
rejects input aliases; tests cover denied publication and NUI AppServices
dispatch without Python. PyTorch VAE inputs now use the bounded native
ZIP/pickle tensor reader; one small `state_dict` fixture passes. Against NUI
Python fixtures, 24 combinations of precision, EMA and optional edits match in
tensor names, shapes, dtypes and bytes. A real 4.27 GB SD1.5 checkpoint was
converted to fp16 with EMA kept, and every tensor was checked against a Java
fp16 cast or exact raw copy. A 248-tensor PyTorch VAE archive made from that
checkpoint was also baked into the real model; one bias was changed to verify
replacement, and every output tensor was checked. Python created the test
archive; the conversion itself used no Python. NUI's actual converter plugin
also passed through the job queue with a small PyTorch VAE, matched the Python
reference bytes, and retained its output after cleanup. Interactive UI
acceptance remains unverified.

A fresh native-only Windows ZIP (SHA-256
`811266150d192760b2ee90292033cc402b4bcf7a78b6e1b324a0396a7b2def04`)
was independently extracted and contained no Python executables, `.py` or
`.pyc` files. Its bundled EXE loaded the real SD1.5 single-file checkpoint on
CUDA and generated a 64x64 one-step image via MCP. The bundled worker also
converted the small reference checkpoint with a PyTorch VAE and produced four
tensors equal to the Python oracle. MCP deliberately refuses the converter
plugin because it writes model files, so this was a direct packaged-worker
protocol test; the NUI queue plugin test above used the development worker.
Interactive packaged UI and realistic image quality remain unverified. During
parallel sbt startup, one process reported an rt.jar access warning; both
assembly and distribution tasks reported success and the extracted package
passed these runtime tests.

The native worker now implements NUI's `memory` request. CUDA driver telemetry
is returned as `device_used_mb` (whole-device use, including other processes),
not attributed to the worker's allocation. NUI displays this separately as
GPU total; Python-backend responses retain the original VRAM display. CPU and
CUDA worker tests, legacy/new protocol decoding, NUI compilation, and NUI
client-to-CUDA-worker testing passed. On a real SD1.5 switch test, memory was
queried after each load and 64x64 one-step CUDA generation succeeded afterward.
Process-specific GPU allocation remains unavailable and retains the legacy
zero default. Windows x64 now reports actual process working set, peak working
set and private committed memory via the Win32 process memory counters, while
JVM heap remains separately labeled. The native working-set value was checked
against Windows `Get-Process`; NUI's client also received it from a real CUDA
worker. Unsupported platforms or API failures retain an unavailable status.
Interactive UI display and a rebuilt package have not been tested for this
change.

Float32, fp16 and bf16 safetensors serialization now streams from the tensor's
existing storage, respecting offset, reversed and broadcast views, without
materializing a second full-size float array. A 4,000,000-element fp16 write
allocated 21,387,776 bytes before the change and under 8 MiB afterward in the
thread allocation test. Writer tests and Python-reference merge/conversion
tests pass; conversion of the real 4.27 GB SD1.5 checkpoint with the modified
248-tensor PyTorch VAE passed full tensor comparison. This reduces temporary
host allocation during writing, not the two-model memory needed for a switch.
Windows process-memory polling now reuses its native library bindings instead
of opening a global-arena library lookup at every sample; 500 repeated samples
and the OS working-set comparison pass. The current packaged build predates
both changes.

Checkpoint merging now reuses exclusively owned float arrays from the reader
and updates the first input in place before streaming output. In a 4-million
element test with two float32 inputs, cumulative thread allocations fell from
183,680,728 bytes to under 48 MiB; this is allocation accounting, not a peak
RSS measurement. Python-reference merge tests pass. A real SD1.5 checkpoint
merged with itself to fp16 was byte-for-byte identical (SHA-256
`02162621470a0e1f050406d3d447d18498f9c60186a99ac652a2e1426fac2e11`)
to the previous result and generated a 64x64 one-step image through NUI CUDA.
The current packaged build predates this change.

The native worker now implements the `system_info` request used by NUI's
system information service. It returns native runtime information and local
checkpoint, LoRA, embedding, hypernetwork and VAE inventories, with opt-in
SHA-256 prefixes and safetensors embedding vector counts. CUDA snapshots now
also include the nested device-wide fields consumed by the memory chart.
Worker tests and NUI `SystemInfoService.full(hashes=true)` against a CUDA
worker pass. This is not complete Python system-info parity: the benchmark,
submission requests, comprehensive library/GPU details, all legacy checkpoint
formats, metadata generation and interactive UI remain unverified or
unsupported. The packaged build predates this change.

`system_benchmark` now runs the native generator with the existing benchmark
scene, batch presets and step presets. Temporary output is removed, cancellation
is checked, and failures stop subsequent batches; wrapped CUDA allocation
failures are classified as OOM. Model fingerprints omit the transient model
token and local path. Unit tests cover validation, cleanup, cancellation,
failure classification and stable identity. NUI's client ran the quick,
256x256, 10-step benchmark with a tiny model on GTX 1650 SUPER using VAE tiling;
the final run completed in 38.63 seconds including test setup. Without VAE
tiling the tiny-model VAE decode failed on this 4 GB GPU; the underlying failure
has not been resolved, and no real-model benchmark, live interruption test,
interactive UI or fresh package test has been completed. Legacy submission
requests remain unsupported.

The untiled tiny-model benchmark failure was traced to the configured total
attention-work limit: 256x256 output uses a 128x128 VAE latent, requiring
268,435,456 attention scores, above the former 134,217,728 worker default.
CUDA attention already slices its temporary workspace. Raising only the CUDA
VAE default to 268,435,456 permits this case; CPU limits and explicitly supplied
limits retain their former values. NUI's untiled 256x256, 10-step CUDA benchmark
now passes with default settings. An earlier run with the same raised limit
sampled peak whole-device usage of 1304 MiB across 79 samples; brief peaks may
be missed. This does not establish throughput or memory bounds for larger
models/resolutions. VAE work limit is included in system status and VRAM
calibration identity; existing profiles require recalibration. Larger work
limits can increase computation even though the attention scratch is bounded.
The Windows package predates this fix.

The native system benchmark subsequently ran through NUI on the real SD1.5
single-file checkpoint with CUDA U-Net/VAE weight offload, 256x256, 10 steps,
batch one. A second run was interrupted after its batch-start event, returned
an interrupted worker error, and was followed by successful 64x64 generation.
The complete test, including loading and recovery, took 93.397 seconds. Across
175 whole-device nvidia-smi samples, peak observed VRAM was 1504 MiB, including
an initial 1156 MiB desktop baseline; sampling can miss brief peaks. This is
one hardware/configuration test, not proof for larger batch presets, SDXL, UI
interaction or the packaged distribution.

Native multi-image generation currently loops over images sequentially. Benchmark
reports now identify this as `batch_execution: sequential`, and NUI labels it
accordingly; this is aggregate sequential throughput, not simultaneous batch
throughput. The same execution mode is part of VRAM calibration identity and
estimate results, so earlier calibration profiles require recalibration. True
parallel batching and its VRAM requirements remain unverified.

The sequential system benchmark now requests one image per iteration rather
than placing the requested image count in `batch_size`. This avoids the normal
generator's batch-size limit of 16 for the benchmark's 32 and 64 image presets.
A CUDA test on the 4 GB GTX 1650 SUPER generated 32 images with the tiny model
at 64x64 and one step; unit tests cover request construction through 64. The
full 32/64 benchmark presets and real-model throughput remain untested.

A new native-only Windows ZIP was built from the current NUI assembly and
scala-torch worker jars (SHA-256
`379cab5dc8f94d9c51663afa3b5f6aaf9a56618ee8eceb07fdc50449f7367a2f`).
After independent extraction, the file inventory contained no Python executable,
`.py` or `.pyc` files. The packaged launcher stayed running for 12 seconds;
the packaged worker and bundled Java runtime loaded the real SD1.5 Diffusers
directory and generated a 64x64 one-step image on CUDA with U-Net/VAE weight
offload. The image is a low-step pipeline smoke test, not a quality result.
This does not verify interactive JavaFX generation, MCP, benchmark presets,
SDXL or all features in the refreshed package.

The same extracted package was then exercised through `SDNUI.exe --mcp` using
a Node JSON-RPC client. It exposed 26 tools, loaded the real SD1.5 single-file
checkpoint and generated a 64x64 one-step image through NUI's generation path
on CUDA. Logs show the worker and Java runtime were resolved from the package.
The output is noise-like at this resolution and step count. This confirms the
headless packaged route, not GUI operation or image quality. Computer Use
requested app approval but reported that its approval dialog was unavailable,
so the interactive GUI test was not performed.

The native VRAM estimator now reports `measured_oom` instead of extrapolating
a numerical estimate when a square request exactly matches an OOM calibration
row for the same mode and image count. This preserves observed failure rather
than presenting a potentially misleading capacity estimate. A CUDA-backed
calibration test with simulated allocation failures verifies this for all four
calibration modes; it is not a real GPU OOM test. Rectangular requests, other
sizes and other image counts are not inferred from an OOM row. Device-wide
sampling and numerical extrapolation remain approximate, and the previously
built Windows ZIP does not include this change.

NUI's calibration history now considers an OOM-only profile measured, so a
newer OOM is not hidden by an older successful profile; empty or error-only
profiles still do not replace measured history. An NUI service integration
test with a tiny model and a CUDA worker saved a synthetic OOM profile, selected
it over an older success and received `measured_oom` from the native estimator,
without creating a Python environment. This verifies the service path, not the
interactive JavaFX screen or an actual GPU OOM. The existing ZIP predates this
NUI change as well.

The extracted native-only Windows package was exercised through JavaFX's
`--autogen` and `--screenshot` options, using the real SD1.5 single-file model
and CUDA with U-Net/VAE weight offload on the 4 GB GPU. A 256x256, 12-step Euler
request saved a recognizable red apple and the captured NUI window displayed
the image in its gallery with `Ready · cuda:0` and `Done · 1 image`.
Independent whole-device memory samples during generation were about 1.2–1.4
GiB; these sparse observations do not establish the peak. This is automated
JavaFX generation, not a manual mouse/keyboard interaction test. The package
still predates the subsequent VRAM OOM-handling changes.

The generation form no longer promises that batch-size images run in parallel;
it shows images per run and explains that execution depends on the engine.
Four focused UI tests passed. A test copy of the extracted native package was
updated with the current NUI and worker jars, then JavaFX autogen produced two
64x64 one-step SD1.5 CUDA images. Its screenshot shows `2 images per run × 1
runs = 2 images` and two gallery entries. This is a low-step smoke test and the
original ZIP was not rebuilt or modified.

The updated extracted-package test copy also ran JavaFX `--autogen` img2img
with the real SD1.5 single-file model on CUDA, using the previous 256x256 apple
image as input, 12 Euler steps and denoising strength 0.75. The generated image
was saved and displayed in the img2img gallery; the screenshot shows the input,
output, prompt, `Ready · cuda:0` and `Done · 1 image`. The prompt requested a
green apple but the output remained red, so this establishes the execution and
display path, not successful color control or general image quality. The test
used bundled Java/worker without Python and did not exercise manual UI input.

An MCP inpaint check on the same updated extracted-package test copy sent the
real SD1.5 model a 256x256 source image and a hard half-image mask, with CUDA
offload, 12 Euler steps and strength 0.75. Comparing all pixels found zero
changes in the protected left half and 32,762 changed pixels in the right half.
The requested green apple remained red and the mask boundary is visible, so
the result establishes hard-mask preservation and editing through packaged
NUI's headless route, not visual quality or prompt adherence. This does not
exercise drawing a mask in JavaFX, and the original ZIP remains unchanged.

With the same hard source mask but `mask_blur=8`, another extracted-package
MCP check generated at 256x256 with four steps. Pixel comparison found zero
changes in columns 0–111 and 3,107 changed pixels out of 4,096 in columns
112–127 immediately on the nominally protected side of the boundary; columns
192–255 were all changed. This demonstrates that feathering propagates into
the composited output while distant pixels remain unchanged for this case.
The generated area is visibly distorted at four steps; no visual-quality or
full numerical parity claim follows from this test.

Native VRAM calibration now stops the affected mode/batch size group after any
case failure, including non-OOM errors, and records `stopped_after_error`.
Previously such errors continued through larger sizes and could leave a
misleading `complete` profile. This follows the existing NUI Python engine's
group behavior. CUDA-backed tests with simulated non-OOM and OOM failures,
profile identity and OOM estimates passed; they do not induce a real OOM.
The existing Windows ZIP predates this fix.

A refreshed native-only Windows ZIP contains the current NUI assembly and
scala-torch worker, SHA-256
`71f914417c01c4d0dd861e005d6af7890363751025782035819470f91d9e0207`.
The independently extracted app has no Python executables, `.py` or `.pyc`
files, and its NUI and worker JAR hashes match the current builds. Its bundled
launcher generated through MCP with real SD1.5 on CUDA, and JavaFX autogen
produced two 64x64 one-step images with the corrected batch label visible.
The NUI CUDA OOM-profile integration test also passed using the extracted
worker classpath; that test used a synthetic OOM record. These checks do not
establish full feature parity or manual UI operation.

The refreshed native-only package also ran JavaFX autogen with the official
SDXL Diffusers model on the 4 GB CUDA device with U-Net and VAE offload. The
256x256 four-step output was saved and shown in NUI with `sdxl-full`,
`Ready · cuda:0`, and `Done · 1 image`. The output is severely distorted;
this is an execution test, not a usable-quality result. A few whole-device
samples were around 1.2–1.4 GiB and the worker working set was about 17 GB;
neither is a measured peak. Older, separate 512x512/16-step runs described
above produced a recognizable image and matched the CPU reference; this
current-package GUI run does not repeat that quality or parity test.

The refreshed native-only ZIP then generated the earlier SDXL baseline through
packaged MCP with 512x512, 16 DPM++ 2M Karras steps, seed 42 and CFG 7.5.
The output is a recognizable apple; all 262,144 RGB pixels match the earlier
native reference image. Elapsed time including startup/model loading was
515.957 seconds on the 4 GB GPU; this is one run, not a speed benchmark.

The first attempt exposed a separate SDXL host-memory failure: startup restored
the last checkpoint and a subsequent MCP `set_model` reloaded the same model,
temporarily retaining both. MCP now reuses an already loaded checkpoint by
default and supports `reload: true` to explicitly reread changed model files.
The existing VAE is retained on reuse unless a VAE is explicitly selected;
different checkpoint loads still preserve the old model on failure. The MCP
suite (31 tests), a tiny CUDA model-token integration test, and an actual SDXL
repeat-selection followed by 64x64 CUDA generation passed. The SDXL check used
an extracted test copy with its NUI JAR updated; the existing ZIP predates this
MCP fix. Forced reload of SDXL may still require memory for two host models.

Follow-up MCP checks found that a stopped engine still retains its last model
record. Same-model reuse now requires a live, ready engine; a stopped engine no
longer reports a successful selection based solely on stale state. Explicit
VAE type and catalog checks also occur before switching checkpoints. Tests
cover stopped-engine behavior, retaining/changing VAE, malformed and missing
VAEs without an unintended model switch, and native CUDA reuse; 34 MCP tests
passed. The existing ZIP predates these follow-up corrections.

MCP `set_model` now constructs its response from the model returned by its own
selection while holding the selection lock, rather than rereading shared model
state after releasing that lock. A concurrent 100-pair fake-engine test and the
native CUDA reuse test pass, as do all 35 targeted MCP tests. The race was not
observed in the pre-fix stress test; this is a code-level mitigation, not a
reproduced failure or proof against all cross-interface model-switch races.
The distribution ZIP predates this change.

ControlNet caching now tracks the config, safetensors archives and shard index
file metadata instead of only the containing directory's metadata. It compares
the component snapshot before and after loading, so observed changes during a
load are rejected before caching. A tiny-model generation regression reproduced
stale cache reuse with a changed config and now passes on CPU and CUDA; CUDA
coverage also changes the single archive and a shard while restoring directory
mtime. This uses size, modification time and file identity, not a content hash,
so deliberate same-metadata mutations may still evade detection. Full trained
ControlNet and packaged-application coverage for this change remain open.

Native file discovery now lists a configured ControlNet model directory itself,
instead of exposing only its internal weight file when that directory is the
search root. A Python-free NUI AppServices/Generator integration test used a
directly configured tiny ControlNet on CUDA, generated an image, then corrupted
the copied weight archive and confirmed the next request fails rather than
using the previous cached model. Native catalog tests passed (one unrelated
model-switch test skipped). This does not validate a trained ControlNet or the
interactive selection widget; the existing ZIP predates the fix.

The standard trained `lllyasviel/sd-controlnet-canny` Diffusers archive exposed
an invalid loader constraint: the conditioning stem's final intermediate width
is 256 while the U-Net's first width is 320, and its `conv_out` legitimately
projects between them. Removing the equality requirement lets the already
shape-validated projection load. A separately generated Diffusers tiny fixture
with differing intermediate/output widths matched native CPU and CUDA prediction
references in four modes (maximum absolute error 0.000039). The model test run
reported 334 passed and 163 skipped. NUI AppServices then loaded the real SD1.5
and trained Canny ControlNet on the 4 GB CUDA device, generated controlled and
uncontrolled 256x256, four-step images with the same seed, and observed distinct
outputs. The controlled image visibly follows the input apple's outlines, but
the short run is not full-quality or numerical parity evidence. The downloaded
weights SHA-256 is `e19821a00e6d1817b37286a21d5c4f8915076949b0e81846c4f92c96ffb46db7`;
model source: https://huggingface.co/lllyasviel/sd-controlnet-canny. The
existing ZIP predates this fix and the desktop selection flow is unverified.

An updated extracted-package test copy was then run through the NUI JavaFX
`--autogen` path with the trained Canny ControlNet configured directly. The
full form screenshot shows one enabled unit, the Canny preprocessor, input
image and the correct model name. The generated 256x256, four-step image was
displayed in the gallery with `Ready · cuda:0` and was pixel-identical to the
prior NUI service result under the same settings. Form state was supplied via
the saved settings file; manually choosing the model was not exercised. The
original ZIP was not rebuilt for this test.

NUI's Generator was exercised with two ControlNet units on the 4 GB CUDA GPU.
For the same model and input, two units at weight 0.5 produced identical RGB
pixels to one unit at weight 1.0. This held for tiny-model base and latent
Hires passes, and separately for the trained SD1.5 Canny ControlNet at 256x256
and four steps. The real-model test took 147 seconds for three generations;
individual GPU/host observations were about 1.3 GB device-wide and 8.5 GB
worker working set, not measured peaks. Different ControlNet models and real
model Hires composition remain unverified.

A native-only Windows ZIP including the ControlNet loader and catalog fixes was
built with SHA-256
`83e51e2dcd0e326d9a92d8145a3c905627e60722df884e94725c888beb2de97c`.
Its independently extracted worker JAR matches the source build, and its file
inventory contains no Python executable or `.py`/`.pyc` files. NUI AppServices
using that extracted worker ran the real SD1.5 and trained Canny ControlNet on
CUDA: one full-weight unit matched two half-weight units and differed from the
uncontrolled generation. This used the NUI test harness and extracted worker,
not the packaged GUI or packaged MCP launcher.

The independently extracted ControlNet ZIP's unmodified `SDNUI.exe` was then
run with JavaFX `--autogen`, saved SD1.5 and trained Canny ControlNet settings,
256x256, four steps, CUDA U-Net/VAE offload. It saved and displayed the output;
the full form screenshot shows the enabled Canny unit and selected model, and
the main screenshot shows `Ready · cuda:0` and `Done · 1 image`. This tests
the packaged GUI generation route without Python, but uses restored form state
rather than manual model selection or mouse interaction.

The SD1 single-file signature check and U-Net configuration now accept both
four-channel and nine-channel input convolutions. A synthetic safetensors
signature test checks both configurations and rejects a five-channel variant.
The trained SD1.5 inpainting single-file checkpoint from
https://huggingface.co/webui/stable-diffusion-inpainting/blob/c510a603e876bb6dbf72ba7abeab428d3f5104d0/sd-v1-5-inpainting.safetensors
was verified against SHA-256
`0ec8f8585b104417a8c34a9fbcc1e922a70b8c15490ec4553087f01c8cf33673`.
On the 4 GB GTX 1650 SUPER, both the native worker test and NUI AppServices /
Generator loaded it and generated a 256x256 partial-mask inpaint using CUDA
weight offload, Euler, eight steps, and 0.75 strength. Both tests confirmed
that every pixel outside the mask was preserved and some pixels inside changed;
the output was visually inspected. The NUI test confirmed the worker reported
CUDA and no Python installation was created in its app home. This verifies the
NUI service path, not the packaged GUI. The released ZIP above predates this
change; peak VRAM was not measured.

Nine-channel SD1 inpainting can now be combined with a standard four-channel
SD1 ControlNet. The U-Net receives the full inpaint sample while the ControlNet
receives its first four latent channels, as in Diffusers' ControlNet inpaint
pipeline. Compatibility checking accepts four- and nine-channel bases while
still checking the remaining U-Net architecture. Unit tests cover batched
channel extraction and incompatible configurations. A real-model test that
previously failed on the ControlNet input shape passed on the 4 GB GTX 1650
SUPER, both directly and through NUI AppServices / Generator: the trained
single-file SD1.5 inpaint checkpoint and trained Canny ControlNet produced
different masked pixels from uncontrolled generation and preserved every
unmasked pixel at 256x256, eight steps and strength 0.75. The NUI test reported
CUDA and used no Python installation. The image was inspected; this is not a
numerical Diffusers parity test, a packaged GUI test, or a peak VRAM reading.
Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py

A native-only Windows ZIP containing that worker was built with SHA-256
`b91c181d951bafb5aacb21c2e8742fc035d566ccfc8f8b0a259a6f0f45e6c75c`.
An independent extraction contained no Python executable or `.py`/`.pyc`
files, and its worker model JAR matched the freshly built distribution. The
extracted `SDNUI.exe`, using restored form settings and a demo-painted mask,
generated an SD1.5 single-file inpaint with the trained Canny ControlNet on
CUDA at 256x256, eight Euler steps, strength 0.75. Its screenshot shows the
inpaint mask, generated gallery image, `Ready · cuda:0` and `Done · 1 image`;
the full form shows the enabled Canny unit. The generation history confirms the
selected model and ControlNet; the saved image differs from its source in
5,894 pixels within bounding box (54,92)-(189,164). Manual model selection and
painting were not exercised. One-second device-wide samples taken after startup
observed a maximum of 1,519 MiB over 120 samples; this is not a full-run peak.

Standard original-format SD1 ControlNet `.safetensors` files can now be loaded
directly, without converting to a Diffusers directory. The loader maps the
original tensor names lazily, enforces the SD1 architecture and strict weight
shapes/consumption, and invalidates its cache when the file stamp changes.
All 340 names of a real ControlNet v1.1 Canny file matched a separately
generated Diffusers converter reference. Native CUDA and NUI AppServices /
Generator tests used that trained FP16 single-file ControlNet with the trained
nine-channel SD1.5 inpaint checkpoint, at 256x256 on the 4 GB GTX 1650 SUPER;
the NUI catalog resolved the file by name, masked output changed versus no
ControlNet and every unmasked pixel was preserved. The output was inspected.
Converting the same weights with Diffusers' reference converter and running
the resulting directory through the same native CUDA pipeline produced
pixel-identical output to the original single-file path at those settings.
ControlNet source: https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors
(`control_v11p_sd15_canny_fp16.safetensors`, SHA-256
`8932b66e15aae835b3490dbf989f56c253104cee08a88bf21283762f557c9f10`).
This covers standard original SD1 ControlNet safetensors, not arbitrary
ControlNet architectures, legacy pickle checkpoints, or the prior ZIP.

The original SD1 ControlNet loader also accepts the standalone layout without
the `control_model.` prefix. All 340 names of the real v1.1 Canny fixture match
the Diffusers mapping with or without the prefix; mixed-prefix archives are
rejected. A test-created prefixless checkpoint retained the real trained tensor
bytes and produced pixel-identical CUDA output to the same weights converted
to a Diffusers directory, using the trained SD1.5 inpaint model. A separate
temporary copy was corrupted after it had been cached; the next request failed
instead of using stale cached weights. This tests size-changing corruption, not
undetectable in-place edits that preserve the full file stamp.
NUI AppServices / Generator also resolved the exact prefixless file through its
catalog and generated the same masked test without Python on CUDA; unchanged
pixels outside the mask and changed pixels inside were checked. The generated
image was inspected. The existing ZIP predates this prefixless support.

Standard SD1 Diffusers-format ControlNet weights also load as a single
`.safetensors` file without a sibling `config.json`. The file path assumes the
standard SD1 ControlNet settings; tensor names, shapes, parameter budget and
complete consumption remain checked, but settings that cannot be inferred from
weights are not verified against a separate config. Mixed original and
Diffusers tensor names are rejected. The trained v1.1 Canny weights, converted
with Diffusers and then copied to an otherwise config-free model directory,
were selected by the NUI catalog and generated with the trained nine-channel
SD1.5 inpaint model on the 4 GB CUDA GPU. The mask-preservation and controlled
versus uncontrolled checks passed and the output was inspected. A separate
native CUDA comparison found the same single-file weights produced pixel-identical
output to their Diffusers directory at 256x256, eight Euler steps and strength
0.75; cache invalidation after size-changing corruption was also exercised.
The pre-existing Windows ZIP does not contain this latest change.

Standard SD1 ControlNet `.pt`, `.pth`, `.ckpt` and `.bin` paths now use the
restricted PyTorch ZIP tensor reader, with model-sized archive, tensor and
parameter limits, then the same strict name and shape checks as safetensors.
This supports direct tensor dictionaries and `state_dict` wrappers; it does not
execute pickle code and does not support legacy non-ZIP archives or arbitrary
PyTorch objects. The reader's 11 tests passed, including rejection of an
unallowlisted reducer, unsafe ZIP paths and noncontiguous tensor views.

The original published FP32 `control_v11p_sd15_canny.pth` from
https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth
was verified against SHA-256
`f99cfe4c70910e38e3fece9918a4979ed7d3dcf9b81cee293e1755363af5406a`.
On the 4 GB GTX 1650 SUPER, the native CUDA worker and NUI AppServices /
Generator each loaded it and generated a 256x256 partial-mask inpaint using a
trained nine-channel SD1.5 base, eight Euler steps and strength 0.75. Both
confirmed mask preservation and a difference from uncontrolled generation;
the NUI catalog resolved the exact `.pth` file and the output was inspected.
An independent reference conversion of the same FP32 tensors to Diffusers
safetensors produced pixel-identical output through the native worker. During
one run the test JVM working set was observed at about 10.4 GiB; this is one
observation, not a peak measurement. The current reader materializes archive
storage and float tensors on the host, so reducing host RAM remains open. The
existing ZIP predates this change; packaged GUI acceptance remains unverified.

The restricted PyTorch ZIP reader now validates declared archive sizes first,
decodes each referenced contiguous tensor directly from its ZIP stream using a
64 KiB buffer, and drains unreferenced entries without retaining their data.
It still checks actual entry lengths, tensor bounds and aggregate limits, and
supports multiple views of a shared storage. Fifteen reader tests passed,
including malformed archives, unsupported reducers, cross-buffer offsets,
shared storage, total limits and allocation bounds. A 16 MiB referenced tensor
allocated under 24 MiB on the reading thread; a 16 MiB unused storage allocated
under 4 MiB while still enforcing aggregate limits.

An isolated read of the same official 1.45 GB Canny `.pth`, in separate Java
25 processes with `-Xms256m -Xmx6g` and 10 ms heap sampling, observed peak heap
usage of 3,257,037,272 bytes with the previous packaged reader and 1,625,401,536
bytes with the streaming reader. These are one run each, not a statistical
benchmark or full-application memory peak. In separate full test runs involving
three generations and two ControlNet models, one-second process working-set
samples peaked at approximately 10.90 GiB with the first lazy-reader revision
and 11.81 GiB with direct streaming; those observations do not establish a
reduction in end-to-end memory, which is affected by heap/GC and model lifetime.
The real CUDA worker still produced pixel-identical output to the independent
FP32 safetensors conversion, and NUI AppServices / Generator again passed the
same `.pth` inpaint test without Python on the 4 GB GPU. The packaged ZIP was
not rebuilt for this change.

The PyTorch ControlNet tensor view now consumes each decoded source array once
and drops its reference after making the model tensor's defensive copy. A unit
test verifies both one-shot consumption and that mutating the source array
cannot mutate the copied tensor. With the original published 1.45 GB Canny
`.pth`, isolated `PretrainedControlNet.load` probes on Java 25 with `-Xms256m
-Xmx2g` showed one previous-reader run failing with Java heap exhaustion while
one consuming-reader run succeeded (10 ms sampled peak heap 2,104,831,176
bytes). With `-Xmx4g`, one run each observed 3,175,471,904 and 3,167,250,720
bytes respectively; that setting did not show a meaningful peak reduction.
These are model-loader probes, not whole-application memory requirements.

The 4 GB CUDA worker again generated pixel-identical output to the independently
converted safetensors reference, and NUI AppServices / Generator successfully
ran the real `.pth` inpaint test without Python. NUI worker working-set samples
started after process launch and peaked at 10,579,374,080 bytes over 93 one-second
samples; this is not a full-run peak or a controlled before/after comparison.
The generated image was inspected. The existing ZIP predates this change.

The restricted PyTorch ZIP reader now accepts a cancellation callback and
checks it during ZIP entry enumeration, bounded metadata reads, tensor streaming,
pickle interpretation, object conversion and validation of unreferenced
entries. Existing two-argument callers retain their API. The ControlNet loader
passes its work-context cancellation callback into the reader. Seventeen reader
tests passed, including cancellation during referenced and unreferenced storage
reads followed by a successful reread.

A real official Canny `.pth` test used the native worker's JSON interrupt
protocol and a deterministic test latch to cancel its ControlNet loader, then
loaded the same file successfully on the same worker. A separate real CUDA
`NativeDiffusionBackend` test cancelled during streamed tensor reading, checked
that no output was published, and then generated successfully on the same
backend; the result remained pixel-identical to an independent FP32 safetensors
conversion. The worker protocol test uses a small test backend around the real
loader, while the generation test injects cancellation directly into the real
backend context; an interactive NUI cancel action during `.pth` loading is not
yet verified. The native worker distribution was rebuilt, but the Windows ZIP
predates this change.

The native worker now emits `Reading ControlNet tensor weights` once, after the
first tensor bytes of a restricted PyTorch ZIP archive have been read. The
reader's 18 tests include the one-shot notification and cancellation triggered
by it. A real NUI AppServices / EngineClient test loaded the trained SD1.5
inpaint base and official Canny `.pth` on the 4 GB GPU, waited for this event,
sent the generation handle's interrupt, observed the interrupted result and no
published output, then retried successfully on the same running worker; its PID
was unchanged before interruption and after retry. The
retry emitted the reading event again, confirming the interrupted model was not
published into the cache; the generated image was inspected. This exercises
NUI's actual child-worker transport rather than a mouse click on the JavaFX
button. The worker distribution was rebuilt; the existing Windows ZIP predates
this change.

A refreshed native-only Windows ZIP including the PyTorch ControlNet reader,
streaming memory changes and cancellation notification has SHA-256
`2698b800881819083033e3f078463412b5cae721eaec459daa3b40e6dcd57ec7`.
Its independently extracted inventory contains no Python executable or
`.py`/`.pyc` files, and the models and IO worker JARs match the source build.
NUI AppServices / EngineClient with the extracted package's worker and bundled
Java 25 interrupted the real Canny `.pth` on its first tensor-data event, saw
no published output, then generated successfully on retry with the same worker
PID. This exercises the extracted worker through NUI's test harness, not the
packaged launcher's UI interrupt button.

The unmodified extracted `SDNUI.exe` was also run with restored settings,
`--demo-mask` and `--autogen`: on the 4 GB CUDA GPU it generated a 256x256,
four-step inpaint using the real SD1.5 inpaint checkpoint and the original
trained Canny `.pth`. The saved form and history show the ControlNet enabled;
the main screenshot shows `Ready · cuda:0` and `Done · 1 image`. Inspection
found 5,892 changed pixels in bounding box (54,92)-(189,164) versus the source.
Manual selection, mask painting and UI-button cancellation were not exercised.

For restoring the original VAE from an SD1 PyTorch checkpoint, the loader now
releases its decoded non-VAE tensor references before constructing the VAE.
NUI's native service loaded the official SD1.5 `.ckpt` on the 4 GB GPU, set an
external VAE, restored the checkpoint VAE, and reproduced the prior image
pixel-for-pixel. Nine focused mapping and checkpoint tests also passed. This
change permits earlier reclamation but does not avoid initially decoding the
whole checkpoint; its effect on peak memory has not been measured. The existing
Windows ZIP predates this change.

ZIP CRC32 validation now covers pickle metadata, selected tensor storage and
unselected entries. Corrupted stored-entry fixtures fail whether the damaged
tensor is selected or unselected. The official SD1.5 `.ckpt` passed full and selective parsing;
the original ControlNet Canny `.pth` passed a real CUDA inpaint and matched the
independent safetensors reference. By default, archives created with PyTorch's
optional CRC32-disabled serialization are rejected; see
https://docs.pytorch.org/docs/stable/notes/serialization. The explicit setting
`SDFX_SCALATORCH_ALLOW_MISSING_CRC=1` accepts an archive only when all its CRC
fields are zero; other archives still have their CRCs checked. Since a missing
CRC cannot prove integrity, the file must be verified independently. A genuine
PyTorch 2.7 CRC-disabled fixture passed the compatibility tests. A copy of the
official SD1.5 checkpoint with all central-directory CRC fields zeroed passed
NUI generation and VAE reset on the 4 GB GPU; its output pixels match the
original checkpoint. The previously built Windows ZIP does not include this.

Approximate live previews now select the SDXL channel-to-RGB projection for an
SDXL pipeline instead of applying the SD1 projection to all models. Independent
pixel vectors from NUI's Python preview implementation match for both SD1 and
SDXL, and a tiny SDXL CUDA generation on the 4 GB GPU emitted two decodable
intermediate JPEG previews.

TAESD and TAESDXL neural decoders now load the pinned external safetensors
weights, use CPU or bounded CUDA according to the VAE execution setting, and
fall back with a warning when unavailable. Decoder outputs match independent
Diffusers fixtures for both trained models; CUDA cancellation and subsequent
recovery pass. Full SD1.5 and SDXL NUI service tests received neural previews
on the 4 GB GPU. A standalone TAESD CUDA 512x512 decode took 0.91 seconds in
one run; this excludes model loading and is not an end-to-end generation
benchmark. TAESD VAE Tools uses the pinned TAESD/TAESDXL safetensors through
the same native codec, with opt-in, hash-checked downloads into the local
cache; unit coverage verifies cache reuse and that downloads require explicit
permission, while a separate opt-in network test loads both official weight
sets. The VAE Tools integration was also run through the NUI service with the
Python-free worker, and `vae_tools` is now advertised in the native hello so
the UI does not mark the plugin unsupported. CMYK JPEG100 remains unsupported
with JDK-only image codecs; CMYK TIFF remains available. The previously built
Windows ZIP does not include the latest feature advertisement.

Preview conversion and delivery failures are now isolated from generation;
cancellation continues to propagate. A CUDA worker test injected a preview
delivery failure and still produced the final image. With trained TAESD weights,
cache tests replaced a loaded weights file with invalid bytes, verified that the
stale model was not reused, restored it, and checked cancellation/recovery.
The NUI service also ran the official SD1.5 model with a deliberately damaged
TAESD file on the 4 GB GPU: it logged the fallback warning, delivered three
approximate previews, and saved an image pixel-identical to the normal run.
The existing Windows ZIP predates the preview-failure isolation fix.

## Packaged NUI RealESRGAN extras verification (2026-11-05)

Closed the catalog naming mismatch in `NativeStandaloneUpscale`: NUI's
`RealESRGAN_x4plus` and `RealESRGAN_x4plus.pth` names map to the native
`R-ESRGAN 4x+` implementation; original catalog names continue to resolve
model paths. Regression coverage passed 3/3.

An actual packaged native-only NUI MCP `upscale_extras` request processed the
768x768 seed-4245 fox image with RealESRGAN x4 at scale 1.5. The scala-torch
worker ran on `cuda:0` and saved a visually coherent 1152x1152 PNG, with no
obvious tile seams. It took approximately 23 minutes on a GTX 1650 SUPER with
4 GiB VRAM; observed GPU memory was about 1.5 GiB. The bounded 128px tiles make
this workload fit, but latency and MCP's static 0% progress are clear usability
limitations. This verifies this one plugin/model/size path only, not full NUI
parity. The app image was temporary and no release ZIP was produced.

## RealESRGAN fast CUDA convolution (2026-11-05)

The actual NUI `upscale_extras` path initially took about 23 minutes for the
768px fox image because every RRDB convolution used the exact/direct CUDA
kernel. `CudaRealEsrganX4Plus` now selects `CudaSession.conv2dFast` by default;
the direct path is still available to compare outputs. An opt-in suite using
the official RealESRGAN checkpoint and a deterministic 32px source passed 1/1,
with max channel error <=1/255 between complete fast and direct model results.

The same packaged NUI MCP request completed in 206.4s on the 4 GiB GTX 1650
SUPER: 768x768 input, `RealESRGAN_x4plus`, requested 1.5x output. The saved
1152x1152 image differs from the direct output by at most 1/255 per channel
(mean 0.000111; zero channel values differ by more than one level), and visual
inspection found no obvious seams. Observed VRAM was about 1.7 GiB. This is
about 6.7x faster than the prior 23-minute run. The 128px core tile is retained;
a 256px tile exceeded the current 1 GiB per-session allocation budget. This is
a measured single-device improvement, not a comprehensive benchmark or parity
claim for other upscalers.

## RealESRGAN Anime 6B support (2026-11-05)

The RRDB loader now detects and validates both supported model depths: the
existing 23-block RealESRGAN x4plus and official six-block Anime 6B. CUDA
inference uses the detected depth, and NUI's `RealESRGAN_x4plus_anime_6B`
catalog entry resolves for standalone Extras and img2img Hires Fix. The real
official Anime 6B checkpoint passed the six-block loader assertion. NUI's
native-only packaged MCP then completed `upscale_extras` on CUDA with a 64x64
source and 1.5x output (96x96); startup plus job took 8.4 seconds. This is a
small smoke image only, not a full-size benchmark or image-quality assessment.

## Native-only txt2img HiresFix with Anime 6B (2026-11-05)

The packaged NUI MCP path completed a real SD1.5 txt2img request at 256x256,
12 Euler steps, followed by the registered `img2img_hires_fix` AlwaysOn plugin
using `RealESRGAN_x4plus_anime_6B`, 1.5x scale, 8 refinement steps and 0.25
denoise. It reported `cuda:0`, reached `img2img_hires step 2/2`, wrote 384x384,
and retained the selected upscaler/settings in infotext. End-to-end packaged
startup and generation took 107.6 seconds; a concurrent `nvidia-smi` sample
reported 1962/4096 MiB. Output SHA-256 is
`E9E127B7B389FE9AA54058983CDBBD1D8CDE7D05FE54504F8329FB7084AE2A38`.
The visual is a coherent stylized fox, but this is one low-resolution sample,
not broad quality/parity evidence. Other pixel-upscaler architectures and
masked latent Hires remains open; pixel HiresFix after masked 4-channel img2img was verified on a real NUI MCP path.

## Masked img2img/inpaint with pixel HiresFix (2026-11-05)

Native-only packaged NUI MCP completed an actual masked img2img request using
the SD1.5 4-channel Diffusers U-Net, followed by the Anime 6B pixel upscaler
and refinement. It used a 384x384 input and white elliptical mask, 8 base Euler
steps, 0.75 denoise, mask blur 4, then 1.25x HiresFix with four steps at 0.2
denoise. The worker ran on `cuda:0`, reached `img2img_hires step 1/1`, and
produced a 480x480 PNG. Startup plus job took 115.7 seconds. Sampled GPU
telemetry during execution was 1846-2050/4096 MiB (87-95% utilization), not a
peak measurement. SHA-256:
`484C3E3C93F767D7FE594A7C2E27971229D1344831716D9ED9A6605A79ACF4B6`.
This verifies one masked 4-channel SD1.5 pixel-HiresFix path. Official nine-channel inpaint is covered below; masked latent Hires and other crop shapes, aspect ratios and parity remain unverified.

## Official nine-channel SD1.5 inpaint through native-only NUI (2026-11-05)

The worker initially rejected Diffusers' `StableDiffusionInpaintPipeline`
model-index name. A focused `NativeInpaintPipelineFamilySuite` reproduced
`unsupported diffusion model family`; accepting this SD1 pipeline class made
the regression test pass. The worker distribution was rebuilt from the fix.

Downloaded the official `stable-diffusion-v1-5/stable-diffusion-inpainting`
FP16 U-Net (1,719,154,104 bytes) into a temporary model directory, with the
existing SD1.5 text encoder, tokenizer, VAE and scheduler linked in. Native-only
packaged NUI MCP launched Scala-Torch directly with that model path and ran a
384x384 masked img2img request, full-resolution crop/paste with padding 32,
then pixel HiresFix at 1.25x with four refinement steps at 0.2 denoise, with no
Python process involved. It
completed `img2img_hires step 1/1` and saved 480x480. Startup plus generation
took 132.8 seconds. Concurrent GPU samples were about 1.9/4 GiB, not a measured
peak. Output SHA-256:
`9F5149E1D89E6D08233AA809F45E505C977CF95B865BC03A0BE468E38AA5D205`.

The sample was visually inspected, but this is one prompt, mask, square size
and low-step run, not a quality/parity benchmark. The explicit-upscaler error
was traced to the async first request being parsed before the engine catalog was
ready. NUI now defers dynamic-choice membership validation only while the list
is empty; populated catalogs still reject unknown values.

Rebuilt and packaged native-only NUI and tested a fresh MCP home with no catalog
lookup before the first request. The first `img2img_async` request explicitly
selected `Lanczos`, was accepted as `waiting_for_engine`, then completed the
official nine-channel full-resolution masked inpaint plus 1.25x HiresFix. It
reported `img2img_hires step 1/1`, saved 480x480 in 79.4 seconds including
startup, and ran without Python. Output SHA-256:
`B37EB312B9BF3E2011E819A1784011104E19E437BE1F2C5ED4C6ABED78F3BA23`.

The focused validation test passed 5/5. The full MCP module test run passed
101, skipped one, and had zero failures.
## Full-resolution masked crop/paste with pixel HiresFix (2026-11-05)

Repeated the real masked img2img request with NUI `inpaint_full_res=true` and
padding 32. Infotext reported `Inpaint area: Only masked` and `Masked area
padding: 32`. Full-resolution crop/paste completed on the 4-channel SD1.5 model,
then Anime 6B performed a 1.25x upscale plus four refinement steps at 0.2
denoise. Native-only MCP reached `img2img_hires step 1/1`, saved a 480x480 PNG
and completed in 115.9 seconds including app startup on the 4 GiB GPU. SHA-256:
`EA47DD04E4B90A052CEC97D896FE831DE5F84C4379E0A398C62D8247792F1F64`.
This verifies one square image/elliptical-mask/padding configuration. Official
nine-channel inpaint is documented above; other mask shapes/aspect ratios and
masked latent Hires remain open.

## SD2.1 Diffusers directory through native-only NUI (2026-11-05)

Packaged native-only NUI loaded `sd2-community/stable-diffusion-2-1-base` as a
Diffusers directory via MCP `set_model` and generated a 512x512 image with
Euler, 20 steps, CFG 7 and seed 4321. The run used the bundled JRE and CUDA
(`cuda:0`) without Python; the package's MCP hello exposed 26 tools. The output
was independently verified as a valid 512x512 PNG (526,583 bytes; SHA-256
`EA01DC1046885F407BD2F3335276612DDE4E6F2EF752393CE9DBFC5609475B6C`) and
visually inspected as a coherent rocky coast/lighthouse scene. The separate
2-step smoke also completed, though it was not a meaningful quality sample.
The longer run took about three minutes including startup; observed VRAM
samples were about 2.1 GiB, not a peak measurement.

This closes the SD2.1 Diffusers-directory load-and-generate gap for this model
and configuration. It does not verify SD2 single-file checkpoints, other SD2
variants, broad quality parity, or release readiness. The test model/output
remain in a temporary directory; no NUI source change was required. The test
harness receipt reported dimensions as 0x0, but direct PNG IHDR inspection
confirmed 512x512; treat the file header as authoritative.

### SD2.1 single-file checkpoint verified (2026-11-06)

The 5.21 GB public SD2.1 single-file safetensors checkpoint,
[`v2-1_512-ema-pruned.safetensors`](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors)
(SHA-256 `df955bdf6b682338ea9b55dfc0d8f3475aadf4836e204893d28b82355e0956d2`),
now loads and generates through Scala-Torch and packaged native-only NUI.
`SingleFileSd2Config` validates the 1,242-tensor OpenCLIP/LDM layout, keeps the
first 23 OpenCLIP layers (`resblocks.0..22`) while ignoring the unused
`resblocks.23.*` weights to match Diffusers, splits fused 3072x1024 QKV
projections, configures U-Net `stageHeads=Vector(5,10,20,20)` with context width
1024 and linear attention projections, and reports `family=sd2`. Setting
`SingleFileSd1Config.vaeConfig` to a 512x512 sample tile (`sampleHeight=512,
sampleWidth=512`) instead of `VaeConfig`'s 32x32 tiny-fixture default prevented
premature 32px tiled VAE decoding on single-file models.

Verification: `SingleFileSd2Suite` (10/10), `SingleFileSd2RealSuite` (component
weights, forward outputs, and 1-step seeded latent parity), `NativeSingleFileSd2Suite`
(1/1 backend 64x64 generation parity), and the 28-test SD1/SD2/ControlNet/backend
regression suite all passed with zero skips. After rebuilding
`models/workerDistribution` and overlaying the 14 worker JARs onto the
native-only NUI package, MCP `set_model(v2-1_512-ema-pruned)` and `txt2img`
(512x512, Euler, 20 steps, CFG 7, seed 4321) ran with the bundled Java 25 JRE,
CUDA `cuda:0`, restricted system `PATH`, and no Python environment or payload.
The output PNG
(`%LOCALAPPDATA%/Temp/sd2-native-model-20261105/mcp-home-single-file/outputs/txt2img/2026-11-06/00000-4321.png`,
IHDR 512x512, 526,589 bytes, SHA-256
`F4013FD01CE6F2429C49A0E1919BF67F9B9CFEC700A1562E37842EE9B54D1DBD`) was
visually inspected and is pixel-for-pixel identical (max channel difference 0,
mean difference 0 across all 512x512 pixels) to the Diffusers-directory image.
SD2 768-v (`v_prediction`) single-file checkpoints remain a separate open item.

### SDXL 1.0 single-file checkpoint verified (2026-11-06)

Full-component SDXL 1.0 base single-file checkpoints (`conditioner.embedders.0.transformer.*`,
`conditioner.embedders.1.model.*`, 3-stage `model.diffusion_model.*`, and
`first_stage_model.*`) now load and generate through Scala-Torch and packaged
native-only NUI. `SingleFileSdxlConfig` validates the dual-CLIP + 3-stage U-Net
layout while rejecting refiner-only and 4-stage checkpoints before allocation;
`SingleFileTensorView` maps CLIP-L, OpenCLIP ViT-bigG (including fused QKV
splitting and 2D transposition of `conditioner.embedders.1.model.text_projection`
into `text_projection.weight`), and U-Net `label_emb` -> `add_embedding`;
`SingleFileSd1Model` exposes `clip2: Option[ClipTextModel]` with `family=sdxl`;
and `NativeDiffusionBackend` wires `SdxlTextConditioning` with `padToken="!"`
for the second tokenizer when only a single `tokenizer` directory is present.

Verification: `SingleFileSdxlSuite` (7/7), `SingleFileSdxlRealSuite` (2/2 exact
tensor parity across CLIP-L, CLIP-G, U-Net, and VAE plus dual-CLIP conditioning
and VAE decode parity on the 6.94 GB `sd_xl_base_1.0.safetensors`),
`NativeSingleFileSdxlSuite` (1/1 backend 64x64 generation parity within `<= 1`
RGB level), and the 35-test regression suite all passed with zero skips. After
rebuilding `models/workerDistribution` and overlaying onto the native-only NUI
package, MCP `set_model(sd_xl_base_1.0)` and `txt2img` (512x512, DPM++ 2M
Karras, 16 steps, CFG 7.5, seed 42) ran with the bundled Java 25 JRE, CUDA
`cuda:0` with U-Net/VAE weight offload, restricted system `PATH`, and no Python
environment or payload. The output PNG
(`%LOCALAPPDATA%/Temp/sdxl-native-model-20261106/mcp-home-single-file/outputs/txt2img/2026-11-06/00000-42.png`,
IHDR 512x512, SHA-256
`B65DA5E3754EC962A4903E86FC7980B3B1834B6DF7EC46E1807FCCA043A511D1`) was
visually inspected and matches the Diffusers-directory baseline image (mean
pixel difference 0.072 / 255).
