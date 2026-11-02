## First standard SD1.5 native load and small-image smoke (2026-11-03)

Provisioned original FP32 Diffusers SD1.5 from public mirror, pinned revision
451f4fe16113bff5a5d2269ed5ad43b0592e9a14. Three safetensor files total~4.27GB;
all upstream SHA256 values verified. Details/hashes in
[SD1.5 acceptance work](docs/research/sd15-native-acceptance.md). Local ignored
model .superpowers/models/sd15, source-metadata.json contains upstream manifest.
Machine has64GiB host RAM/~42GiB available at start and ample disk space.

Discovered legacy CLIP position_ids buffer in real weights. PretrainedClip now
validates I64,shape(1,maxPositions),canonical0..N-1 and marks it consumed without
adding a parameter. Single/sharded fixtures plus bad shape/value tests: red before
fix,8 PretrainedClipSuite tests passed after fix, independent Candle parity intact.

Ran .superpowers/run-sd15.ps1 with64px,2steps through Java-Xmx12g, both CUDA
weight-offload settings,1GiB allocation budgets,134217728 attention work limit.
No Python runtime. load_model succeeded;DPM++2M Karras generation and PNG saved
in77.9136696s. Inspected image is coarse colored blobs, not quality acceptance.
Protocol/stderr and PNG under .superpowers/sd15-64-2; worker exited normally.
The checkpoint originally uses PNDM; harness uses explicit compatible beta noise
settings for implemented DPM sampling, not silently passing PNDM config through.

Next: optimize expensive host logicalArray copies seen in jcmd during weight
upload (Layout.index for every contiguous element), verify storage/stride
semantics, then512px standard-model generation with adequate steps and image
inspection. No standard-resolution/performance acceptance yet; full NUI feature
parity/SDXL/checkpoint formats/packaging remain open. NUI unchanged this turn.
scala-torch sbt retains tiny fixture properties/CUDA test flag; no inference or
download process intentionally left running. Only explicit scripts use full SD1.5.
## Attention work limit for larger spatial inputs (2026-11-03)

PretrainedUNet.load appends maxScoreElements=16777216 and propagates it to every
self/cross attention block. NativeDiffusionBackend accepts maxAttentionScoreElements
(default unchanged,0..Int.MaxValue) and applies it to U-Net, VAE and replacement
VAEs. NativeWorkerMain reads SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS; status
runtime.max_attention_score_elements reports it. This explicit work guard remains
separate from sliced CUDA scratch/session byte caps. CPU still uses dense scores.

Actual GPU test:4096 tokens,8heads,width1 has134217728 scores, rejected by default
and accepted with explicit limit under20MiB session budget; constant output exact,
no allocation leaks. Width1 isolates score geometry, not normal-model head width,
full image generation or practical throughput. Passed4 sliced-attention tests.
PretrainedUNetSuite passed8 with public fixture, including propagation to all
loaded blocks and original independent Candle reference outputs. Worker process
suite passed5 (including limit1 rejection before progress, four successful CPU/
CUDA/offload generation modes); backend suite passed2. Final worker/status rerun
passed7 with no skips. Initial loader run had skipped public tests because server
lacked properties; restarted with explicit CUDA flag and all3 model properties
before collecting passing evidence.

Only tiny-sd-pipe was found in this workspace model folder; the historical NUI
D:/stable-diffusion-webui-reForge model location did not yield local model files.
Next locate/provision a standard Diffusers checkpoint, inspect host memory needs,
and run real-model acceptance; then continue missing NUI features/formats. Goal
remains incomplete. scala-torch sbt now has CUDA tests and model properties; NUI
sbt stopped. No NUI edits in this checkpoint; its dedicated branch is preserved.
## CUDA offload connected to pipeline, worker and NUI (2026-11-03)

UNetExecution.Cuda / VaeExecution.Cuda now expose offloadWeights=false and pass
it to runtime snapshots. CUDA environment accepts independent literal true/false
SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS and SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS.
CPU ignores CUDA-specific options. Status exposes runtime.unet_offload_weights
and vae_offload_weights; device/host-load semantics and defaults remain unchanged.
No silent fallback, FP16 or whole-process GPU memory accounting is implied.

Verification on real GPU:
- models/testOnly *CudaImagePipelineSuite:6 passed. Environment parsing rejects
  invalid values; offloaded Euler/DPM CFG output exactly equals resident at2MiB
  per-session budgets for10x6 tiny images.
- NativeWorkerProcessSuite/NativeWorkerSuite/NativeDiffusionBackendSuite:8 passed.
  Four separate Java process modes include offloaded U-Net+VAE at2MiB, status,
  progress, preview and PNG with no Python launch.
- NUI engine/testOnly *NativeScalaEngineSuite:1 passed2.838s, both offload options,
 16MiB per session,64x64 tiny output, keyword mapping and VAE replacement/restore.
  Initial2MiB run failed VAE decode and wrote recovery latents. Test cleanup hid
  that with DirectoryNotEmptyException; recursive cleanup now preserves primary
  errors. Repeating2MiB confirmed vae_decode_failed;16MiB then passed. No claim
  of standard-model throughput or total GPU usage at these limits.

NUI dedicated branch commit826356b updates test/docs. Scala-torch sbt stopped;
NUI sbt currently has both CUDA/offload options and16MiB limits, plus native
integration classpath/model variables. Restart servers when changing env.
Next: standard model memory/performance, attention work limits and remaining
Python-free feature parity. Full objective remains incomplete.
## Per-operation CUDA weight offload (2026-11-03)

CudaUNet.upload and CudaVaeDecoder.upload accept offloadWeights=false by default.
True retains immutable host Tensor snapshots and uploads only weights needed by
each operation, scoped with Using.Manager. Activations remain on CUDA. Linear
projections upload a transposed host view directly, avoiding two GPU weight copies.
Shape checks use snapshot metadata without uploading weights. Close drops host
snapshot references; failure/cancellation releases temporary GPU weights/output.
This trades transfer throughput for VRAM; no FP16, prefetch, pinned host memory,
or activation offload. Parameter.load after construction cannot change snapshots.

New CudaOffloadSuite exercises complete public tiny VAE/U-Net at a session budget
of largest parameter bytes +1MiB, less than total model weight bytes. Resident VAE
upload fails at that budget while offloaded decoding matches resident exactly.
U-Net repeats exact outputs after a source Parameter.load mutation. VAE cancellation
then reuse, insufficient inference memory and closed-runtime rejection check cleanup.

Verification: models/testOnly *Cuda* passed22, zero failures/errors/skips on the
real GPU after final source edits. Existing resident pipeline tests also passed.

Pipeline/worker still use resident defaults: next wire this option through
UNetExecution/VaeExecution, environment and status, and verify native NUI generation.
Then test standard models and resolve remaining attention work limits/performance.
Normal model acceptance and full Python-free NUI parity remain incomplete.
## Bounded CUDA diffusion attention workspace (2026-11-03)

Noncausal scaledDotProductAttention now slices query rows and reuses Double
score scratch, capped by maxWorkspaceBytes (default16MiB) and the session budget
after reserving the output. At least one complete key row must fit. PTX indexes
scratch by local row and inputs/output by global row, preserving arithmetic order.
Causal/cache paths retain their existing full-workspace budget behavior.

maxScoreElements remains the total-work guard (default16777216); this change does
not remove normal-resolution work limits, reduce FLOPs, implement FlashAttention,
or solve full FP32 model residency. No cancellation callback runs between slices.

Verification after LF normalization: sbt "ffm/testOnly *; models/testOnly *Cuda*"
passed52 FFM tests and20 CUDA model tests, zero failures/errors/skips on the real
GPU with pinned public tiny CLIP/U-Net/VAE fixtures. New tests compare raw output
bits across slice boundaries, batches, GQA, unequal widths and NaN/Infinity;
check allocation cleanup and empty shapes; and run an otherwise8MiB score matrix
with64KiB scratch under a139264-byte session budget. Existing causal/cache and
backward tests pass. CPU versus GPU image pipeline max error remains7.45e-7.

Next: real-model weight residency/offload and attention work-limit configuration,
then normal-model acceptance and remaining NUI feature parity. The full goal is
still open. Current scala-torch sbt server has CUDA tests enabled plus the three
pinned model directory properties; NUI server is stopped. Older server-state
notes below are historical. No NUI code changed in this checkpoint; its dedicated
feat/scala-torch-native-engine branch is preserved.
## CUDA U-Net denoising connected to pipeline, worker and desktop (2026-11-03)

UNetExecution (default Cpu) provides a predictor scoped to the entire sampling
loop. Cuda snapshots U-Net weights and positive/negative contexts once, performs
resident U-Net/CFG each step, downloads guided predictions for existing CPU
Euler/DPM schedulers, then releases the session before VAE decode. CLIP and small
sinusoidal timestep constants remain CPU. This is hybrid execution, not a fully
GPU-resident scheduler or FP16/offload. No layer activation downloads.

NativeWorkerMain accepts SDFX_SCALATORCH_UNET_DEVICE=cpu|cuda|cuda:<index> and
SDFX_SCALATORCH_UNET_MAX_BYTES (default1GiB), independently of VAE placement.
Model load remains fp32/cpu host storage. set_vae retains both execution settings.
Hello.device/status.device.active reflect configured U-Net placement; hello.cuda
reflects either CUDA component. Status lists unet/vae/text_encoder/scheduler devices
and reports hybrid when applicable. CUDA availability is checked at inference;
no silent fallback. Cancellation/failure tears down session-scoped weights.

Verification:
- models/testOnly *CudaImagePipelineSuite *PretrainedVaeSuite:19 passed, no skips.
  Euler/DPM3steps with CFG1/7.5 and CUDA U-Net+VAE match CPU images within max7.45e-7.
  Cancel after a sampling step then regenerate succeeds; low U-Net memory budget
  throws outside VaeDecodeFailure. Existing exact-latent VAE recovery still passes.
- models/testOnly *NativeWorkerProcessSuite *NativeWorkerSuite
  *NativeDiffusionBackendSuite:7 passed. Separate Java processes cover CPU,
  CUDA VAE-only, and CUDA U-Net+VAE JSON Lines progress/preview/PNG plus status.
- NUI engine/testOnly *NativeScalaEngineSuite passed1 in2.636s with both CUDA opts:
  real EngineClient generation, VAE replacement/restore, typed preview/PNG and status.
- Actual JavaFX ui/run --autogen --exit-after35 also completed. Inspected screenshot:
  .superpowers/nui-ui-8389b9ccd0c940b787ecad1783f6d50b/native-ui-cuda-unet.png
  shows Ready cuda:0, Done1image and gallery. Saved outputs/txt2img/2026-11-03/
  00001-42.png (64x64,DPM++2M Karras3,CFG7,seed42), SHA256
  91c346a97e597f19af3d5d90d6e8a98fca9947021ebfe7886ddc1837b498900e
  equals prior CPU PNG after quantization. Tiny model output is noise-like; no
  normal-model quality claim. VRAM estimator remains unsupported. ui/run reported
  success47s; sbt thin-client printed an InterruptedException during shutdown after
  success, not an inference error.

NUI dedicated branch has updated tests/docs in a separate commit. NUI sbt server
currently inherits all native launch vars plus UNET_DEVICE=cuda and VAE_DEVICE=cuda;
scala-torch server stopped. Artifacts/classes updated by tests. Restart server
with CUDA tests and clip/unet/vae properties for further scala-torch testing.

Next: improve real-model memory/performance (dense quadratic attention, FP32 storage,
FP16/offload), continue full feature parity and standard checkpoint support. The
original Python-free NUI goal is still incomplete despite this tiny GPU milestone.
## Complete resident CUDA U-Net forward (2026-11-03)

CudaUNet.upload(session, ConditionalUNet2d, cancelled) snapshots the full model.
Forward composes learned timestep MLP, input convolution, all down layers,
residual/attention middle, LIFO skips and up layers, output norm/activation/conv.
Skips hold independent retained handles and release immediately after concatenation;
all exceptional/cancel paths are scoped. Only small sinusoidal timestep constants
are computed on CPU and uploaded; learned layers, context projections and image
activations stay on GPU. No intermediate activation readbacks.

ConditionalUNet2d.timestepFeatures shares input/context/time validation and CPU
sinusoidal constants between CPU and GPU paths. Both common padding-one downsample
and explicit right/bottom zero-pad downsample are supported. The latter composes
resident fill/cat operations, with extra copies; it is not an optimized padding
kernel. Odd-size upsampling targets the next skip's exact spatial dimensions.

Actual GPU verification:
- Three independent Candle U-Net cases:3x5 t7.5 maxAbsError1.3113022e-6,
  4x4 t0 1.0132790e-6,4x6 t19 2.2649765e-6. Initial test mistakenly used t7.5
  for all files; corrected fixture inputs to the existing reference specification.
- Batch2 shared/per-sample timesteps and padding-zero/one paths match CPU within
  max3.8743019e-7. Outputs survive model/input closure. Invalid conditioning,
  cancellation then reuse, and limited-workspace failure preserve allocation counts.
- models/testOnly *CudaUNetSuite:3 passed after fixture correction.
- PretrainedUNetSuite:7 passed in initial combined run (pinned weights/reference
  validation); that combined run still had the fixture-input test failure above.
- Final models/testOnly *Cuda* *ConditionalUNetSuite *DiffusionTimeSuite:26 passed,
  no skips, including existing CUDA VAE/pipeline regressions and CPU gradient tests.

No worker/pipeline U-Net selector is connected yet. Next keep uploaded U-Net weights
and encoded contexts alive across the full denoising loop, integrate explicit CUDA
selection, then verify full prompt-to-image through NUI. Sampling and CLIP remain
CPU today; dense score budgets, normal-model memory, FP16/offload, SDXL and other
existing NUI features remain outstanding. Current scala-torch sbt server retains
CUDA test flag and all three pinned model directory properties. NUI unchanged.
## Resident U-Net spatial Transformer and GEGLU (2026-11-03)

CudaSession.geglu now executes final-axis split Gaussian-error GELU gating on GPU.
It preserves the CPU's separate GELU Float32 rounding then Float32 multiplication.
PTX uses an independently implemented incomplete-gamma series/continued fraction
and existing Double exp, not tanh GELU. Mathematical sources and tested limits are
recorded in docs/research/cuda-geglu.md. No runtime Python or new dependencies.

CudaDiffusionRuntime adds layerNorm, feedForward, transformer and spatialTransformer.
These compose resident Linear/GEGLU, LayerNorm, self/cross attention, GroupNorm,
1x1 projections, token/image transposes and residuals. Scoped intermediate cleanup
and snapshot parameter ownership follow the existing VAE executor.

Validation with actual GTX1650 SUPER and pinned tiny U-Net:
- autograd/testOnly *CudaGegluSuite:2 passed;30,721 gate grid points plus IEEE
  extremes, two Float32 ULP tolerance, layouts/empty/lifetime/budget checks.
- ffm/testOnly *:49 passed after the shared PTX module change.
- models/testOnly *Cuda*:15 passed (all GPU model tests, no skips).
- Loaded U-Net feed-forward exactly matched CPU in its test; token and image
  Transformers maxAbsError1.1920928955e-7. Repeat inference frees intermediates;
  outputs survive model/input closure; bad context rejects without allocation.
- Existing CUDA VAE, pipeline, cancellation and recovery regressions also passed.

Full U-Net forward composition and sampling-loop integration are next. Dense
attention still allocates quadratic Double score scratch with explicit limits;
normal model memory/performance, FP16/offload, CLIP GPU and NUI feature parity
remain outstanding. No end-to-end GPU denoising claim in this checkpoint.
NUI server stopped; scala-torch sbt server inherits CUDA tests plus clip/unet/vae
fixture directory properties. NUI dedicated branch remains unchanged this turn.
## CUDA VAE decoding connected through pipeline and NUI worker (2026-11-03)

EulerImagePipeline now takes explicit VaeExecution (default Cpu). Cuda opens a
thread-local session per decode, uploads decoder weights, runs resident inference,
downloads raw RGB, and closes all resources. CLIP/UNet/sampling remain CPU. The
pipeline still applies inverse VAE scaling exactly once and wraps non-cancellation
failures as VaeDecodeFailure with the exact decoder input. No silent CPU fallback.

NativeWorkerMain reads SDFX_SCALATORCH_VAE_DEVICE=cpu|cuda|cuda:<index> and optional
SDFX_SCALATORCH_VAE_MAX_BYTES (default1073741824). These augment the existing native
launch config. Model loading remains fp32/cpu; set_vae preserves decoder placement.
NativeSystemStatus.runtime.vae_device reports placement and gpu_status states that
CUDA decode is configured, without claiming live GPU memory telemetry.

CudaDiffusionRuntime checks cancellation during weight upload and between scopes
and decoder operations. Running kernels remain synchronous/non-preemptible. Close
paths ignore cancellation so cleanup succeeds. Test cancels inside inference,
checks exact allocation accounting, then reuses the same weights successfully.

Validation on actual GTX1650 SUPER4GiB:
- models/testOnly *CudaImagePipelineSuite *CudaVaeDecoderSuite
  *CudaDiffusionAttentionSuite *CudaDiffusionResnetSuite *PretrainedVaeSuite
  *NativeDiffusionBackendSuite *NativeWorkerSuite:31 passed, no skips.
- models/testOnly *NativeWorkerProcessSuite:2 passed (separate Java processes,
  CPU and CUDA VAE; JSON Lines status, PNG and preview). Python/test JVM env removed.
- Euler CPU-vs-CUDA-decode maxAbsError7.45e-8; DPM5.96e-8; recovery input, upload
  cancellation, runtime cancellation, settings parsing and CPU regressions pass.
- NUI feat/scala-torch-native-engine engine/testOnly *NativeScalaEngineSuite:
  CUDA VAE opt-in passed1 in14.039s; real EngineClient serialization and PNG workflow,
  VAE replacement/restore plus reported device verified. NUI test/docs committed
  separately. No new JavaFX screenshot in this checkpoint.

NUI server currently inherits SDFX_SCALATORCH_VAE_DEVICE=cuda and test cp/model env.
scala-torch server was shut down before NUI test. Restart with CUDA tests plus
JAVA_TOOL_OPTIONS for scalatorch.{clip,unet,vae}.dir when further tests require it.
Native worker artifacts were rebuilt and .superpowers/worker-classpath.txt points
at those updated jars. Remaining work includes GPU U-Net/CLIP composition, FP16,
offload and full existing NUI features. CUDA VAE alone is not goal completion.
## Complete resident CUDA VAE decoder (2026-11-03)

CudaVaeDecoder.upload(session, VaeDecoder) now executes the complete loaded FP32
decoder: optional post-quant convolution, input convolution, residual/attention
middle, all up-stage residuals and nearest/convolution upsampling, output norm,
SiLU and RGB convolution. Input is raw unscaled latents; output is raw RGB without
clamp. Upload once, inference keeps activations resident, download final output.
Decoder and returned output handles have independent lifetimes. No NUI/pipeline
GPU selector is connected yet; EulerImagePipeline continues using CPU decode.

CudaDiffusionRuntime (package-private) shares snapshot weights, scoped convolution,
GroupNorm, Linear including local LoRA, residuals, projected self/cross attention
and VAE image attention. CudaDiffusionResnet delegates to it without API changes.
Attention supports unequal context tokens/widths and preflights score layout;
quadratic Double score scratch remains, not memory-efficient attention. VaeDecoder
preflight is now shared by CPU and GPU, preserving complete feature-budget checks.

Actual GPU tests with existing CUDA flag and scalatorch.vae.dir:
- models/testOnly *CudaVaeDecoderSuite *CudaDiffusionAttentionSuite
  *CudaDiffusionResnetSuite *PretrainedVaeSuite:20 passed,2 skipped (CPU full-pipeline
  tests need additional properties). This ran before the last two test expansions.
- Final models/testOnly *CudaDiffusionAttentionSuite *CudaVaeDecoderSuite:6 passed.
  Includes added bad-shape/allocation cleanup and optional post-quant/mid-attention.
- Full decoder matches independent Candle binary references: batch1 3x5 latent
  maxAbsError2.0023435e-6, batch1 4x4 3.1143427e-6, batch2 3x5 2.5033951e-6.
- Loaded VAE middle residual -> attention -> residual, projected self/cross,
  repeat calls, model/output ownership, upload/inference budget cleanup verified.
Existing CPU VAE six Candle cases and other loader tests also passed as above.
No changes to FFM kernels in this checkpoint; previous FFM49 remains its last run.

Next: connect optional resident decoding to pipeline/worker with explicit lifecycle
and error/cancel handling, then continue complete U-Net execution and memory modes.
This does not establish normal-model FP16/offload or complete Python-free NUI parity.
## Uploaded CUDA diffusion residual blocks (2026-11-03)

CudaDiffusionResnet.upload(session, DiffusionResnet2d) snapshots existing model
parameters once. Inference composes resident GroupNorm, SiLU, biased convolution,
optional timestep projection/addition, shortcut and output scaling. Parameter
and temporary handles are explicit; intermediate activations close after use;
Using.Manager cleans exceptional paths. Returned tensors survive block closure.
CPU parameter mutation after upload does not affect the snapshot. Local Linear
low-rank adapters on time projections also execute resident (this does NOT add
NUI LoRA loading/application or claim whole diffusion LoRA support).

CudaSession.addBroadcast implements trailing-axis addition with zero strides,
without expanding inputs. Integer geometry metadata is budgeted and cleaned.
Equal shapes retain existing binary fast path; empty shapes skip metadata/kernels.

Validation: SCALATORCH_CUDA_TESTS=1 and JAVA_TOOL_OPTIONS with
-Dscalatorch.vae.dir=C:/Users/Mizushima/dev/scala-torch/.superpowers/models/tiny-sd-pipe/vae
were set before restarting sbt. models/testOnly *CudaDiffusionResnetSuite
*DiffusionResnetSuite passed6, including actual safetensors VAE decoder mid first
and second residuals chained on GPU (attention intentionally not part of that
residual-only test). Conditioned/unconditioned/channel-changing cases, time
projection LoRA, repeated calls, snapshot isolation, closed-block rejection and
upload/execution budget cleanup passed. CPU parity tolerance3e-6 (loaded VAE5e-6).
ffm/testOnly * passed all49 including broadcast bias/time/scalar/empty layouts
and scratch cleanup when output allocation fails. No Python runtime involved.

The full loaded VAE/UNet graph and NUI worker still use the CPU pipeline. Next
connect projected attention and remaining graph operations to a reusable resident
model executor, then verify complete inference; FP16/offload remain outstanding.
Current sbt server inherits the CUDA flag and scalatorch.vae.dir above.
## Resident CUDA concatenation and skip connection (2026-11-03)

CudaSession.cat joins packed tensors along any axis with one output allocation,
no scratch and one raw-bit copy kernel per nonempty input. Shape validation is
shared with CPU autograd via ConcatLayout; repeated inputs and empty tensors are
supported, including empty shapes with Long-sized dimensions. Input handles are
not retained; output survives their closure. Invalid shape, sum overflow and
budget failures leave inputs and allocation accounting intact.

Actual GPU autograd/testOnly *ConcatSuite passed5 (new CUDA3, existing CPU2).
New tests directly compare CPU Variable.cat for all axes of ranks1/2/4, unequal
axis widths, empty/repeated inputs, strided uploads and IEEE payloads. A composed
resize -> skip concat -> GroupNorm -> convolution chain matches CPU bit-for-bit,
with only the final output downloaded. ffm/testOnly * passed all48 regressions.
This does not yet connect the loaded diffusion graph to GPU inference; model
composition, missing operations and memory planning remain active work.
## Resident CUDA nearest resize (2026-11-03)

CudaSession.resizeNearest2d implements NCHW Floor and Exact nearest resize,
sharing CPU Resize2d output geometry validation. PTX uses 64-bit integer ratios
and raw 32-bit value copies, preserving signed zero, infinities and NaN payloads.
No intermediate host activation transfers or device scratch are required.

Tests cover noninteger up/downsampling, both modes, batches/channels, transposed
CPU uploads, large coprime widths (65537 -> 65539, overflowing 32-bit products),
empty batches, bad geometry, released handles and allocation budget cleanup.
Resident resize -> convolution agrees bit-for-bit with CPU. Actual GPU suite
ffm/testOnly * passed all48; core/testOnly *Resize* passed2. No end-to-end GPU
pipeline claim: graph composition, other missing operations and memory strategy
remain necessary for normal diffusion models.
## Resident CUDA noncausal attention (2026-11-03)

CudaSession.scaledDotProductAttention supports noncausal self/cross attention,
grouped query heads, unequal query/key token counts and value widths. Shared PTX
attention now receives an explicit causal flag; existing causal and KV-cache
callers retain causal behavior. Activations and scores stay on device.

SCALATORCH_CUDA_TESTS=1 sbt "ffm/testOnly *": all45 passed on the actual GPU.
Two new tests compare CPU full-key references, prove future keys participate,
check future NaN propagation, empty batches, score limits and allocation cleanup.
Existing causal/cache/backward tests also pass with the changed kernel signature.
Scores still require quadratic Double workspace; this is not memory-efficient
attention or end-to-end GPU diffusion. Model composition remains outstanding.
## Resident CUDA GroupNorm/LayerNorm (2026-11-03)

CudaSession.groupNorm supports (N,C,...) groups with independently optional
resident per-channel weight and bias. PTX one thread/group computes Double mean
and biased variance in CPU reference order, casts normalized result to Float32,
then separate FP32 multiply/add (matches nn.GroupNorm without FMA contraction).
Core GroupNormOps.width(Shape,...) shared for identical geometry validation.
CudaSession.layerNorm uses retained reshape views and GroupNorm(groups=1), matching
Variable.layerNorm final-axis semantics; temporary handles closed on all paths.

Hardware tests passed5 new tests: groups1/2/4, rank2/rank4, empty batch, all4 affine
combinations, noncontiguous upload, extreme/NaN/Inf values, lifetime/budget errors,
LayerNorm shape/ownership, resident convolution -> GroupNorm -> SiLU chain. Exact
CPU equality for normalization; SiLU chain tolerance2 Float32 ULPs as existing GPU
SiLU contract. No intermediate downloads in composed GPU chain, final output only.
Core GroupNormSuite2 passed. ffm/test ran38 due sbt incremental selection, so then
explicit ffm/testOnly * ran ALL43 passed with SCALATORCH_CUDA_TESTS=1. Use testOnly *
for whole-suite claims, ordinary test may select only impacted suites. This also
clarifies prior conv wave's ffm/test36 was its selected set, not every FFM test.

Still not full GPU image generation: EulerImagePipeline and nn model objects
remain CPU. Next noncausal attention, concat/nearest resize/spatial padding and
GEGLU, then resident U-Net/VAE model composition and parameter lifetime/offload.
Normal-model/GPU performance and fp16 remain unverified. No Python or nvcc used.
Current sbt scala-torch with CUDA_TESTS=1, no running tests. NUI unchanged.
## Resident CUDA convolution for diffusion (2026-11-03)

Added CudaSession.conv2d on resident CudaTensor inputs/output, NCHW/OIHW grouped
cross-correlation. PTX one thread/output, Double accumulation in identical product
order to CPU reference (no tiling/performance claim). Parameters staged on GPU and
charged to allocation budget; all scratch freed on success/failure. Geometry checks
shared with CPU through private[scalatorch] Conv2dOps.Geometry/geometry(Shape,...).
GPU module now loads conv2d entry alongside existing transformer kernels.

Verified actual hardware again: GTX1650 SUPER, 4096MiB, driver610.88. Enabled
SCALATORCH_CUDA_TESTS=1 before starting scala-torch sbt. core Conv2dSuite 3 passed;
ffm/test ALL 36 passed, no skipped convolution tests (XML tests3 failures0 skipped0).
New tests cover 6 configs (padding/stride/dilation/groups including depthwise),
bit-for-bit CPU reference agreement, noncontiguous upload, empty batches, NaN/Inf
and padding, released handle rejection, allocation failure/no-leak accounting.
No Python used. CUDA SDK/nvcc absent on PATH; committed PTX loaded by CUDA driver.

This does NOT yet run image pipeline on GPU. EulerImagePipeline/Variable/nn Conv2d
remain CPU-only; CudaSession resident API is separate from Tensor[Float]. Next add
GPU diffusion primitives (GroupNorm/LayerNorm, noncausal attention, concat/resize,
GEGLU etc) and backend-generic/explicit resident U-Net/VAE composition; integrate
memory lifetime/offload/fp16 for full-sized models. Current direct Double conv is
correctness foundation and may be slow on GTX1650; no acceleration claim.
NUI unchanged this wave. Current scala-torch sbt server has CUDA_TESTS=1; no tests
running. Full user goal remains active; earlier tiny CPU UI success is retained.
## Native keyword mapping commands and atomic edits (2026-11-03)

Added NativeKeywordMappings handler for model_keyword check/save/delete, routed by
NativeDiffusionBackend.supported/execute. Configured Diffusers directory uses full
native model hash (cached while loaded, computed otherwise); checkpoint/LoRA files
use legacy hash. match_name alias preserved, NUI response choices/source/filename/
hash/keywords/default_catalog_available/deleted implemented. Mapping metadata for
LoRA works but actual LoRA inference remains unsupported. No catalog downloader or
bundled upstream catalog yet; only local default and custom files are searched.

KeywordMappingEditor preserves unrelated raw records/comments/line endings,
replaces duplicate matching rows, CSV-quotes new content, rejects malformed CSV.
Uses same Windows one-byte .lock as Python NUI, cancellation-aware 30sec lock wait,
fsynced staging and original-byte backup, final content comparison and atomic move.
Publication refusal cleans staging/backup without altering original; lock file
persists. Late cancellation after successful write is not falsely reported as a
rolled-back operation. Non-Windows Python flock interoperability is not claimed.

12 unit tests passed (editor3, handler2, reader7), including lock cancellation,
malformed preservation, backups and rejected publication. Real NUI EngineClient
integration passed 17sec: mapping check/save -> generation resolves a red cat ->
delete/check, along with prior loader/VAE/previews. UI compile also passed. This
is client/worker transport evidence, NOT a new actual button-click test.

NUI dedicated branch feat/scala-torch-native-engine commit 86e45de includes extended
NativeScalaEngineSuite and correct missing-catalog label for checkpoint vs LoRA.
NUI worktree clean; no push. scala-torch changes committed separately. Current sbt
server NUI started with SCALATORCH_WORKER_CLASSPATH_FILE and SCALATORCH_TINY_MODEL;
scala-torch server stopped. No active background tests remain.

NEXT: normal-model/GPU diffusion and remaining inference options remain substantial;
keyword catalog install/default resources and feature advertisement also pending.
Original goal still active; tiny UI generation is not full Python-removal parity.
## First successful actual NUI desktop generation (2026-11-03)

NativeKeywordHook now accepts model_keyword in generation, reads mapping CSV,
uses ModelKeyword planner and passes actual per-image prompts into inference.
Returns effective_prompt/model_keyword_source_prompt. NativeDiffusionBackend
commits keywordState only after all output moves and staging deletion succeed.
Explicit unsupported LoRA/embedding selections and extra-network tags still fail.
Unknown active hooks fail, explicitly disabled hooks do not execute.

Diffusers identity: full 64-character modelSummary hash, directory basename,
_mapping_dir/model-keyword.txt + custom-mappings.txt. This intentionally extends
legacy CSV keys to full native directory hashes; do not hash a directory as a
legacy checkpoint file or truncate to 8 chars. Short-hash upstream catalogs cannot
automatically identify converted directories. Mapping UI requests not yet wired.

Disabled ControlNet slots (explicit boolean enabled=false) now accepted; active,
missing enable flag and malformed slots rejected. Native hook/default tests 4
passed; real tiny-model backend tests 2 passed including pixel-for-pixel equality
of mapped red/blue prompts with direct generation, metadata and iteration rollback
on failed generation/denied publication. Asset properties explicitly enabled.

ACTUAL UI SUCCESS: screenshot native-ui-keyword-fixed.png in isolated home
.superpowers/nui-ui-8389b9ccd0c940b787ecad1783f6d50b shows Done 1 image and gallery.
PNG outputs/txt2img/2026-11-03/00000-42.png visually inspected: 64x64 colored noise,
expected tiny-model behavior, NOT cat-quality/full-model acceptance. SHA256
91c346a97e597f19af3d5d90d6e8a98fca9947021ebfe7886ddc1837b498900e.
DPM++2M Karras, 3 steps, seed42 CFG7, CPU. Java UI + NativeWorkerMain observed.
VRAM estimator banner remains unsupported. UI run exited0 after45sec; thin client
printed InterruptedException on disconnect AFTER successful screenshot, not a
worker/generation failure. Current sbt server NUI, scala-torch server stopped.
NUI worktree clean dedicated branch; no NUI edits in this wave.

NEXT: native mapping check/save/delete UI workflow; normal models/GPU inference,
A1111 weights, LoRA/embeddings/img2img/batching/extensions and packaging remain.
See docs/research/nui-native-worker-boundary.md for current evidence and limitations.
Goal remains active, far from entire original Python-removal feature coverage.
## Keyword mapping reader and filename matching (2026-11-03)

Added KeywordMappings: CSV skipinitialspace/quoted multiline fields, normalized
hash keys, default/custom source identity, filename SequenceMatcher similarity
and later-row tie-breaking. Custom rows do NOT always win: a closer default
filename wins over an unrelated custom filename. Matching uses Unicode codepoints
and Python autojunk behavior above 199 chars. Fresh reads observe edits/deletion;
no caching and no file creation. Before/after identity/mtime/size checks reject
observed source replacement, but there is no lock across both mapping files.
Legacy file hash reads 64KiB at offset 1MiB and takes first 8 SHA256 hex chars;
directories deliberately rejected by this API (native Diffusers identity pending).

Validation: ModelKeywordSuite 8 + KeywordMappingsSuite 7 = 15 passed. Includes
160 static difflib oracles + 10 static CSV oracles, existing 144 planner cases,
Unicode/non-BMP and repeated-name matching, lookup refresh, UTF8 errors, resource
limits, cancellation and deterministic concurrent-edit rejection. Python was only
used to generate static fixtures. Regenerator, docs and PSF notice bundled.

NEXT: connect native worker model_keyword hook to planner and mappings, decide
and document full-SHA identity for Diffusers directories (modelSummary hash is
available), preserve per-image effective_prompt/model_keyword_source_prompt,
commit iteration state only after successful publication. Current worker STILL
rejects hooks; no desktop success yet. Native selected LoRA/textual inversion
inference and mapping save/delete/check endpoints are not implemented. Disabled
ControlNet entries in desktop requests also need handling without dropping active
units. NUI tree has not been changed this wave. Current scala-torch sbt server
has no tiny-model properties; all 15 tests in this wave are asset-independent.
## Model Keyword planner primitives (2026-11-03)

Added ModelKeyword pure prompt planner with joined/explicit/random/iterative
selection, all 4 placements, ordering, optional embedding and LoRA tag rendering,
preview-first random consumption, and immutable BigInt iteration counters. Caller
receives nextState and must commit only when its transaction succeeds. This is
selection/rendering, NOT actual LoRA/embedding loading or worker hook integration.
KeywordRandom implements integer-seeded CPython choice behavior (MT19937); it is
NOT Torch latent noise. Upstream MT copyright/license bundled in model resources.

Eight Scala tests passed: includes 144 static cases generated from existing NUI
plan_with_preview and IterationState.reserve, signed seed limits, and SHA256 of
2,000 bounded random draws crossing MT refresh boundaries. Python used only to
produce committed fixtures; test execution has no Python dependency. Regenerator
and provenance in models/src/test/resources/model-keyword/. Cases revealed/fixed
weight string formatting (1e-05, 0.0001, -0.0). No broad floating-format claim beyond
verified values; Java/Python shortest-double representations can differ at extremes.

NEXT: mapping CSV reading/lookup with filename similarity and custom precedence,
checkpoint identity for Diffusers directories (existing Python legacy_hash opens a
file, cannot hash directories); then native worker model_keyword hook and mapping
commands. Current worker still rejects the hook, so no new desktop success claim.
Actual LoRA and textual inversion remain future dependencies, not silently ignored.
No NUI files modified; dedicated branch stays clean. Current scala-torch sbt server
active without tiny-model props (planner tests do not need models).
## Generation metadata and desktop hook boundary (2026-11-03)

Confirmed prior commit 250fc2f. New NativeGenerationMetadata accepts the NUI string
infotext map (retained by NUI for PNG metadata), validates all values are strings,
and interprets prediction_type epsilon/v_prediction/sample per request. Both Euler
and DPM++2M receive the override; checkpoint config remains unchanged. Explicit rng
selection is rejected because current Java Gaussian noise is not PyTorch/NV noise.
Default native seed noise still has that compatibility gap.

Metadata suite 2 tests and defaults suite 1 passed. Backend suite 2 passed WITH tiny
model properties: both samplers change output for v_prediction, explicit epsilon
plus metadata matches default exactly, later default generation matches original.
An earlier run skipped both backend tests due to server env; that was corrected by
restarting scala-torch sbt with JAVA_TOOL_OPTIONS and rerunning (2 actual passes).

Actual UI screenshots in the same isolated home:
- native-ui-envelope-fixed.png: previous fixes worked; rejected nonempty extra.
- native-ui-metadata-fixed.png: extra acceptance works; now rejects hooks.
The UI defaults enable model_keyword; hookFields plus _base_prompt/_mapping_dir
are supplied by Generator. NEXT implement model_keyword behavior, not discard it.
Python reference base.py generation_keyword_plan/_model_keyword_plan and
model_keyword.py contain mapping/choice/placement semantics. Note legacy_hash
currently expects a checkpoint file, whereas native uses Diffusers directories;
need deliberate directory identity semantics, no silent bypass. UI also includes
disabled ControlNet units (not accepted yet). vram_estimator banner remains.
No desktop generation success yet. NUI tree remains clean on its dedicated branch.

NUI sbt 2.0.12 server deadlocked during shutdown in
VirtualTerminal.cancelRequests -> ArrayBlockingQueue.put (jcmd confirmed twice).
Stopped only owned server PIDs 31872 and 35532, then restarted and reran UI.
Current NUI sbt server idle after successful screenshot run; scala server stopped.
Old thin-client sessions 2580/53254 may remain disconnected; no live work there.
## Desktop envelope and VAE recovery (2026-11-03)

Native backend now accepts vae_recovery_dir and uses it on VAE decode failure.
Pipeline wraps nonfatal VAE exceptions/nonfinite output as VaeDecodeFailure carrying
exact Float32 decoder input AFTER scaling. NativeVaeRecovery writes tensor to a
unique create-new safetensors under vae-recovery; worker returns vae_decode_failed
with artifact path and cleans normal staging. No Broken-preview substitution yet,
no RGBA packed PNG, and native metadata is not yet NUI vae_tools provenance schema:
this is exact tensor preservation, NOT full existing recovery feature parity.
Test forces decoder budget failure and verifies actual decoder input values;
writer roundtrip test verifies safetensors readback. Default full-image reference
still matches. 17 tests passed before later envelope edits; final VAE suite 14
passed, envelope/recovery/backend 4 passed, then subseed backend 2 passed.

Checkpoint/vae selectors are checked against loaded model rather than requiring
null. Empty manual_loras/hires_loras/img2img_hires_loras arrays accepted along with
null; nonempty remains rejected. Non-default subseed accepted only while variation
strength remains zero; returned/incremented as metadata with overflow checks.
Generator normalizes subseed from -1 even when variations are inactive.

Desktop rerun screenshot native-ui-recovery-fixed.png (same isolated home in prior
checkpoint) proved vae_recovery_dir no longer blocked. NEXT failure was empty
manual_loras; fixed afterwards but desktop NOT rerun yet. Likely next issue:
UI sends model_keyword hook by default (Generator keywordActive implies manual_loras
Some(empty)). Need inspect full actual Generator payload/port hook semantics, not
silently drop hooks. No desktop-generated image exists yet; vram_estimator banner
also remains. Current sbt scala-torch with all tiny props. NUI unchanged this turn.
## First actual native desktop run (2026-11-03)

Ran NUI JavaFX Launcher with --home isolated under scala-torch/.superpowers,
--autogen, --screenshot and --exit-after. Exact retained home:
C:/Users/Mizushima/dev/scala-torch/.superpowers/nui-ui-8389b9ccd0c940b787ecad1783f6d50b
Pointer: .superpowers/latest-native-ui-home.txt. Settings fp32/cpu, last_model and
checkpoint dirs tiny-sd-pipe; UI params cat/64x64/3 steps/seed42. No normal settings
changed. CIM observed Java UI plus Java native worker, no python.exe.

Found real UI bug: VramModes.normalizeSettings maps legacy cpu to sequential_offload.
Fixed in NUI native branch: optional nativeEngine preserves explicit cpu; legacy
Python mapping unchanged. Round2Suite 3 passed. SdfxApp supplies native selection.
Latest screenshot native-ui-native-fixed.png confirms Ready cpu, loaded tiny-sd-pipe,
Python-free startup. GENERATION STILL FAILS: unsupported non-default option
vae_recovery_dir. UI Generator always supplies real output directory, unlike the
EngineClient integration fixture. Also visible unsupported vram_estimator banner.
No generated output PNG exists. Next implement actual Generator envelope semantics
(including vae_recovery_dir; likely checkpoint/vae/extra next), then rerun desktop.
Do not claim desktop generation works based on prior EngineClient integration.

Second screenshot native-ui-cpu-fixed.png is an accidental Python bootstrap run:
SDFX_SCALATORCH_* env were omitted on that invocation. sbt run does not reliably
inherit prior client env: SET ALL THREE NATIVE VARIABLES on EVERY ui/run invocation.
No Python installation was performed. Third run explicitly supplied variables and
confirmed fixed CPU load. Use euler-demo.json in tiny root for scheduler env.
Current sbt server NUI, UI/worker processes exited after screenshots. Full goal active.
## Native VAE selection (2026-11-03)

set_vae now stages a replacement VAE and pipeline before publication; failure or
denied publication leaves previous pipeline/model metadata untouched. Null/empty
path restores root/vae. Supports Diffusers directories (including indexes) and
standalone Diffusers-key safetensors using checkpoint VAE config. load_model also
accepts vae_path. VAE tiling and original LDM/A1111 key remapping remain unsupported.
Set rejects changed latent channels/spatial factor. Metadata includes selected
VAE name/hash, refreshed model_token and original checkpoint hash. Standalone VAE
hash is file SHA256; indexed hash concatenates inventory source bytes.

PretrainedVae exposes optional configOverride/filename. SafeTensorSet basename
validation permits spaces/dots/Unicode while rejecting separators/colon/NUL and
wrong suffixes, so ordinary external filenames work without copying model config.
Tests cover standalone file in a config-less directory, missing file preserving
metadata, denied-publication preserving state, restoration, and existing generation.
Shard tests: 6 pass / 1 Linux-only skip; backend+process: 3 pass. NUI native branch
integration now calls client.setVae(file) and None before generation; passed in
17.014 seconds. Full application migration still incomplete. Current sbt server
is NUI with integration env; stop before building scala-torch.
## Native auxiliary file catalogs (2026-11-03)

NativeFileCatalog and list_files handle NUI file kinds vae/lora/controlnet/embeddings/
upscalers/face_restore/svd/zero123/photomaker. Scan recursively, filter standard weight
extensions case-insensitively, skip hidden directories, deduplicate real paths,
collapse ControlNet config.json and SVD/Zero123 model_index.json child folders,
and return relative names/absolute paths/rounded MiB sizes. No symlinks followed
(including file links, a deliberate current limitation vs Python file-link behavior).
Traversal/results bounded; cancellation checked during scan and directory size walks.
Discovery does not imply native application of those extensions is implemented.

Catalog unit and existing backend suites: 3 passed. NUI NativeScalaEngineSuite
now calls real client.listFiles for VAE and client.systemStatus, then load/generate/
preview/unload; passed in 17.034 seconds on the native branch. NUI sbt server is
currently active with native integration variables, no SDFX native launch env set.
Full migration remains incomplete; actual desktop interaction, normal models and
GPU execution, full plugins and production packaging still need implementation.
## Native system status (2026-11-03)

system_status is now an immediate reader-thread control in NativeWorker. Backend
systemStatus must only read published snapshots; NativeDiffusionBackend publishes
immutable volatile model name/kind/hash metadata on successful load/unload.
NativeSystemStatus reports timestamps, JVM uptime, CPU Float32 runtime/device,
OS/platform, JVM heap used/committed, and OS physical RAM when MXBean supports it.
Heap is not mislabeled process RSS; unavailable RSS/CUDA are explicit status strings.
Parameters currently must be empty, matching EngineClient.systemStatus.

Tests prove status answers while an injected generation is blocked, model hash
appears after load and disappears after unload, and a separate real worker JVM
returns metadata before image generation. Final focused run: 5 passed, no skips.
Current sbt server is scala-torch with only scalatorch.clip.dir set (backend gate
uses sibling model folders); VAE full-suite gates need all three props restored.
NUI remains on its native branch, unchanged this turn. Full migration incomplete.
## NUI native application launch route (2026-11-03)

On NUI dedicated feat/scala-torch-native-engine branch, ScalaTorchLaunch now reads
SDFX_SCALATORCH_CLASSPATH_FILE, SDFX_SCALATORCH_MODEL, SDFX_SCALATORCH_SCHEDULER and
optional SDFX_SCALATORCH_JAVA (Java25; important for Java21 packaged runtimes).
AppServices chooses it before PythonEnv when configured; explicit launchOverride
still wins. SdfxApp skips Python bootstrap when any native variable is set.
Partial config fails explicitly, not via fallback; native dry-run rejects.
Classpath requires absolute existing entries; Java/model/config paths validated.
No change to default Python selection when no native environment variables exist.

Two launch tests passed and ui/compile passed (one preexisting deprecation warning).
Real NativeScalaEngineSuite now constructs launch through ScalaTorchLaunch and
passed its model/sampler load/64x64 generation/preview/unload flow in 17 seconds.
Desktop interaction was NOT exercised yet. Native AppServices still encounters
unsupported list_files/system_status and other missing features; model loading
must be configured CPU/Float32. Full migration remains incomplete.
NUI sbt server currently active with integration env values; stop before building
scala-torch. docs/scala-torch-native-engine.md documents native launch variables.
## Native iteration generation (2026-11-03)

Native worker accepts n_iter 1..100 with batch_size still 1. Each iteration uses
seed+index, own solver history and image index/total progress metadata. Signed-Long
seed overflow is rejected before inference. Per-iteration tensors are released
between images; PNGs go into a unique staging directory under the caller output.
Only after all images finish does context.beginPublication admit moving them to
unique output names and returning paths. Cancel/failure removes owned staged and
already-moved files without touching preexisting outputs. Files.move does not replace.

Backend tests verify seeds 100/101, second-image cancellation cleanup, denied
publication cleanup, overflow rejection, and existing indexed/single model paths.
Backend plus real JVM subprocess tests passed (3); after adding publication/overflow
assertions backend suite passed again (2). Runtime stays Python-free. This adds
iteration count, not tensor batching, and does not establish complete NUI migration.
NUI unchanged this turn; sbt is scala-torch with all tiny model properties enabled.
## Indexed Diffusers weights (2026-11-03)

SafeTensorSet.open now accepts a validated basename (default model.safetensors)
and exposes sourceFiles including index and shards. U-Net/VAE use this reader
with diffusion_pytorch_model.safetensors, enabling indexed weights while retaining
existing assignment/count/byte-budget checks and ambiguous single+index rejection.
Native worker fingerprints actual source inventory rather than hardcoded weight
files, including indexed CLIP weights which PretrainedClip already supported.

Verification: ShardSuite 6 passed / 1 Linux-only descriptor test skipped on Windows;
22 U-Net/VAE/backend tests passed including pinned model numerical references.
Then backend suite 2 passed with an additional indexed-all-components fixture:
CLIP/U-Net/VAE copied to one-shard indexes, hashed, loaded and generated a PNG.
Multi-shard routing is tested separately in ShardSuite. This does not establish
normal SD model or GPU inference; it removes a file-layout restriction only.
Current sbt server is scala-torch with all three pinned tiny model props enabled.
NUI unchanged this turn. Both dedicated branches remain in use; goal incomplete.
## Native CLIP skip integration (2026-11-03)

DiffusionImageRequest adds clipSkip=1; pipeline validates against encoder depth and
uses skipLastLayers=clipSkip-1 for positive and negative conditioning. Existing
ClipTextModel already applies final LayerNorm to earlier-layer outputs, matching
NUI SD1.x hidden_states[-clip_skip]. Native worker now parses clip_skip rather than
rejecting non-default values. SD2/SDXL layer-selection semantics remain unimplemented.

Added pinned full-image clip-skip2 reference (optional argument to offline image
reference generator). Maximum error 2.6375055e-6 against Diffusers/Transformers.
16 tests passed (VAE, generation, inactive options, backend); then backend test
passed with clip_skip=2 via the native request map and Uniform DPM generation.
Bounds tests reject zero/excess skip. Existing default full-image reference still
matches at 3.7252903e-6. NUI cross-process test not rerun in this turn.
Current scala-torch sbt server has all three tiny model directory properties enabled.
Full migration remains incomplete; native UI launch, normal models/GPU, SDXL,
batching, LoRA/img2img/Hires/extensions and packaging are still outstanding.
## DPM++ Uniform support (2026-11-03)

DpmSchedule.uniform implements Diffusers DPM linspace: steps+1 evenly spaced
training indices, ties-to-even rounding, descending order, discard zero endpoint,
terminal zero sigma, unit initial noise. Native worker accepts Uniform for DPM++ 2M
as well as Karras, eliminating the earlier catalog/execution gap for normal step
counts. This is explicit DPM linspace, not Euler's endpoint-inclusive schedule.
Duplicate-sigma schedules are still rejected by DpmSchedule validation.

Extended the offline pinned Diffusers 0.33.1 reference generator with an optional
uniform argument. uniform-trajectory.tsv adds 12 updates across epsilon/sample/
v_prediction; static Scala tests verify timestep/sigma and recurrent solver output
against those rows. Existing Karras reference still passes. Backend test now writes
both Karras and Uniform real tiny-model PNGs. Final focused run: 3 tests passed.
Python was used only to create the checked-in oracle, never by native runtime/tests.
NUI integration was not rerun this turn; its earlier Karras path remains verified.
Current sbt server is scala-torch with tiny-model property enabled. NUI branch and
product remain unchanged this turn. Full migration remains active and incomplete.
## Native model/sampler catalog checkpoint (2026-11-03)

NativeDiffusionBackend now supports list_models and list_samplers. Model listing
returns only the configured loadable root when it is beneath one of the requested
directories; it reports directory size without following symlinks while walking.
This is not yet arbitrary checkpoint discovery/loading. Sampler catalog advertises
Euler (Uniform default) and DPM++ 2M (Karras default), plus Automatic/Uniform/Karras.
DPM++ Uniform remains unsupported by execution; the protocol catalog does not
encode per-sampler schedule compatibility. Unit tests cover directory filtering
and names. NUI NativeScalaEngineSuite now verifies typed model and sampler lists
before load/generation/unload and passed (16.948 seconds).
Both repos still use their dedicated branches. NUI sbt server is currently active;
shutdown before returning to scala-torch builds. Classpath file was refreshed.
## Real NUI EngineClient integration (2026-11-03)

Created NUI branch feat/scala-torch-native-engine BEFORE edits, as requested.
NUI commit e9ab842 adds NativeScalaEngineSuite plus docs/scala-torch-native-engine.md.
It launches NativeWorkerMain in Java 25 through the REAL EngineSupervisor and
EngineClient, using NUI GenerationRequest/GenerationParams default serialization.
Passed: hello, fp32/cpu load, model hash, 64x64 three-step DPM++/Automatic generation,
typed progress 1/2/3 and preview, final PNG and seed, unload. Runtime was 17 seconds.
No Python launched. This validates the real client transport but not the desktop
UI or all features. NUI default launch is unchanged; both repos still have substantial
migration work (catalog, normal models, CUDA, SDXL, batching, extensions, packaging).

NUI sbt server currently running (2.0.12), with SCALATORCH_WORKER_CLASSPATH_FILE set
to C:/Users/Mizushima/dev/scala-torch/.superpowers/worker-classpath.txt and
SCALATORCH_TINY_MODEL to the .superpowers/models/tiny-sd-pipe directory. Shut it down
before switching back to scala-torch sbt. Classpath file was extracted from
show models/Test/javaOptions (sbt export fullClasspath returns virtual refs in sbt2).
Regenerate it after rebuilding jars. Native integration env is read at sbt startup.
## Native live preview checkpoint (2026-11-03)

NativePreview implements SD1.x cheap latent-to-RGB projection, float32 channel
factors and byte rounding from NUI python/sdfx/preview.py. It scales to max side
256 with bilinear interpolation and encodes JPEG at quality 0.7 using ImageIO.
Resampling/JPEG bytes are not claimed identical to Pillow. CPU single-image
four-channel latents only; SDXL/TAESD and batch grids remain future work.
Native backend now accepts preview_every (default 2), preview_method (default
approx; none disables), emits preview with jpeg_b64 and step between progress
events, and skips the final step as NUI does. This supersedes the earlier requirement
to disable previews. Unit projection/JPEG tests and backend generation passed;
the subprocess test now runs three steps and verifies the step-2 JPEG (256x154),
final PNG, typed error, and clean shutdown. Final subprocess test passed separately.
NUI itself remains unmodified. Next: dedicated NUI branch, actual EngineClient
integration with full default envelope, catalog operations, broader model support.
## NUI generation envelope validation (2026-11-03)

NativeGenerationDefaults validates and strips known inactive options emitted by
NUI GenerationParams/GenerationRequest. Active unsupported options (batch, clip skip,
LoRA, Hires, hooks, etc.) and unknown nested fields fail rather than being ignored.
Automatic resolves DPM++ 2M to Karras and Euler to Uniform, checked against NUI's
python/sdfx/schedulers.py and tests/test_schedulers.py. The tiny backend test now
uses all these inactive fields and Automatic, proving actual generation still works.
Validation and real-model tests passed (2); standalone process test also passed
before the last inactive-preview option addition.

IMPORTANT: preview currently requires preview_every=0 and preview_method=none;
NUI defaults (2/approx) still need implementation. This is not full default UI
compatibility. Next integrate real EngineClient serialization, first creating a
NUI branch, and add preview plus catalog operations. NUI remains untouched.
## Native JVM subprocess verification (2026-11-03)

NativeWorkerProcessSuite now launches NativeWorkerMain in a separate Java process
using Compile/fullClasspath (exported by the models Test/javaOptions setting).
sbt 2's forked test java.class.path contains only sbt worker jars, so it cannot be
reused for the application child process. This was reproduced as ClassNotFoundException
and fixed with sbt's fileConverter for virtual classpath entries.

The gated tiny-model subprocess test passed: hello, model load, two progress events,
DPM++ PNG result, exact Long.MaxValue seed round trip, unload, typed no_model error,
shutdown acknowledgement and zero exit code. Child removes JAVA_TOOL_OPTIONS and
Python environment overrides. Runtime command is Java plus application classes/jars;
no Python is launched by the worker. Test verifies a real 10x6 PNG, not model quality.
Next: accept NUI's actual default request envelope and integrate its EngineProcess
on a dedicated NUI branch. Full application migration remains unfinished.
## Native diffusion adapter checkpoint (2026-11-03)

NativeDiffusionBackend now implements load_model/unload/txt2img with the real
CPU Float32 pipeline, explicit Euler checkpoint config, Euler Uniform/Karras and
DPM++ 2M Karras. NativeWorkerMain provides a UTF-8 stdin/stdout entry point.
Typed backend errors preserve no_model/unsupported codes. Load stages new weights
before atomic publication, computes a SHA-256 fingerprint of model/tokenizer files,
and preserves existing state on unsupported settings or cancellation. Generation
emits progress and writes a unique create-new PNG after publication admission.
11 native worker/control/protocol/backend tests passed, including gated real tiny
checkpoint load, 10x6 DPM output, unload, unsupported settings and cancellation.
Run: models/testOnly scalatorch.models.worker.NativeDiffusionBackendSuite
scalatorch.models.NativeWorkerSuite scalatorch.models.NativeProtocolSuite
scalatorch.models.NativeJobControlSuite, with scalatorch.clip.dir set to tiny text_encoder.

Still incomplete: full NUI default parameter envelope (currently strict subset),
list_models/list_samplers, standalone subprocess verification, actual EngineProcess
integration, normal model/CUDA inference and remaining NUI operations. No claim of
full NUI compatibility. NUI remains clean on main; CREATE A DEDICATED BRANCH before
any NUI edits. The older paragraph below describing no adapter is superseded here.
# Handoff — 2026-11-03

## Windows continuation (2026-11-03)

`NativeWorker` now connects JSON envelopes and NativeJobControl to an injected
`NativeWorkerBackend`: bounded Reader framing, immediate ping/protection/interrupt/
restart-authorization/shutdown handling, and serialized backend work on a daemon
thread. Backend contexts expose cancellation, event emission and atomic publication
admission. Final results acquire publication admission too; histories/cancellation
remain job-local. EOF/shutdown cancel active work and discard pending work. Close
waits up to five seconds for cooperative completion; it cannot forcibly interrupt
a running numerical kernel. Tests use a blocking injected backend to prove ping
responsiveness, owned cancellation and recovery, plus malformed stream recovery,
protection and shutdown. This is not yet the real-model adapter or a subprocess
entry point: connect model listing/loading/txt2img and typed backend errors next,
then test the actual NUI EngineProcess on a dedicated NUI branch.

Native JSON transport primitives now exist in `worker/NativeProtocol.scala`:
bounded/depth-limited duplicate-rejecting JSON, exact BigDecimal values (including
64-bit seeds), validated request IDs/owners/object params, and synchronized UTF-8
hello/result/error/event emission. Tests cover Unicode/newline framing, malformed
envelopes, exact numbers and 80 progress emissions from four concurrent threads.
The parser targets NUI's string-ID requests; legacy numeric/missing IDs and unknown
envelope options (including image encryption transport) reject explicitly for now.
This is not yet a runnable worker: bounded stdin framing, dispatch, control/job
thread wiring, model handlers and integration tests remain next. The emitter's
hello describes CPU inference and must advertise only implemented features.

Native worker control has begun in `models/worker/NativeJobControl.scala`.
It serializes admission, owner/target matching, interruption protection revisions,
queued cancellation, publication admission and shutdown under one monitor. Each
job exposes a volatile cancellation flag for the generation loop; history belongs
to that job, not a global flag. Cancelled queued jobs still reach the worker for
terminal responses; shutdown explicitly returns discarded jobs. Queue size is
bounded. `NativeJobControlSuite` covers stale owners, duplicate admission,
protection, publication, queued cancellation and shutdown, plus 32 races between
publication and interrupt. This is a control primitive, not yet a runnable
JSON Lines worker or NUI integration. Next connect request envelopes, the reader
and worker threads, and the native model operations according to the boundary doc.

Full tiny-image parity is now checked against Diffusers 0.33.1 / transformers
4.49.0 / torch 2.7.1 CPU. With explicit noise, two prompts, CFG 3 and three
DPM++/Karras steps, Scala's raw 1x3x6x10 decoded output matched all 180 values
with max absolute error 3.7252903e-6. Static hashed reference bytes and an offline
generator live in `models/src/test/resources/dpm/`; ordinary Scala tests do not
invoke Python. This supersedes the earlier absence of full-image parity for
this specific tiny case only. Practical models, more prompt/schedule cases,
GPU diffusion and the NUI application remain unverified/unintegrated.

Independent DPM verification now exists in `DpmReferenceSuite` and
`models/src/test/resources/dpm/`. A disposable ignored reference venv used
Python 3.12.12 / torch 2.7.1+cpu / diffusers 0.33.1 to export three four-step
trajectories (epsilon/sample/v_prediction). Scala recursively feeds its own
outputs and matches all 12 upstream updates within 1e-5 absolute error. Training
sigma generation agrees within 3e-5 and rounded model timesteps match exactly.
The final standalone reference suite passed with SHA256 fixture validation.
Normal tests/inference use only the static TSV, not Python. Full-image parity
and other scheduler configurations remain unverified. See fixture PROVENANCE.md
for exact inputs and regeneration. Reference tools/env are under `.superpowers`.

DPM image integration: `EulerImagePipeline.generateDpm(request, DpmSchedule, ...)`
now runs VP DPM++ 2M with unit initial noise, unscaled model inputs, local history,
shared prompt/guidance/decoder handling and cooperative cancellation.
`DpmSchedule.karras(noise, steps)` rounds model timesteps for the linear/scaled-linear
training profile. Public tiny-model integration verifies finite images, progress,
exact seed replay and cancellation/recovery alongside the existing Euler path.
PretrainedVaeSuite + initial DPM solver suite: 17 passed, no failures/skips; an
additional schedule test checks 999/687/146/0 timesteps and unit initial scaling.
This supersedes the earlier note that the DPM image loop is unconnected. Native
worker/NUI integration, complete DPM config loading, independent Diffusers numeric
parity, practical GPU diffusion and real-model quality remain open.

DPM++ follow-up: `DpmPlusPlus2M` implements deterministic VP midpoint updates,
epsilon/sample/v-prediction conversion, first-step order one, order-two history,
and terminal-zero clean output. It is a pure indexed API: callers pass the prior
`predictedOriginal`, so rejected calls do not advance a hidden cursor. Inputs
use VP scaling and initial unit noise, NOT Euler latent/input scaling.
Analytical tests cover a full three-step scalar trajectory, prediction conversion,
history gradients, rejected histories/budgets and replay. Reference is NUI's pinned
diffusers 0.33.1, not the initially inspected newer version:
https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
The image loop and DPM timestep/config construction remain to be connected;
in particular this version rounds Karras timesteps for linear/scaled-linear betas.
Do not reuse the Euler image loop unchanged or claim full NUI sampler parity.

Karras follow-up: `DiffusionNoiseSchedule.karrasEuler(steps, rho=7)` creates
descending Karras sigmas, a terminal zero, and fractional model timesteps using
log-linear inversion of training sigmas. The formulas were checked against local
NUI `python/sdfx/sigmas.py` (`karras`, `DiscreteModel.sigma_to_t`). Analytical tests
use training sigmas 0.5/1/2 and rho 1/2, plus endpoints, single-step schedules and
invalid rho values. EulerConfig now accepts `use_karras_sigmas=true` and retains
leading-spacing initial-noise scaling. Euler/config suites: **11 passed, 0 failed,
0 skipped**. This adds the schedule, not the DPM++ solver. Full NUI sampler parity
and external float32 reference comparisons remain open.

`TinyDiffusionRun` now accepts optional trailing `[seed [euler-config.json [steps]]]`
arguments. It validates the explicit Euler config before loading weights and
forwards both schedule and prediction to the image pipeline. With no config,
the original four-step profile remains the default. This does not auto-convert
the tiny checkpoint's DDIM configuration.

Windows CLI verification used prompt `cat`, seed 42, four steps, and an explicit
Euler JSON with 1000 training steps, scaled-linear betas 0.00085 to 0.012,
epsilon prediction, linspace spacing and steps_offset=1. The output
`.superpowers/outputs/windows-config-euler-42.png` exactly matches the prior
fixed-profile PNG: SHA256
`808c2836dfc16dca4bdaf9f9b83a8d839a60b74134b62c72944007cd428a6a32`.
This is a tiny CPU pipeline check, not practical image-quality evidence.

Euler JSON loading is now implemented as `EulerConfig.read(path)` /
`EulerConfig.fromJson(json)`. It validates bounded UTF-8 JSON with duplicate
rejection, linear/scaled-linear or explicit trained betas, prediction and spacing.
`config.scheduler(steps)` returns both schedule and prediction. Image callers use:

```scala
val scheduler = EulerConfig.read(configPath).scheduler(steps)
val image = pipeline.generate(request, scheduler.schedule, prediction = scheduler.prediction)
```

Unsupported sigma transformations/interpolation/zero-SNR rescaling and unknown
fields reject explicitly. This API loads Euler configs, not arbitrary scheduler
families: the downloaded tiny pipeline's DDIM config requires an explicit future
conversion layer matching NUI's sampler selection. That conversion and NUI's
default DPM++/Karras remain open. Config-file tests also cover malformed UTF-8,
size bounds, trained coefficients, duplicate keys and unsupported semantics.

Image prediction follow-up: `EulerImagePipeline.generate` now accepts an explicit
`prediction: DiffusionPrediction` (default Epsilon) and forwards it to Euler.
Previously the image path always used epsilon even though the scheduler supported
other modes. The public tiny-model test confirms exact default/explicit-epsilon
replay and finite, different output when selecting VPrediction. This is parameter
plumbing coverage, not a quality/parity claim for a v-prediction-trained model.
EulerSchedulerSuite + PretrainedVaeSuite: **19 passed, 0 failed, 0 skipped** with
the pinned external CLIP/U-Net/VAE configured. Scheduler JSON loading remains open.

User constraint: create a dedicated branch before changing NUI. Its current
`main` checkout was rechecked clean; no NUI files have been modified.

Scheduler follow-up: the pinned tiny checkpoint carries `steps_offset=1`.
NUI `build_scheduler` preserves this when changing spacing. Euler now ignores
this offset for linspace/trailing, applying it only to leading, consistent with
the upstream Euler scheduler. A regression test failed on the old rejection,
then the Euler/image suites passed all 7 tests after correction. Config-file
loading itself remains to be implemented. Reference:
https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/schedulers/scheduling_euler_discrete.py

The user has resumed the goal in `C:\Users\Mizushima\dev\scala-torch`, targeting
`C:\Users\Mizushima\dev\stablediffusion-nui`. The stop instruction below is
historical and superseded by that explicit continuation. NUI already launches
as a standalone application; the goal is to replace its internal Python
dependency while preserving its current features, not merely make it launch.
NUI remains clean at d9f13ea; no integration has been performed yet.

Windows validation found and corrected two portability defects:

- Shared CUDA driver loading rejected Windows before calling the driver.
  It now selects `nvcuda.dll` on Windows and `libcuda.so.1` on Linux, requiring
  a 64-bit JVM. Existing real-device probe: two reproducible failures before,
  three passes after. Full FFM suite: 35 passed with the CUDA gate enabled.
- `core.autocrlf=true` changed pinned text-fixture bytes and broke SHA256
  verification in CLIP/tokenizer/U-Net/VAE tests. `.gitattributes` now disables
  conversion for test resources. Existing working-copy resource CRLFs were
  restored to the repository's LF bytes; expected hashes were not changed.

The machine has Windows 11, Oracle Java 25.0.5 and NVIDIA driver 610.88.
The user reports installing the Windows CUDA driver. Use one sbt process at
a time. The sbt 2 thin server retains its original environment: restart it
after changing `SCALATORCH_CUDA_TESTS`. Its client returned zero even on failing
tests, so inspect test results, not just the process exit code. Multi-task
commands must use one quoted semicolon-separated argument. `testOnly *`
forces actual test execution rather than the `testQuick` cache.

Final current-tree Windows regression: **326 passed, 0 failed, 23 skipped**.
Per-module passed/skipped: core 19/0, cpu 15/0, ffm 35/0, autograd 74/0,
io 16/1, nn 43/0, models 124/22. All final XML reports have zero failures/errors.
The IO skip requires Linux `/proc/self/fd`; 22 model cases need external pinned
checkpoints absent from this Windows checkout. Bundled independent CLIP,
tokenizer, U-Net and VAE fixtures did run after byte restoration. Public-model
prompt-to-image/cancellation gates remain unverified here. Core was rerun with
`set core / Test / parallelExecution := false; core/testOnly *` because its
parallel aggregate/report output undercounted executed cases (19 serial passes).
Other modules ran with `testOnly *`, real CUDA enabled through the server's
environment. `git diff --check` and staged diff checks are clean.

Next: checkpoint scheduler configuration and NUI prompt semantics, then native
worker integration through the existing protocol boundary. Windows GPU primitive
coverage does not establish a CUDA diffusion backend or Python-free NUI.

### Public image-model verification on Windows

The next continuation downloaded the eight CLIP/U-Net/VAE/tokenizer files from
`hf-internal-testing/tiny-sd-pipe` at the pinned revision
`39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917` into
`.superpowers/models/tiny-sd-pipe`. This is ignored local test data.
The existing suites verify checkpoint hashes before their public parity gates.

`PretrainedClipSuite`, `PretrainedUNetSuite`, `PretrainedVaeSuite` and
`ClipFastTokenizerSuite`: **32 passed, 0 failed, 0 skipped** on Windows.
This includes public weights, independent Candle outputs, prompt effects,
seed/noise replay, progress, cancellation and recovery. The earlier statement
that those image gates are unverified on Windows is now superseded. This does
not verify practical image quality, SD1.5/SDXL, GPU diffusion, or NUI integration.
The remaining external Llama/SmolLM2 gates were not rerun.

To reproduce in PowerShell, first shut down the existing sbt thin server, then:

```powershell
$modelRoot = 'C:/Users/Mizushima/dev/scala-torch/.superpowers/models/tiny-sd-pipe'
$env:JAVA_TOOL_OPTIONS = "-Dscalatorch.clip.dir=$modelRoot/text_encoder -Dscalatorch.unet.dir=$modelRoot/unet -Dscalatorch.vae.dir=$modelRoot/vae"
sbt "models/testOnly scalatorch.models.PretrainedClipSuite scalatorch.models.PretrainedUNetSuite scalatorch.models.PretrainedVaeSuite scalatorch.models.ClipFastTokenizerSuite"
```

Git identity is now configured. The Windows/config changes are ready for the verified checkpoint commit.

## Historical stop point and user direction

The user ended the previous autonomous implementation goal and requested:
write the goals/current state into Markdown, commit, and finish. **Stop after
this checkpoint. Do not continue overnight or start NUI integration without a
new user request.** The roadmap below preserves their intended next direction;
it is not an active instruction to keep running after this turn.

The immediate product objective is to remove Python from the **current local
`/home/mizushima/dev/stablediffusion-nui`**, while preserving its existing
features. Advance scala-torch toward that application and begin NUI integration
when the components are usable. A broader Scala-native PyTorch-inspired library
for LLM inference/training, and eventually a Transformers-equivalent layer,
remain desired longer-term directions.

The user authorized working on an NUI branch and trying partial integration.
They then emphasized the WSL2 environment: do not treat GUI launch as verified
here. At a suitable verified checkpoint, update NUI main; retain explicit
Windows UI verification requirements. No NUI branch, integration, merge or push
was performed during this session.

## Repositories and environment

- scala-torch: `/home/mizushima/dev/scala-torch`, branch `feat/tensor-foundation`.
  Java25, Scala3.3.8 LTS, sbt2.0.10. Communicate with the user in Japanese.
- NUI: `/home/mizushima/dev/stablediffusion-nui`, clean `main` at
  `d9f13ea72409ea543948bacb5807536ba6dc4f83` when last inspected. Product files
  were not modified. Read its `AGENTS.md` and current `HANDOFF-CODEX.md` before
  implementation; preserve partial/unmerged work and the existing application.
- NUI's older issue/release handoff goals are historical context, not a
  replacement for this user's Python-removal direction.
- Run one foreground sbt process at a time. Keep semicolon-separated tasks in
  one quoted argument; use `testOnly` when execution rather than cache reuse
  must be established. Java25 still emits the known Scala Unsafe warning.

## Verified components

Existing committed work includes CPU autodiff, Java25 FFM CUDA inference/training,
Llama/SmolLM2 checkpoints and tokenization, pretrained CLIP/U-Net/VAE, VAE posterior
sampling/KL/scaling, Euler scheduling, and native CLIP fast-profile BPE.

Recent checkpoints before this commit:

- `9f08252`: published VAE loading and six independent encoder/decoder cases.
- `413c510`: differentiable posterior and explicit latent scaling.
- `a2e1972`: Euler schedules and public token-conditioned denoising loop.
- `1061c95`: native CLIP fast-profile BPE,15 independent token-ID cases.
- `092fb07`: refreshed inventory of the actual current local NUI.

The pending image-pipeline work is included in this checkpoint:

- `EulerImagePipeline`: single-image CPU inference with prompt/negative prompt,
  explicit Euler schedule, JVM Gaussian seed or caller-supplied noise, guidance,
  completed-step callbacks and cooperative cancellation/recovery.
- `ClassifierFreeGuidance`: exact-shape branch combination with gradients.
- `DiffusionImageIO`: finite NCHW RGB to PNG, with create-new output semantics.
  It preserves existing destinations; the caller owns staging/publication.
- `TinyDiffusionRun`: standalone tiny-model export with an explicit fixed Euler
  profile. It is not a general scheduler/pipeline configuration loader.

Verification at this stop point:

- Latest full all-module/public/CUDA run, at the posterior checkpoint:
  **337 passed,0failed,0skipped**. This predates the subsequent tokenizer/Euler
  and image API additions; do not call it a full test of this final tree.
- Subsequent affected public diffusion/tokenizer run: **72 passed**;
  existing LLM/SmolLM2 tokenizer run: **5 passed**, no failures/skips.
- Latest image API/public VAE targeted run: **16 passed,0failed,0skipped**,
  covering guidance gradients, PNG readback/existing-file preservation, prompt
  effects, seed/noise replay, progress, cancellation and recovery.
- Read-only review found no correctness blockers in the image pipeline/export.
- Standalone `TinyDiffusionRun` completed four steps and wrote a32x32 PNG:
  `.superpowers/outputs/native-tiny-euler-42.png`. It was visually inspected:
  colored noise from tiny test weights, **not** a semantically recognizable cat
  or evidence of practical image quality. Logs/artifacts under `.superpowers`
  are ignored local evidence, not published assets.

The tiny demo used prompt `a red cat on a blue chair`, empty negative prompt,
seed42, guidance7.5, and explicit1000-step scaled-linear betas0.00085→0.012 with
four Euler inference steps. Models were the pinned tiny-sd-pipe checkpoint
`39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`.

## NUI integration roadmap

See [the current target and completion criteria](docs/research/nui-current-migration.md).
The refreshed inventory contains **48 worker requests,121 standard plugins,
100 parameter fields and138 Python modules**. All migration rows remain open.
The older8823aa4 inventory is historical; current additions must not be dropped.

1. Finish the application-ready image path: checkpoint scheduler configuration,
   NUI prompt semantics, image/metadata handling and required model variants.
   The native fast tokenizer is only a component: slow ftfy/HTML repair,
   weighted75-token chunks and textual-inversion expansion remain unimplemented.
2. Start NUI integration on a branch behind its existing protocol-v1 JSON Lines
   boundary. `AppServices` still launches `PythonEnv.engineLaunch`. Preserve
   request IDs, progress/results/errors, queue admission, cancellation ownership,
   interrupt protection, recovery and publication. Verify JVM worker startup,
   generation/cancellation and external MCP stdio/HTTP without a GUI.
3. Bring validated integration to NUI main at a sensible checkpoint. Do not
   relabel missing behaviors as complete or remove the existing backend before
   corresponding native behavior has been implemented and verified.
4. Implement remaining schedulers (including NUI's default DPM++2M/Karras),
   practical CUDA diffusion, SD1.5/SDXL, image/video tools, model management,
   plugins and diagnostics. Python `run_code` and legacy Torch-format migration
   require an explicit compatibility policy; no feature deletion was approved.
5. Replace Python interpreter/venv/pip bootstrap, packaging and maintenance
   dependencies. Final acceptance is a Python-absent environment with actual
   NUI GUI/MCP workflows, real-model images and preserved settings/replay.
   **Windows GUI startup/operation and package testing remain unverified.**

A standalone tiny CPU PNG, a native worker subset, or library tests alone do not
complete Python removal. Current cancellation is cooperative between model calls;
it does not interrupt a running CPU kernel or establish NUI ownership semantics.
JVM seeds do not reproduce PyTorch sequences. No full cross-framework image
pipeline parity or practical large-model throughput has been established.

## Release status

The user requested scala-torch0.1 on Maven Central at a suitable milestone.
Commit10878ca prepared/staged seven artifacts and verified a separate consumer;
there has been **no Central publication**. License/source-repository decisions
and signing/publisher credentials were still pending. Do not infer answers or
publish with invented metadata. Later code changes are not covered by that
historical staged-artifact verification; restage/reverify before release.

## Restart notes

The next session should first inspect both worktrees and this handoff, then
continue only the work the user resumes. No sbt/cargo process or implementation
agent remains intentionally running at this stop point. No main merge or remote
push was performed for scala-torch or NUI during this checkpoint.
