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
