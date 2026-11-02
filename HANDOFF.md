# Handoff — 2026-11-03

## Windows continuation (2026-11-03)

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
