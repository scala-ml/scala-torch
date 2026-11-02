# Handoff — 2026-11-03

## Stop point and user direction

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
