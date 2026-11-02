# CUDA weight offload through native worker (2026-11-03)

With CUDA placement selected, SDFX_SCALATORCH_UNET_OFFLOAD_WEIGHTS=true and
SDFX_SCALATORCH_VAE_OFFLOAD_WEIGHTS=true independently enable operation-scoped
weight transfers. Both default to false; CUDA configuration accepts only literal
true/false. CPU placement ignores CUDA-specific settings. Existing MAX_BYTES
limits still apply to allocations tracked by each session, not total GPU usage.

UNetExecution.Cuda and VaeExecution.Cuda expose offloadWeights=false. U-Net host
snapshots live across sampling steps; GPU contexts/activations remain resident.
VAE snapshots live for each decode. Arithmetic remains FP32 CUDA and default
behavior remains resident. There is no automatic fallback or host-memory saving.
Transfers may reduce throughput. Status runtime exposes unet_offload_weights and
vae_offload_weights; existing device and host-loading vram_mode fields retain
meaning. VAE replacement carries the immutable execution configuration forward.

Tiny Euler and DPM++/Karras pipelines at2MiB per session match resident output
exactly. This is not a standard-model memory/performance claim. Normal-resolution
attention work limits, standard-model acceptance and full NUI parity remain open.
# CUDA U-Net and VAE desktop checkpoint (2026-11-03)

In addition to CUDA VAE decode, SDFX_SCALATORCH_UNET_DEVICE=cpu|cuda|cuda:<index>
and SDFX_SCALATORCH_UNET_MAX_BYTES (default1GiB) select resident U-Net prediction
and CFG across sampling steps. CLIP and scheduler updates remain CPU; latent input
and guided prediction cross the device boundary each step. Resources release
before VAE decoding, on completion/cancel/failure. No silent CPU fallback.

Hello and status report U-Net placement and individual component devices. Host
model loading remains fp32/cpu. Both selections survive VAE replacement. Separate
Java worker, real NUI EngineClient and the JavaFX desktop passed generation with
CUDA U-Net+VAE. The screenshot in HANDOFF.md shows Ready cuda:0 and Done1image;
this uses the noise-like tiny checkpoint, not normal-model quality acceptance.
The VRAM estimator, full feature parity, memory modes and packaging remain open.
# Optional CUDA VAE decode (2026-11-03)

NativeWorkerMain supports SDFX_SCALATORCH_VAE_DEVICE=cpu|cuda|cuda:<index> plus
SDFX_SCALATORCH_VAE_MAX_BYTES (default1GiB). The existing native launch variables
remain required. CLIP/U-Net and scheduler run on CPU; raw VAE decode uses resident
CUDA FP32 then releases its session. Status runtime.vae_device reports placement.
Model load stays fp32/cpu and VAE replacement preserves placement. Cancellation
checks run between operations; failures retain exact decoder-input recovery.

Actual Java subprocess PNG generation and NUI EngineClient integration passed
with CUDA decoding; see HANDOFF.md and NUI docs/scala-torch-native-engine.md.
This extends the CPU desktop checkpoint below; full GPU denoising is not done.
# Native desktop generation checkpoint (2026-11-03)

The actual NUI JavaFX app now completes a tiny CPU generation through
`NativeWorkerMain`, saves the PNG through its normal output workflow and displays
it in the gallery. This is a runtime integration checkpoint, not normal-model
quality or GPU diffusion acceptance. The tiny model produces noise-like images.
The UI's VRAM estimator still reports an unsupported request.

The `model_keyword` generation hook now resolves checkpoint mappings and applies
the planner's prompt separately for each image. It returns `effective_prompt` and
`model_keyword_source_prompt`. Iteration counters advance only after output
publication succeeds; errors and denied publication leave the counters unchanged.
Tests compare the resulting pixels with direct generation using the resolved
prompt, as well as checking counter rollback and output metadata.

Diffusers directory identity uses the complete `load_model` response `hash`
(64-character SHA256 of the named model sources), not the legacy eight-character
checkpoint-file hash. Native directory catalogs live under `_mapping_dir`:
`model-keyword.txt` supplies defaults and `custom-mappings.txt` supplies user rows.
Both use `hash,keywords,filename` CSV; filename is the model directory basename.
For example, `FULL_SHA256,red|blue,tiny-sd-pipe`. The full hash must replace the
placeholder. Missing catalogs mean no matching keywords. Legacy short-hash
catalogs do not automatically identify converted Diffusers directories.

Selected LoRAs, textual inversion embeddings and extra-network tags still fail
explicitly pending their native inference implementation. Mapping UI commands
(`model_keyword` check/save/delete) now support the configured Diffusers directory
and legacy-hash checkpoint/LoRA files. File mappings preserve `match_name` aliases.
The NUI client integration test saves a mapping, generates using it, checks the
resolved prompt, deletes the mapping and checks that it is gone. UI compilation
also passes; this test does not click the mapping panel's buttons.

Edits retain unrelated CSV records and comments, including original line endings.
They acquire the NUI-compatible Windows byte lock, reject malformed CSV, write a
temporary file and an original-byte backup, and atomically replace the mapping.
Cancellation before publication preserves the original file and removes temporary
artifacts (the reusable lock file remains). Lock contention is cancellable.
Catalog installation, bundled upstream catalogs and model keyword capability
advertisement are not yet implemented. Disabled ControlNet slots are
accepted; active or malformed slots still fail.

Evidence: isolated home
`.superpowers/nui-ui-8389b9ccd0c940b787ecad1783f6d50b`, screenshot
`native-ui-keyword-fixed.png`, output `outputs/txt2img/2026-11-03/00000-42.png`.
The output is 64x64, seed 42, DPM++ 2M Karras, 3 steps, CFG 7. PNG SHA256:
`91c346a97e597f19af3d5d90d6e8a98fca9947021ebfe7886ddc1837b498900e`.
The observed app and worker processes were Java; no Python engine was launched.

## Historical worker boundary inspection (Windows, 2026-11-03)

At the original inspection, NUI was at d9f13ea on clean main, before native branch creation.
The user requires a dedicated branch before any NUI edits. The previous issue
release instructions are historical; the current objective is Python removal.

## Verified integration points

- `AppServices.launchOverride: Option[() => EngineLaunch]` already bypasses
  Python discovery for custom engines. Use this for integration checks; do not
  replace the default launcher before native feature coverage exists.
- `EngineLaunch` supplies argument-vector command, working directory, environment
  and handshake timeout. Native runtime needs Java 25 for scala-torch FFM.
- `EngineProcess` transports UTF-8 JSON Lines and correlates string request IDs.
  The first worker line is `hello`, protocol 1. `EngineHello.python` defaults to
  empty and torch/diffusers fields are optional: a JVM worker need not invent
  Python versions. Advertise only actual native features.
- `AppServices.start()` refreshes the catalog and may load saved model/VAE
  choices. A ping-only worker does not establish usable application startup.
- `Generator` puts `output_dir` in the app-owned `.tmp` directory, then performs
  publication. A worker must return the current `GeneratedImage`/`GenerationResult`
  schema, not independently publish images into the final gallery directory.

## Lifecycle requirements from the current server

The reader handles ping, interruption protection, restart authorization,
interrupt, system status and shutdown while model work is serialized separately.
An implementation that blocks its reader inside image generation cannot satisfy
the existing cancellation contract.

Interruption admission and ownership must be atomic: target request ID and optional
owner ID identify the active or queued job. An old cancellation must not affect a
new owner. Unknown/finished targets leave no remembered cancellation. The protection
state has an enabled flag and monotonic revision. Publication state can make an
otherwise accepted interrupt too late. Shutdown acknowledges, cancels active work
and exits; EOF drops queued work. Emit whole JSON lines under one writer lock.

## Implementation sequence and acceptance

1. Add a Scala worker in scala-torch, with injected generation operations for
   deterministic lifecycle tests, strict protocol envelopes, a reader/control path,
   a serialized work queue and explicit per-job ownership/publication state.
2. Implement native model-list/load and tiny txt2img through the verified image API.
   Reject unsupported generation settings rather than silently changing them.
   Validate temp-file paths and preserve create-new semantics. Keep diagnostics on
   stderr and responses on stdout.
3. On a dedicated NUI branch, exercise `EngineProcess` and `AppServices` against
   the real JVM subprocess via `launchOverride`: hello, catalog/load, generation,
   progress, stale/queued/protected cancellation, recovery, shutdown and crash.
4. Connect the optional native launcher to UI/MCP after those tests. Verify external
   MCP stdio/HTTP, then Windows UI, real SD1.5/SDXL and packaged Python-absent runs.

This sequence does not remove the larger 48-request/121-plugin migration scope.
Tiny-model generation and a partial native worker are intermediate checkpoints.

## Current environment evidence

`sbt "engine/compile"` completed successfully in the actual Windows NUI checkout
using Oracle Java 25.0.5, sbt 2.0.12 and Scala 3.3.6. It compiled 79 Scala + 1 Java
core sources and 33 engine sources. This proves source compilation only, not
JavaFX startup, engine runtime, tests or packaging. scala-torch currently uses
Scala 3.3.8; a subprocess boundary avoids coupling the applications' classpaths.
