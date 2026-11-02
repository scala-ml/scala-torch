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
(`model_keyword` check/save/delete), catalog installation and model keyword
capability advertisement are not yet implemented. Disabled ControlNet slots are
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
