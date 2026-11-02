# Current local NUI: Python-removal target

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

Work toward the actual application in this order:

1. Complete a reusable JVM image-generation path using CLIP/U-Net/VAE, scheduler,
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
