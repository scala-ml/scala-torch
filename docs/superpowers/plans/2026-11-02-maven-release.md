# First Maven Central release

The user explicitly requested publishing scala-torch0.1 to Maven Central at a sensible milestone, rather than immediately interrupting feature work. Authorization covers preparing and publishing that release; do not ask for redundant publication approval. Credentials, namespace ownership and any genuinely missing license decision still require evidence before publishing.

Proposed checkpoint: finish the current CLIP conditioning slice, then audit whether the verified LLM inference/training and diffusion components form a coherent0.1 API. Full NUI replacement remains the ongoing goal and need not block an explicitly labelled early library release.

Before publication:
- Inspect existing Maven namespace ownership/credentials without printing secrets; choose unique scala-torch-prefixed module artifact names and consistent organization/version.
- Resolve project license/source repository metadata from authoritative project context, generate Central-compatible POMs, sources, docs, signatures and checksums; skip aggregate-only artifacts.
- Verify clean build, CPU tests, real CUDA gates and pinned public-model gates; verify a standalone consumer resolves staged/local artifacts on Java25 with Scala3.3.8 and sbt2.
- Document dependency coordinates, tested examples, hardware requirements and0.1 limitations; retain release notes and reproducible version tag.
- Publish using supported Central tooling after those checks; verify artifacts resolve from Central before claiming released. If credentials or namespace control are missing, complete all independent preparation and request only the missing input.

Do not equate this release milestone with completing Python-free Stable Diffusion NUI.

Initial read-only audit after CLIP implementation: no git remote is configured and no project LICENSE file exists. The approved design only lists BSD-3-Clause as a candidate, not a decision. Resolve both from authoritative context before final publication metadata.
