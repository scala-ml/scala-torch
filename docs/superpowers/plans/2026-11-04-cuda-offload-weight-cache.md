# CUDA Offload Weight Cache Implementation Plan

> **For agentic implementers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development; execute each checkbox in order.

**Goal:** Reuse a bounded subset of offloaded CUDA diffusion weights between layers and denoising steps, reducing synchronous host-to-device transfers without preventing larger models from running offloaded.

**Design:** Add a per-`CudaDiffusionRuntime` first-fit cache for parameters held in its existing host snapshot. Limit retained bytes to `min(session.maxBytes / 4, 256 MiB)`. Keep weights that do not fit the remaining cache space transient without evicting retained entries; evict the newest retained entries only when needed to meet the hard session allocation budget. Release all cached allocations with the runtime. Resident mode is unchanged. An LRU variant thrashed on sequential passes. A size-priority variant had fewer hits and more uploads; first-fit remains the best measured policy.

**Interfaces:** Keep `CudaDiffusionRuntime.upload` and public model APIs unchanged. Add package-private cache accounting only if needed by behavioral tests; prefer proving reuse and cleanup through `CudaSession.allocatedBytes`.

### Task 1: Add behavior tests

**Files:** `models/src/test/scala/scalatorch/models/CudaOffloadCacheSuite.scala`

- [x] Add a CUDA-gated test where repeated calls to the same small offloaded `Linear` return CPU-reference outputs and keep its weight allocation resident between calls; runtime close releases it.
- [x] Run the focused test and observe the intended failure on the current uncached implementation (first result leaves only the input allocation live).
- [x] Add a test that walks weights in model order, verifies earlier fitting weights stay cached when a later weight does not fit, checks outputs, and keeps live allocation within `session.maxBytes`.
- [x] Verify that a weight that does not fit the remaining cache is uploaded transiently without displacing retained entries.
- [x] Re-run the focused test and confirm the new behavior fails for the uncached implementation, not due to CUDA or fixture setup.

### Task 2: Implement per-runtime first-fit cache

**File:** `models/src/main/scala/scalatorch/models/CudaDiffusionRuntime.scala`

- [x] Add insertion-ordered mutable cache and byte accounting for offloaded host parameters only.
- [x] Implement `weight` cache hits; on miss, retain the tensor only when it fits the remaining first-fit cache space, otherwise upload transiently without evicting retained entries.
- [x] When session headroom is insufficient, evict newest retained tensors until the next upload fits.
- [x] Cache an uploaded tensor only when its size is no greater than the cache limit; otherwise register the upload with the current `Using.Manager` as today.
- [x] If no cache entries can free enough space, preserve current session-budget failure behavior by attempting the upload through the existing path.
- [x] Release cached tensors during `close` and on operation failure/cancellation.
- [x] Run the CUDA-gated cache and offload suites with tiny SD1.5 VAE/U-Net fixtures; all four tests passed and session allocations returned to zero.

### Task 3: Validate NUI and measure

- [x] Run relevant Scala-Torch model and CUDA regression suites.
- [x] Rebuild `models/workerDistribution`.
- [x] Run NUI `scalaTorchEngine/testOnly sdfx.engine.NativeRealCheckpointSuite` against full SD1.5 with CUDA U-Net/VAE and offload enabled.
- [x] Compare total time and upload profile with baselines. The 256px Service run passed in 146.536s versus 149.883s. The 512px EngineClient run passed in 112.609s with a 256 MiB cache; the 128 MiB version took 113.607s. No image-quality claim is made.
- [x] Update NUI migration evidence and `HANDOFF-KIMI.md` with the verified result; keep the goal open for further optimization and Python parity work.

## Review focus

- Cache ownership must not conflict with tensors managed by an operation's `Using.Manager`.
- Eviction must preserve snapshot behavior when source `Parameter` values change after runtime upload.
- Runtime close and exception paths must release every cached or transient allocation.
- The conservative cache limit must not silently alter resident-weight mode.

## Follow-up: size the offload budget from free VRAM (2026-11-05)

- [x] Add pure coverage for half-free-VRAM budgeting (2 GiB cap), explicit
  budget precedence, and the 512 MiB cache cap with the quarter-session rule.
- [x] Apply automatic budgets only in `NativeWorkerMain`; keep direct
  `UNetExecution.fromEnvironment` and `VaeExecution.fromEnvironment` defaults
  predictable for callers and CPU-only settings tests.
- [x] Rebuild the worker distribution and run full SD1.5 through NUI's
  EngineClient at 512px/four Heun steps on the 4 GiB GTX 1650 SUPER without
  explicit max-byte environment settings. It passed in 102.533s, retaining
  410.3 MiB and reducing upload calls/bytes; total-time change is within run
  variation.
