# CUDA Offload Weight Cache Design

## Goal

Reduce repeated host-to-device copies and per-copy stream synchronization during
CUDA inference when model weights are configured for offload. Preserve the
existing ability to run a model whose full weights exceed the CUDA session
budget.

## Evidence

- A profiled NUI SD1.5 256x256 run with 12 txt2img steps and masked img2img
  passed in 149.883 seconds on a 4 GiB GTX 1650 SUPER.
- CUDA upload/allocation accounted for 42.475 seconds in txt2img and 28.341
  seconds in masked img2img.
- `CudaDiffusionRuntime.weight` uploads offloaded parameters for each use and
  hands each tensor to the current `Using.Manager`, which closes it after that
  operation.
- `CudaDriver.copyHostToDevice` performs `cuMemcpyHtoD_v2` and then synchronizes
  the CUDA stream before returning.
- `CudaSession` enforces `maxBytes` for every live allocation.
- An initial bounded LRU run retained 128 MiB but left upload-call counts
  unchanged (16,502 and 11,002) because the model's ordered passes thrashed it.
- First-fit admission reduced those counts to 9,050 and 6,142 in the 256px
  NUI Service smoke. CUDA profiling reported 7,452 and 4,860 cache hits.
- Full SD1.5 EngineClient inference at 512x512 and four Heun steps passed in
  113.607 seconds with a 128 MiB retained cache. A prior 148.369-second run
  predates both dense FP32 attention and this cache, so the difference includes
  both changes.
- A size-priority cache variant reported only 156 hits and 9,471 upload calls,
  taking 115.104s. The 128 MiB first-fit variant had 4,212 hits, 5,415 upload
  calls, and took 113.607s; size priority was rejected.
- Increasing first-fit to `min(session.maxBytes / 4, 256 MiB)` retained 256 MiB
  and passed the same 512px four-step run in 112.609s. It reported 5,044 cache
  hits representing 3,328 MiB and 4,583 upload calls. A half-session cap failed
  the small allocation-budget test, so the quarter-session cap is retained.

## Proposed design

Add a bounded, per-runtime first-fit cache for uploaded offloaded parameters.
The runtime keeps the first weights that fit, in model traversal order, for its
whole lifetime. Later weights that do not fit remain transient; they do not
evict retained weights. Cache hits return the same device tensor without
registering it in the current operation scope. Non-offloaded weights retain
their current all-resident behavior.

The cache has a conservative limit derived from the CUDA session budget, capped
at 256 MiB and limited to one quarter of `session.maxBytes`. This leaves most of
the explicit session budget available to activations and temporary workspaces.
If session headroom is insufficient for the next tensor, the most recently
admitted retained weights are evicted first until the upload fits. If the cache
limit is zero or the tensor cannot fit, the tensor stays transient. Closing the
runtime releases both cache and resident weights.

No NUI setting is added in this first change. The bound is an internal safety
policy and can be exposed later if measurements show that a user control is
needed. CUDA allocation failure continues to follow existing error behavior;
the cache does not change model outputs or weight snapshot semantics.

## Validation

- Unit or CUDA-backed tests verify reuse of early weights across repeated
  ordered passes, first-fit admission, non-fitting transient uploads,
  session-budget compliance, output parity, and cleanup after runtime close.
- Existing `CudaOffloadSuite` exercises multiple consecutive U-Net calls with
  a budget below total resident weight size.
- Rebuild the native worker distribution and repeat the full SD1.5 NUI CUDA
  smoke with profiling. The FP32 dense-attention plus first-fit cache version
  passed the same 256px Service smoke in 146.536 seconds versus 149.883 seconds
  for the attention-only baseline. Upload call count fell by about 45%, while
  upload duration was nearly unchanged (42.732s/28.365s vs 42.475s/28.341s).
  Keep the bounded cache as an incremental improvement and continue profiling
  large transfers and allocation cost.

## Risks and constraints

- Retained cache allocations reduce activation headroom. The conservative fixed
  fraction and cache cap bound the extra residency; tests must use a session with
  a tight allocation budget.
- CUDA kernel execution is asynchronous. Cache eviction must occur only through
  existing tensor release behavior and must not invalidate a tensor while a
  caller still uses it. Runtime operations are thread-confined and issue each
  weight-consuming operation before requesting the next layer's weight.
- A cache that is too small may have limited impact; profile measurements will
  decide whether to tune the cap.

## Follow-up update (2026-11-05)

The original 256 MiB absolute cap was selected with a 1 GiB session budget.
NUI worker startup now fills an unspecified U-Net/VAE budget from half the
currently free device VRAM, capped at 2 GiB; explicit environment budgets still
take precedence. The first-fit cache remains at one quarter of the selected
session budget, with a 512 MiB absolute cap. Small explicit budgets therefore
retain the original behavior. On the 4 GiB GTX 1650 SUPER, a 512px/four-step
NUI EngineClient run retained 410.3 MiB without OOM, cut 1,157 upload calls
and 2,006 MiB of upload volume, and passed in 102.533s. Upload time fell by
0.777s; the 0.615s total-time change versus the preceding run is within
run-to-run variation and is not a standalone speed claim. Pure budget tests,
CUDA cache tests and full-checkpoint inference all passed.
