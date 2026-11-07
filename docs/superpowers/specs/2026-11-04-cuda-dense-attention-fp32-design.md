# CUDA dense attention FP32 path

## Intent

Reduce the measured CUDA time in NUI's native Diffusers inference. A 512x512
SD1.5, four-step Heun run on the 4 GiB GTX 1650 SUPER spent 28.7s in dense
attention score kernels and 21.2s in attention output kernels. The current
noncausal kernels accumulate in FP64, which is disproportionately slow on this
consumer GPU, and allocate eight bytes of score scratch per pair.

## Goals

- Use FP32 accumulation and FP32 score/softmax scratch for noncausal
  `scaledDotProductAttention`.
- Keep numerically stable max-subtracted softmax and NaN propagation.
- Preserve the causal attention path and its current FP64 reference behavior.
- Verify CPU-reference tolerances, workspace accounting and NUI full-checkpoint
  generation; measure kernel time and total runtime.

## Non-goals

- Add mixed precision model weights, TF32, Tensor Cores or a fused FlashAttention
  kernel.
- Change the public attention API or causal/GQA cache behavior.
- Promise bitwise parity with the CPU double-accumulation reference.

## Approaches

1. Keep FP64 and only enlarge query slices. This reduces launch count but leaves
   the measured FP64 work unchanged.
2. Convert the dense score, stable-softmax and weighted-output stages to FP32,
   halving scratch and matching the model's FP32 arithmetic. Chosen because it
   directly addresses the measured device bottleneck and remains isolated from
   causal attention.
3. Fuse score, softmax and output into a tiled FlashAttention kernel. This may
   reduce launches and scratch further, but adds a larger algorithmic change;
   consider it only if the FP32 path leaves substantial attention time.

## Constraints

- Preserve input/output tensor layouts, GQA head mapping, masked NaNs and
  max-subtracted softmax.
- Keep CUDA driver-only FFM and the existing `sm_50` PTX floor.
- Stay within `CudaSession.maxBytes` and 32-bit kernel indexing.
- Preserve existing uncommitted work and do not commit.

## Verification

- CUDA dense attention tests compare GQA and uneven head/value shapes against
  `CpuFloatBackend` within the suite's `3e-6` relative/absolute tolerance.
- Profile-enabled test asserts the three new FP32 kernel entries ran.
- Existing causal attention tests remain unchanged and pass.
- Repeat full SD1.5 NUI EngineClient generation at 512x512/four Heun steps with
  CUDA U-Net/VAE and weight offload; record per-kernel time and total duration.
