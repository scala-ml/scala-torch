# Fused CUDA diffusion attention design

## Goal

Reduce the measured SD1.5 inference time spent in dense CUDA attention on a
4 GiB GTX 1650 SUPER while preserving the existing FP32 result contract,
GQA layouts, NaN behavior, session allocation accounting and Python-free NUI
runtime.

## Evidence

A native-only packaged NUI SD1.5 512x512, Euler 20-step generation completed
in 247.1s including startup. CUDA profile totals were 39.910s for FP32 score
kernels, 16.871s for softmax preparation and 7.685s for weighted output. The
totals overlap the full runtime. The existing implementation materializes
score/probability scratch in global device memory and launches three kernels
for each query slice.

## Approach

Add a fused exact dense-attention path for noncausal calls with at most 4096
key tokens. One CUDA block handles one `(batch, head, query)` row. Threads
cooperatively compute scores, reduce the stable softmax maximum and
normalizer, and produce the weighted value output. Scores/probabilities stay
in bounded per-block shared memory; no global score scratch is allocated.
This follows FlashAttention's IO-aware goal of avoiding global score
materialization (Dao et al., 2022, https://arxiv.org/abs/2205.14135), but is
not its online-softmax recurrence: this first kernel retains one complete
4096-key score row in shared memory. Keep the current three-kernel
implementation as fallback for longer key sequences.

The initial kernel uses FP32 arithmetic, max-subtracted exponentials and the
current Q/K/V tensor layouts. It preserves NaN propagation and GQA head
mapping. The CTA writes one output per lane, so fused dispatch additionally
requires value width at most 256; wider values use the existing path. It does
not change causal attention, public APIs, model weights,
workspace limits or CUDA runtime dependencies. The key-token bound prevents
shared-memory use from exceeding the fixed kernel allocation; larger shapes
continue through the current budget-sliced path. The experimental route is
disabled by default and requires `SCALATORCH_CUDA_FUSED_ATTENTION=1` because
the full SD1.5 measurement did not show a speed improvement.

## Verification

- CUDA tests compare fused outputs with `CpuFloatBackend` for existing GQA,
  uneven query/key/value widths, singleton and NaN cases, and check allocations
  return to zero.
- Profile tests verify value widths above 256 use the legacy fallback.
- Profile-enabled tests confirm fused dispatch below the bound and legacy
  dispatch above it.
- With profiling enabled, default-off dispatch is verified to use the legacy
  kernels. Both default and opt-in focused CUDA suites pass.
- Rebuild the Scala-Torch worker distribution and run native-only NUI SD1.5
  512x512, 20-step generation on the 4 GiB GPU. Compare output dimensions,
  status, peak sampled VRAM, kernel totals and wall time with the 247.1s
  baseline. Record the result without treating one seed as a quality benchmark.

## Initial result and decision (2026-11-05)

The opt-in fused kernel passed CUDA parity, NaN, GQA, 4096/4097-boundary and
allocation-cleanup checks. Through the native-only packaged NUI MCP, SD1.5
512x512, Euler 20 steps completed in 252.8s including startup. The CUDA profile
reported 71.453s for 1,280 fused U-Net attention calls and 0.850s for the VAE
fused call. The legacy 20-step profile recorded 39.910s scores, 16.871s softmax
preparation and 7.685s output (64.466s aggregate across 2,680 calls per
kernel). Overall wall times were 252.8s fused and 247.1s legacy; this single
comparison is noisy, but the fused attention total is slower, so fused dispatch
is opt-in only. The packaged run sampled up to 1,962/4,096 MiB whole-device
usage; sampling can miss peaks.

## Warp-reduction follow-up (2026-11-05)

Replaced the 256-thread CTA reductions with one 32-thread warp per query row
and `shfl.sync.bfly` reductions. NVIDIA's PTX ISA documents `shfl.sync` as
available from `sm_30`, so this is compatible with the project's `sm_50` PTX
floor ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)).
CUDA parity and fallback tests passed, but the packaged native-only SD1.5
512x512, Euler 20-step run took 265.5s and reported 82.319s in 1,280 fused
attention calls. The prior 256-thread fused run took 252.8s and reported
71.453s. The warp variant serializes eight times as many keys per lane and is
rejected; source dispatch remains on the 256-thread experiment, still disabled
by default. Next investigate a multi-query CTA tile to reuse K/V while retaining
enough query parallelism.
