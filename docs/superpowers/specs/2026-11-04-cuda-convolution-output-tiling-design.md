# CUDA convolution output tiling

## Intent

Improve the FP32 CUDA inference path that NUI uses on low VRAM systems. Current
`conv2dFast` lowers each convolution to im2col plus a 16x16 output GEMM. The
kernel uses one accumulator per thread and covers 16 K values per shared-memory
tile. The full SD1.5 NUI path at 512x512 took 147 seconds for four Heun steps
with CUDA weight offload, so convolution throughput is a useful optimization
target.

## Goals

- Add a separate output-tiled FP32 GEMM kernel for convolution with a 16x32
  output tile and 32-wide K tile.
- Reuse loaded A/B tiles across more output values, preserve the current generic
  `matmul` kernel and output semantics, and handle incomplete M/N/K tiles.
- Verify CUDA output against the CPU convolution reference and measure a full
  NUI checkpoint run before/after.

## Non-goals

- Add Float16 tensor/storage semantics or claim Tensor Core acceleration.
- Change model precision, offload policy, generic matrix multiplication, or
  NUI's backend API.
- Promise a speedup on every CUDA device; measurements decide whether to keep
  the dispatch.

## Approaches

1. Keep the 16x16 kernel and tune instruction scheduling. This has the smallest
   risk but does not increase output reuse or reduce grid size.
2. Add an FP32 16x32 output / 32 K tiled kernel and use it only for
   `conv2dFast`. This retains FMA arithmetic and float storage while reducing
   tile count and increasing work per block. Chosen for the first optimization
   because it is isolated and works on the existing SM 5.0 baseline.
3. Introduce FP16 storage and Tensor Core convolution. This could give much
   higher throughput but changes the tensor, loader, FFM and all diffusion
   operators; it remains a separate project after this measured kernel work.

## Constraints

- Keep the kernel compatible with the existing PTX `sm_50` target and Java 25
  FFM driver-only runtime.
- Preserve float32 input/output storage and finite values within the current
  fast-convolution tolerance.
- Keep shared memory use within CUDA limits and honor `CudaSession.maxBytes`.
- Do not alter the user's existing branches or discard dirty work.

## Verification

- A CUDA convolution test checks grouped, padded and non-divisible M/N/K tails
  against `Conv2dOps.forward` and asserts the new kernel was dispatched.
- Existing CUDA convolution tests remain green.
- Run the same full SD1.5 512x512 four-step NUI EngineClient generation before
  and after with CUDA U-Net/VAE and weight offload; retain the new kernel only
  if it completes and does not materially regress runtime or output validity.
