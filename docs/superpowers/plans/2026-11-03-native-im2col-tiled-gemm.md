# Native im2col tiled GEMM implementation plan

**Goal:** Replace the scalar CUDA inference GEMM/convolution route with tiled matrix multiplication and use it for native Stable Diffusion convolutions.

**Architecture:** Add a 16x16 shared-memory PTX GEMM kernel while retaining the existing scalar kernel as an explicit fallback. Add a CUDA im2col kernel that emits one batch/group's convolution patches in spatial chunks capped at 128 MiB, then use tiled GEMM to compute output-channel by spatial-position products into the correct NCHW output slice. Keep the public `CudaSession.conv2dFast` API and its exact `conv2d` reference path unchanged.

**Tech stack:** Scala 3, Java FFM CUDA Driver API, PTX ISA, existing `CudaSession` allocation and launch helpers. No CUDA Toolkit or cuBLAS dependency.

**Scope:** `ffm/src/main/resources/scalatorch/gemm.ptx`, `ffm/src/main/resources/scalatorch/transformer.ptx`, `ffm/src/main/scala/scalatorch/ffm/CudaSession.scala`, `models/src/main/scala/scalatorch/models/worker/NativeDiffusionBackend.scala`, plus NUI native-engine evidence notes.

**Constraints:** Preserve current `conv2d` exact path, grouped convolution semantics, padding/stride/dilation, batch NCHW layout, session ownership/budget accounting, Java 25 FFM runtime, and driver-only deployment. Do not add or run automated tests. Verify with compilation and real SD1.5 NUI image generation.

## Tasks

### 1. Tiled GEMM — complete

- [x] Add a fixed 16x16 output tile PTX kernel with 16-wide K tiles, shared-memory operand staging, masked edge loads/stores, and FP32 FMA accumulation.
- [x] Keep `gemm` as the scalar fallback and load both entry points from the same PTX module.
- [x] Route `CudaSession.matmul` through tiled GEMM; provide a private strided-output variant for convolution so grouped/batched outputs can land directly in NCHW storage.
- [x] Preserve checked dimension bounds and release output allocations on launch failure.
- [x] Compile `ffm` and the dependent worker distribution.

### 2. im2col convolution — complete for the implemented API; grouped/batched coverage remains open

- [x] Add an im2col PTX entry mapping each `(channel * kh * kw, output position)` item to the corresponding input value, or zero for padding.
- [x] Build one batch/group patch matrix at a time in chunks capped at 128 MiB; this cap was needed after a full VAE matrix exceeded the existing 1 GiB session budget.
- [x] Run the tiled GEMM for each batch/group, writing to the correct output channel slice with full-channel row stride.
- [x] Keep exact `conv2d` unchanged and route only `conv2dFast` through im2col + tiled GEMM.
- [x] Compile `ffm`, `models`, and worker distribution.

### 3. Runtime verification — complete for SD1.5 batch 1

- [x] Run a small CUDA-backed NUI generation to exercise PTX JIT, boundary tiles, convolution, and output layout.
- [x] Run local SD1.5 512x512/20-step generation with wall time, exit, PNG and sampled GPU/RSS observations.
- [x] Inspect the output and compare against the earlier 12m09s warp-convolution run; the direct worker completed in 422.64s, about 42% faster. A separate packaged NUI MCP run completed the same size/step count with no `SDFX_*` environment variables.
- [x] Record results and the attention-limit adjustment in NUI native-engine notes.

## Review focus

- Non-multiple-of-16 M/N/K dimensions must mask shared-memory loads and output stores without reading or writing out of bounds.
- Batch size greater than one must preserve NCHW channel planes.
- Grouped convolution must offset weights, input channels, and output channels independently.
- Padding, stride, and dilation must match `Conv2dOps.geometry`.
- Oversized scratch/output requests must fail through existing session budget checks and close partial allocations.
