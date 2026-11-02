# SD1.5 native acceptance work (2026-11-03)

Source: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
Pinned revision:451f4fe16113bff5a5d2269ed5ad43b0592e9a14 (public SD1.5 mirror).
The model card identifies CreativeML OpenRAIL-M. This local test does not publish
or redistribute the weights. All model/config/tokenizer files are under ignored
.superpowers/models/sd15; source-metadata.json records the upstream manifest.

Downloaded original FP32 safetensors, verified against upstream SHA-256:
- text_encoder/model.safetensors,492265874 bytes:
  d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334
- unet/diffusion_pytorch_model.safetensors,3438167540 bytes:
  19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1
- vae/diffusion_pytorch_model.safetensors,334643276 bytes:
  a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c

Standard CLIP has a persistent text_model.embeddings.position_ids I64 buffer,
absent in the tiny checkpoint. Loader now accepts only shape(1,maxPositions)
and values0..maxPositions-1, matching implicit positions, without treating it as
a trainable parameter. Unknown extra tensors remain rejected. A new single and
sharded archive test failed before the fix and passed after; corrupt values and
incorrect shape are rejected. PretrainedClipSuite passed8, including existing
independent Candle outputs and dtype conversion tests.

The local .superpowers/run-sd15.ps1 harness launches Java with-Xmx12g, CUDA U-Net
and VAE operation-scoped weight offload,1GiB session budgets,134217728 score work
limit. CLIP/schedulers remain CPU. It explicitly writes Euler-compatible noise
settings from the original beta schedule; this is DPM++2M/Karras testing, not a
claim to implement the checkpoint's original PNDM scheduler. Prompt is a photo
of a red apple on a wooden table, seed42,CFG7.5. Protocol/stderr/output are under
.superpowers/sd15-<side>-<steps>. No Python process is launched by this harness.
64x64/two-step smoke completed in77.9136696s reported generation time. PNG:
.superpowers/sd15-64-2/native-b0278d12-d2f4-4923-b4ac-3722b8832b1c.png
Visually inspected: coarse green/yellow/black blobs, not recognizable prompt
content or quality acceptance. Full512x512 and adequate sampling remain untested.
Live jcmd sampling at53s showed the worker in Tensor.logicalArray/Layout.index
while preparing a convolution weight upload. This identifies a host-copy
optimization candidate; one stack sample is not a complete time profile.
## Contiguous host-copy optimization

Tensor.logicalArray uses System.arraycopy for contiguous layouts, including an
explicit nonzero storage offset; strided layouts retain existing indexing.
Fresh64x64/two-step run with otherwise identical Java/model/sampler settings
reported31.6210033s versus77.9136696s before (~2.46x for this smoke run, not a
controlled general benchmark). Both PNG files have SHA256
92a0f854009740b088f484a538e401661333aff5c466df60f7ba5a7d478ecd86.
Output remains the same coarse blobs. Core20, CUDA session/ops13, offloaded
model2 tests passed; additional extraction coverage includes offsets, raw NaN/
negative-zero bits and independent primitive arrays. Existing transpose/reverse/
broadcast/empty cases pass. Further normal-resolution verification is in progress.
## Live512px investigation

During the first512px/20-step run (worker33796), jcmd at86s sampled CUDA
attention in a Transformer. At131s it again sampled attention, now in the
negative CFG prediction (UNetExecution line61) in the up path. Thus computation
had advanced despite no completed-step event yet. GPU utilization was100% with
about791MiB total memory at the earlier observation. No timeout/restart was done.
The current kernel assigns an entire query row to one thread, including serial
key dot products, exponentials and weighted sums across all value channels.
At4096 keys the16MiB scratch cap admits512 query rows per slice; this limits
parallelism. Candidate follow-up: parallel score generation and per-output-channel
weighted sums while preserving per-dot/reduction arithmetic order and scratch cap.
This is a code/stack observation, not a completed benchmark or proven speedup.
## Parallel noncausal attention output channels

Noncausal attention now has a prepare kernel that computes scores, maxima,
exponentials and probabilities in the previous order, followed by one thread
per(query row,value channel) for the weighted sum. Each key reduction retains
serial ordering and Double multiply/add without fusion. Probability division is
performed once per score instead of once per score/output channel. The same
bounded scratch buffer is reused; no new activation allocation is introduced.
The original attention entry is retained unchanged for causal/cache execution
and as a serial reference with all keys visible. Existing running workers retain
compatible original entry behavior even if resources are rebuilt for tests.

Tests compare raw Float32 bits against that serial reference for multiple slice
sizes, batches, GQA, unequal token/value widths, NaN/Infinity and diffusion head
widths40/80/160. Performance of the new path has not yet been measured in isolation.
The original512px run remains active (session11471, actual Java PID33796); step2
and a preview were received. Short GPU regression tests ran concurrently, so its
wall time is no longer an uncontended baseline. Do not use that time as a clean
before/after benchmark. Final image correctness is still useful evidence.