# Published U-Net loading and numerical reference

`PretrainedUNet.load(directory, trainable = false, maxParameters = 1000000000L)` loads a Diffusers `config.json` and `diffusion_pytorch_model.safetensors` directly into the existing CPU `ConditionalUNet2d`. Opt-in trainable parameters use the same autograd graph; the default freezes every parameter and does not track inference with constant inputs.

The bounded config parser rejects duplicate keys, unknown fields and unsupported semantic options. Supported configurations use additive timestep conditioning, SiLU residual blocks, convolutional spatial projections, self/cross attention and erf-GEGLU. Stage channel widths and down/up attention placement are configurable; layer counts, Transformer depth and legacy head count are scalar. `attention_head_dim` means the head count in this legacy Diffusers format. Residual GroupNorm uses config epsilon; spatial Transformer GroupNorm uses `1e-6`; token LayerNorm uses `1e-5`. Timestep dimensions/frequency ordering and middle residual output scaling are explicit.

The loader indexes the archive and checks the aggregate parameter budget before any payload allocation. Each read validates the expected shape and supported F32/F16/BF16 dtype, then materializes Float32 parameters. Missing names, duplicate use and unused tensors reject the load; a partially built graph is never returned. The archive is closed on success/failure. This is single-file loading, not indexed/sharded Diffusers loading. The parameter budget bounds weight elements, not peak activation memory.

## Pinned checkpoint

Repository: [hf-internal-testing/tiny-sd-pipe](https://huggingface.co/hf-internal-testing/tiny-sd-pipe/tree/39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917/unet), revision `39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`.

- `unet/config.json`: SHA256 `2f80e0a35261d11d52c976b8bf15ec45ab414f46c5506e7a1f0c48851382ef06`.
- `unet/diffusion_pytorch_model.safetensors`: 5,742,160 bytes, SHA256 `d122d9ced096e51609cd202cb97ff6f2614e0be5394f3b38f84c8c58df04e07c`.
- 304 parameter tensors, stage channels32/64, eight heads, context width32, groups32, two residual layers per down block.

The checked-in shape manifest is extracted from this checkpoint header and supports offline synthetic archive validation tests without downloading weights. Weights remain ignored under `.superpowers/models/tiny-sd/unet`.

## Independent reference and activation discrepancy

Reference code, dependency manifest/lock, patch and binary outputs are in `models/src/test/resources/tiny-sd`. It uses Rust Candle0.9.1 on CPU, without Python execution. An initial comparison against unmodified Candle failed the preselected `2e-5` maximum-absolute-error threshold: the3x5 case differed by `4.44948673248291e-5`.

Source inspection identified an activation mismatch: Candle's Stable Diffusion GEGLU calls `Tensor.gelu()` (tanh approximation), while [Diffusers0.25 GEGLU](https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/activations.py) calls `F.gelu` with its default erf semantics. The isolated reference crate has one explicit patch changing that call to `gelu_erf()`. No Scala production arithmetic or test tolerance was changed. This is agreement with **Candle0.9.1 plus the documented erf correction**, not unmodified Candle and not a Python/Diffusers runtime comparison.

Inputs: `(1,4,H,W)` sample entries `Float(sin(i*0.17))`; `(1,3,32)` context entries `Float(cos(i*0.13))`. Output files contain contiguous NCHW little-endian Float32 values.

| Shape | Timestep | Maximum absolute error | Output SHA256 |
|---|---:|---:|---|
|1x4x3x5|7.5|1.430511474609375e-6|945c4f0cb4751cba2f1e681103c4ed7326bc17351e16c9888eeb11869231d318|
|1x4x4x4|0|1.087784767150879e-6|41cb5f022aa87374607ec338a9906369145252f807ef4dd6cfc49b93c56e8270|
|1x4x4x6|19|2.3245811462402344e-6|bec95207b1bc79df2337864b8b6e3b2bc779449cf6706407c2bf563d193a94a2|

To regenerate, create an isolated Rust project using the checked-in Cargo.toml/Cargo.lock and reference.rs as src/main.rs. Copy pristine candle-transformers0.9.1 into its candle-transformers directory, then apply candle-erf.patch there with `patch -p1`. Run `cargo run --release --locked -- /absolute/path/to/unet /absolute/output/directory`.

To validate Scala with the downloaded checkpoint:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Test / javaOptions += "-Dscalatorch.unet.dir=/absolute/path/to/unet";models / Test / testOnly scalatorch.models.UNetConfigSuite scalatorch.models.PretrainedUNetSuite'
```

Without that property the two public-checkpoint tests explicitly skip; config and synthetic archive tests remain offline. The initial nine tests passed with the property enabled. Review then added enforced config/checkpoint/fixture SHA256 checks and a corruption regression. The synthetic tests check training opt-in, frozen F16/BF16 loading, exact output-bias mapping, missing/extra/misshaped/unsupported-dtype tensors and budget rejection.

This tiny public model verifies checkpoint interoperability, not image quality. CLIP text conditioning, VAE decode, schedulers, integrated image generation, CUDA image kernels and full NUI migration remain unfinished.

The original Candle attention.rs SHA256 is `a5c47be3b77dae5db2940c5e6dedb3688fc38bf64ef7e41ec0fa6820b0da46d3`; after the single-line patch it is `1b0340a7028920d68d34ffc2fbc614f2cba9877bd2a80d28bd08b888e10ba59e`. The shape-manifest SHA256 is `56d586abab1789e1e880d0f9b7677be5fb3ee062aee01d1d3382e58d7efebef9`.

A complete model regression with public single/sharded Llama, trained SmolLM2, U-Net and CUDA enabled passed101 tests with no failures or skips. Existing Java25 Unsafe deprecation warnings remain.

Final diffusion-component verification after the review fix:31 tests passed,0 failed,0 skipped. The review found no production correctness blockers; its important provenance gap was fixed with retained patch/lock/reproduction instructions, enforced input/output hashes and a corruption test observed failing before the hash guard and passing after it. No deferred review findings remain.
