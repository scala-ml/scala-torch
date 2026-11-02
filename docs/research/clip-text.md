# CLIP text conditioning

`ClipTextModel` returns per-token text features for diffusion conditioning. The graph combines learned token and absolute position embeddings, biased causal multihead attention, pre-norm residual MLP blocks and final LayerNorm. Inputs are Int64 `(batch,tokens)` IDs; outputs are differentiable Float32 `(batch,tokens,width)` Variables. It accepts shorter contexts and independent batched rows. Future tokens cannot influence earlier positions.

`skipLastLayers = 0` uses all encoder layers. A positive value omits that many layers from the end, then applies the same final LayerNorm. Skipping the entire stack returns normalized embeddings. The API names the exact count; callers translating another application's “clip skip” convention must map its index explicitly.

QuickGELU is now a stable differentiable activation (`Variable.quickGelu`), computing `x*sigmoid(1.702*x)` with Double intermediates. It shares the existing sigmoid-weighted implementation with SiLU (slope1); finite extremes, nonfinite limits and analytical gradients are covered. CLIP also supports Gaussian-error GELU and explicitly requested tanh GELU. The model budget checks hidden, MLP and attention sizes before projection work. No dropout or arbitrary padding-mask API is implemented.

`PretrainedClip.load(directory, trainable = false, ...)` reads `config.json` and single/indexed `model.safetensors` weights. Config parsing is bounded, duplicate-detecting and rejects unsupported/unknown semantics. Nonzero attention dropout, decoder/cross-attention variants and configurations requiring a padding mask are rejected. Projection/initializer metadata is validated but does not add a projection or pooled output to this CLIPTextModel.

The loader validates the aggregate parameter budget before payload materialization, then checks every exact weight name/shape and floating dtype. F32/F16/BF16 become Float32. Missing, unused and misshaped weights fail without publishing a model; archive handles close on all paths. Defaults freeze all parameters, while `trainable = true` uses the ordinary autograd graph. Synthetic archive tests exercise all three dtypes in both single-file and two-shard layouts, final-norm bias mapping and invalid archives.

## Pinned public checkpoint

Source: [hf-internal-testing/tiny-sd-pipe text_encoder](https://huggingface.co/hf-internal-testing/tiny-sd-pipe/tree/39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917/text_encoder), revision `39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`.

- `config.json`:590 bytes, SHA256 `aae70cf2c528c154e5cbd2eba1249e759b91bbce15a9be74698f4a6492ec4879`.
- `model.safetensors`:283212 bytes, SHA256 `57f8ef72a6c2d5abd52829b83ed38d217cce0d515021db79147baf60dd94175f`.
- Vocabulary1000, width32, MLP37, five encoder layers, four heads, context77, erf GELU, LayerNorm epsilon1e-5;84 parameter tensors.

This is a tiny public testing checkpoint, not evidence of useful semantic or image quality. Token IDs in the reference are deterministic and do not come from a prompt tokenizer.

## Independent reference

`models/src/test/resources/tiny-clip` retains the Rust executable source, configuration factory, Cargo manifest/lock, shape manifest and binary outputs. Computation uses unchanged Candle0.9.1 `stable_diffusion/clip.rs`; its private Config fields are made usable by appending the supplied factory inside a local copy of the module. The original source SHA256 is `8d04ed7208655ae87b488a721e4eac797a780ab62ab9f61c340cb05484930483`. No model arithmetic is patched, and no Python model runs.

Candle's diffusion CLIP mask fails to broadcast at batch2 (mask2x77x77 versus scores2x4x77x77). The reference therefore computes each batch row independently and concatenates results. Scala's actual batch2 output is checked against those independent rows; upstream batched Candle execution is not claimed. Earlier-layer output comes from Candle's intermediate-layer API followed by its learned final LayerNorm.

IDs are `((i*17+3)%997)` across all77 tokens per row. Reference outputs are contiguous little-endian Float32. The QuickGELU/tanh cases deliberately reuse the public weights with an alternate activation to verify the implementation; the published configuration itself uses erf GELU.

|Case|Maximum absolute error|Output SHA256|
|---|---:|---|
|erf GELU, batch1|1.3746321201324463e-6|4e8e48472fd91f41052539720af67b6f65ed3a692cb21a6e5b9c83d9252fdf3f|
|erf GELU, batch2|1.3746321201324463e-6|49d0501a5a8e3cb92ece701466df31ddad891de5b0b96fdeee8d0e3d265a6ef0|
|erf GELU, skip last layer|1.5497207641601562e-6|37eebdd92d23db4c60f03c94d5761a5edd7515ad8d78c985a01b932411a340c3|
|QuickGELU, batch1|1.3262033462524414e-6|929f0c363453b7c37652f14344716dd2ed254c0eb58ff464fbd748b20a4c26b0|
|tanh GELU, batch1|1.3262033462524414e-6|a440bff4dff75d019f2e2b7f4e22a886d2faffb7fc0a2ee2d06afc9650648d70|

All pass the preselected2e-5 threshold. Tests enforce hashes of config, checkpoint, shape manifest and output fixtures; a corrupted-output regression rejects altered bytes. The shape manifest is extracted from the pinned public safetensors header, not generated by the Scala loader.

To reproduce, create an isolated Rust project using Cargo.toml/Cargo.lock from the resource directory and reference.rs as src/main.rs. Copy pristine Candle0.9.1 stable_diffusion/clip.rs to src/reference_clip.rs, verify the hash above, then append config.rs separated by a newline. Run `cargo run --release --locked -- /absolute/path/to/text_encoder /absolute/output/directory`.

To run public Scala gates:

```sh
sbt --server -Dsbt.io.virtual=false -Dsbt.server.autostart=false -batch ';set models / Test / javaOptions ++= Seq("-Dscalatorch.clip.dir=/absolute/path/to/text_encoder", "-Dscalatorch.unet.dir=/absolute/path/to/unet");models / Test / testOnly scalatorch.models.ClipTextSuite scalatorch.models.PretrainedClipSuite'
```

Absent these properties, the public gates explicitly skip. Offline graph tests check causality, short-prefix equivalence, batch isolation, exact skip indexing, all-parameter finite gradients and three whole-graph directional derivatives. Public integration passes the frozen CLIP output into the frozen public U-Net and verifies finite, text-sensitive predictions.

Prompt tokenization/normalization/weighting, VAE, schedulers, complete image generation, resident CUDA diffusion and full NUI behavior remain unfinished. Maven Central0.1 publication is separately authorized for a coherent milestone; it does not imply completion of that wider goal.

Final verification after review: **312 passed,0 failed,0 skipped** across all modules (core19,CPU15,autograd70,tensorio17,nn43,models113,FFM35), with every public model and real-CUDA gate enabled. Review found an empty-batch positional-allocation budget bypass; its regression failed before adding the independent positional shape bound and passed afterward. Empty batches remain supported within that bound. No deferred findings remain. Existing Java25 Unsafe warnings persist.
