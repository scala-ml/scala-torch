# Llama model execution evidence

The `models` module assembles a standard bias-free Llama decoder using the Scala nn/autograd implementation: embedding, pre-attention RMSNorm, q/k/v projections, rotary positions, causal GQA, output projection/residual, post-attention RMSNorm, gated SiLU MLP/residual, final norm and language-model head.

`LlamaModel.init(config,seed)` produces a differentiable model using config.initializerRange; norm scales initialize to one. `LlamaModel.load(directory)` reads config.json and either a single model.safetensors or an indexed shard set directly, frozen by default. It checks every expected name, shape and floating dtype, rejects unexpected/missing tensors, shares tied embedding/head identity and permits an omitted tied head. If a tied head is redundantly stored it must agree with embedding values. `forward` returns differentiable `(batch,tokens,vocab)` logits; `logits` runs noGrad. Module derivation registers nested parameters.

## Supported configuration

Bounded duplicate-free JSON with exact integer dimensions; standard model_type=llama, SiLU MLP, full unscaled half-pair RoPE, no projection bias or attention dropout. Unsupported rope scaling/parameters, sliding windows, partial rotary dimensions, custom head_dim, projection bias, non-SiLU activation or pretraining_tp!=1 reject. Unknown metadata/config fields are otherwise ignored; this is not an interpreter for arbitrary future architecture extensions.

Layer/parameter/logit/score limits are configurable through LlamaLimits, and safetensors materialization limits apply separately. Defaults are at most 512 layers, 1 billion logical unique parameters, 16777216 logit elements and 16777216 attention-score elements. These bound specific allocations/counts, not total process memory. CPU Float32 execution is currently supported.

`generateIds(prompt,maxNewTokens)` uses a persistent CPU KV cache by default (`useCache=false` retains full-prefix recomputation), selects the lowest ID on equal logits and stops on configured EOS IDs. The complete requested length must fit the configured context. A [native byte-level BPE tokenizer](byte-bpe-tokenizer.md) is available separately. Explicit SmolLM2 chat formatting and [trained text generation](trained-text-generation.md) are implemented. [Seeded sampling](token-sampling.md) is also supported. [Indexed shard loading](sharded-safetensors.md) is supported; padding masks remain open.

## Public checkpoint verification

[HF tiny-random Llama](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/tree/9fb191250dd56d0ba7ec9785a025ed29c03d5998) has hidden width 16, 2 layers, 4 heads, vocabulary 32000 and 1032272 Float32 parameters. Revision and file hashes are pinned in [reference provenance](../../models/src/test/resources/llama/PROVENANCE.md). Model files remain outside Git.

An independent Rust Candle 0.9.1 program generated all 32000 vocabulary logits for each prefix of `[1,42,17,321]`. Scala matched all 128000 values with maximum absolute error `5.960464477539063e-8`, inside fixed tolerance `1e-5 + 1e-5*abs(reference)`. Greedy next ID is 16196 in both runtimes. The token-ID CLI produced `1,42,17,321,16196` without Python.

This verifies a published checkpoint's architecture/weight interoperability. Because the checkpoint is random, it does not prove meaningful language generation. The executed reference is Candle, not PyTorch; broader model and backend comparisons remain open.

## Reproduction

From repository root, download immutable artifacts (curl is a development convenience, not a runtime dependency):

```sh
mkdir -p .superpowers/models/tiny-llama
curl -fL https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/9fb191250dd56d0ba7ec9785a025ed29c03d5998/config.json -o .superpowers/models/tiny-llama/config.json
curl -fL https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/9fb191250dd56d0ba7ec9785a025ed29c03d5998/model.safetensors -o .superpowers/models/tiny-llama/model.safetensors
sbt 'models / Compile / runMain scalatorch.models.LlamaRun .superpowers/models/tiny-llama 1,42,17,321 1'
```

To run the actual model test, substitute your absolute directory:

```sh
sbt ';set models / Test / javaOptions += "-Dscalatorch.llama.dir=/ABS/PATH/tiny-llama";models / Test / testOnly scalatorch.models.*'
```

Without this property, the public model test reports a skip; synthetic model tests still run. The verified full-suite run enabled both this gate and the real CUDA gate, so neither was skipped.

## Remaining gates

Tokenizer/config variants, broader trained-model text generation and large-shard-model validation, broader persistent KV-cache model references, broader GQA model references, CUDA model execution/performance, diffusion and complete NUI Python removal remain required. Synthetic tests verify GQA composition and gradients, ties, initialization, EOS/tie behavior, empty inputs and loader rejection; they do not replace those public-model gates.
