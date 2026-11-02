# Public tiny Llama reference logits

Repository: `hf-internal-testing/tiny-random-LlamaForCausalLM`

Revision: `9fb191250dd56d0ba7ec9785a025ed29c03d5998`

The model weights are not committed. Download the pinned files from Hugging Face to enable the optional public-model test. The checkpoint is randomly initialized and is not evidence of useful language generation.

| File | SHA-256 |
|---|---|
| config.json | `5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382` |
| model.safetensors | `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992` |
| tiny-candle-f32.bin | `9c94f848e6203198bd0d469cb529d4130aca4f8c7d53a44d8ceca917212cf2c4` |

`tiny-candle-f32.bin` contains 4 rows × 32000 vocabulary logits, little-endian Float32. Each row was independently computed by Rust Candle 0.9.1 for the next prefix of `[1,42,17,321]`, with CPU/F32 and KV caching disabled. Row argmax IDs: `[5971,9456,6745,16196]`.

Generator source and Cargo.lock are in `models/src/test/reference-generator/`. From repository root:

```sh
cargo run --release --locked --manifest-path models/src/test/reference-generator/Cargo.toml -- .superpowers/models/tiny-llama models/src/test/resources/llama/tiny-candle-f32.bin
```

Ordinary tests do not execute Rust or Python. The optional public checkpoint test verifies all three file hashes and compares all 128000 Scala logits at `atol=1e-5, rtol=1e-5`; observed maximum absolute error on the verified environment was `5.960464477539063e-8`.
