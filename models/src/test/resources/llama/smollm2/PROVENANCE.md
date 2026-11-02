# SmolLM2 trained-model reference

Model: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0. Download original `config.json`, `model.safetensors`, and `tokenizer.json` from this revision; no conversion required. Weights are not committed.

| Artifact | SHA-256 |
|---|---|
| config.json | `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a` |
| model.safetensors (269060552 bytes) | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| tokenizer.json | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| tokenizer_config.json (template provenance, not required by runtime) | `4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df` |

Architecture: 30 layers, hidden576, intermediate1536, Qheads9/KVheads3/headDim64, vocab49152, tied embeddings/head, theta100000, eps1e-5. Original BF16 storage is converted to F32 independently by both loaders. Parameter count134515008.

Reference generator: `models/src/test/reference-generator/src/bin/smollm.rs`, Candle0.9.1 CPU F32, tokenizers0.22.1, Cargo.lock committed. Cache enabled. No Python used.

```sh
cargo run --release --manifest-path models/src/test/reference-generator/Cargo.toml --bin smollm -- /path/to/model models/src/test/resources/llama/smollm2
```

Prompt: default SmolLM2 system message plus user `What is the capital of France?`, followed by assistant prefix, as the pinned chat template specifies. `ids.txt` holds the 37 prompt IDs then eight generated IDs (including EOS2). Greedy generation requests at most16 tokens, stops at EOS, and chooses the lowest ID on ties. `text.txt` is the continuation `The capital of France is Paris.` with special tokens skipped.

`logits-f32.bin` is 4×49152 little-endian F32 values: last-position full-vocabulary logits from prompt prefill, then three successive cached decode steps. Comparison tolerance fixed before execution: `abs(error) <= 2e-4 + 2e-4*abs(reference)`. Observed maximum absolute error: `4.3392181396484375e-5`; all196608 values pass. The complete eight-token greedy continuation matches exactly.

| Fixture | SHA-256 |
|---|---|
| ids.txt | `456ccb39f252eedc58f2286798457bba2c1bcd43fb736fbda554000c35c0f626` |
| logits-f32.bin | `3e72506c0e7d8bb67a2601a18931566a17c20ee1fa14979ccd16b9a447f4d5f4` |
| text.txt | `a1b7eb2ee7a6aded8dda4e6cf30826f5afffb28a5597ee9389e91eb326d4e319` |

This verifies one trained model, prompt and greedy path. It does not establish broad model quality, performance parity, stochastic generation or GPU model execution.
