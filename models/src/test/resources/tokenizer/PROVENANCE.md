# SmolLM2 tokenizer reference

Source: [HuggingFaceTB/SmolLM2-135M-Instruct tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json), Apache-2.0, pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

Tokenizer JSON SHA-256: `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

Reference is Rust `tokenizers = 0.22.1`, exact dependency graph in `models/src/test/tokenizer-reference/Cargo.lock`. No Python used. Generator calls `encode(text,false)` (no inserted special tokens), then `decode(ids,false/true)`.

Reproduce from repository root:

```sh
cargo run --release --manifest-path models/src/test/tokenizer-reference/Cargo.toml -- /path/to/tokenizer.json models/src/test/resources/tokenizer/smollm2-decode.tsv > models/src/test/resources/tokenizer/smollm2.tsv
```

`smollm2.tsv` has 13 rows: hex UTF-8 input, comma-separated IDs, hex decoded text retaining specials, hex decoded text skipping specials. SHA-256 `c6c3c9ecc92d1dda6e0d84460d163b96a536f17224e26a390fa56cb254916934`.

`smollm2-decode.tsv` has 7 rows: arbitrary comma-separated IDs, hex decoded text. It covers truncated sequences, surrogate/overlong/out-of-range encodings, stray continuation bytes, and a valid multibyte character. SHA-256 `40b413e6ac79cffb82bc337f0f2b5b021e69a887739d3517a21ee96777bc86c6`.

The published tokenizer lacks 21 byte-alphabet entries. With no unknown token configured, Rust BPE omits those symbols; some inputs cannot roundtrip losslessly. The fixture includes missing control bytes and U+40000 to verify that behavior rather than assuming lossless UTF-8 encoding.
