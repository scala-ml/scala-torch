# Native CLIP fast-profile tokenizer

`ClipFastTokenizer.load(vocabPath,mergesPath,maxLength=77)` explicitly selects
the [Transformers4.56.2 CLIPConverter profile](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/convert_slow_tokenizer.py#L1033).
It reads bounded vocabulary/merge files, validates contiguous IDs and merge
references, and applies NFC, whitespace normalization, code-point lowercase,
CLIP Unicode splitting, byte mapping and ranked merges with the `</w>` suffix.
The standard start/end strings identify BOS/EOS; EOS also represents unknown
symbols. Embedded normalized special tokens are preserved.

`encode` returns unpadded IDs with optional BOS/EOS. `batch` produces fixed-width
ID and attention-mask tensors, pads with EOS and rejects overlong input unless
truncation is explicit. Truncation retains final EOS. Input, piece, vocabulary,
context and batch allocation bounds reject before unbounded output allocation.
The implementation is Scala/JVM, with no Rust or Python runtime dependency.

The public vocabulary and merges come from tiny-sd-pipe revision
`39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`. Test resources enforce SHA-256 hashes
for both assets and the independent reference IDs. `reference.rs`, Cargo.toml and
Cargo.lock reconstruct the canonical normalizer/pre-tokenizer/model/postprocessor
using tokenizers0.22.1. Run it with the public tokenizer directory and reference
JSON output path. Rust is exclusively offline fixture-generation tooling.

Fifteen exact-ID cases cover English, whitespace, contractions, single digits,
accent composition, Greek, Turkish, German, Japanese, emoji, HTML-like literal
text, special tokens and controls. The initial comparison exposed Java's
contextual final-sigma lowercasing: tokenizers lowercases each code point
individually. The implementation now does the same, and the unchanged Greek
reference passes. A public CLIP integration accepts two natural-language prompts
and produces distinct frozen token features.

This API deliberately does not select a profile from tokenizer_config.json.
It does not implement the slow CLIPTokenizer ftfy/HTML-cleaning path, arbitrary
added-token rules, decoder text cleanup, NUI's weighted long-prompt chunks or
textual-inversion expansion. Those remain required for the [current NUI target](nui-current-migration.md).
Fast-profile parity is not evidence of complete NUI prompt compatibility.

Affected public diffusion/tokenizer regression:72passed,0failed,0skipped.
Existing LLM tokenizer regression:5passed,0failed,0skipped, including pinned
SmolLM2. Read-only review found no correctness blockers.
