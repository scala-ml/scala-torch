# Native Scala byte-level BPE

`ByteBpeTokenizer.read(path)` reads tokenizer JSON directly; `encode(text)` returns Long token IDs and `decode(ids,skipSpecialTokens=true)` returns text. No Python, native tokenizer library, BOS/EOS insertion, or implicit chat formatting is involved.

Supported subset: BPE with null dropout/unknown token/subword prefix/suffix, disabled byte fallback/ignore-merges/fuse-unknown, no normalization/postprocessing/truncation/padding, ByteLevel without prefix space and with regex splitting, optionally preceded by individual-number Digits. Literal added special tokens must have disabled word-boundary/whitespace/normalization flags and match vocabulary IDs. String and pair-array merge formats are accepted. Unsupported pipeline settings are rejected.

The implementation follows ranked adjacent merges with leftmost ties, UTF-8 byte mapping, Unicode-aware regex pretokenization, earliest/longest literal-special matching, and byte decoding with Rust-compatible replacement for malformed UTF-8. Published-tokenizer equivalence is tested against [Rust tokenizers 0.22.1](https://github.com/huggingface/tokenizers/tree/v0.22.1/tokenizers/src) using the pinned [SmolLM2 tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json); this is evidence for that pipeline, not universal Hugging Face tokenizer compatibility. Different Unicode-table versions may differ for newly assigned characters.

The published vocabulary omits 21 byte symbols. Its no-UNK BPE behavior drops those symbols, so arbitrary text is not guaranteed to roundtrip. Scala deliberately matches that reference behavior. Decoding happens across the entire ID sequence; decoding tokens individually can replace partial UTF-8 sequences prematurely.

Bounds: tokenizer file 32MiB; `fromJson` accepts up to 32Mi UTF-16 units instead (so non-ASCII strings can occupy more bytes); JSON nesting 16; vocabulary 256 to one million and merges up to one million; encode input up to 1Mi UTF-16 units, each BPE regex piece up to 16384 UTF-8 bytes, decode up to 1Mi IDs/32MiB output. Vocabulary IDs must be contiguous and unique. JSON duplicate keys, fractional IDs, malformed Unicode and unsupported semantics are rejected. JSON-size bounds do not equal peak heap bounds. The simple reference merge loop is quadratic in piece length; no production-throughput claim is made.

Synthetic tests cover exact merge IDs/overlap, byte roundtrip, special removal, absent bytes, invalid IDs/Unicode/limits, duplicate keys, malformed vocabulary/merges and unsupported config. The public gate checks exact IDs and both decode modes for 13 multilingual/control/whitespace/number/special-token inputs, plus seven decoder byte sequences. See [fixture provenance](../../models/src/test/resources/tokenizer/PROVENANCE.md).

Run the public gate with:

```sh
sbt -batch ';set models / Test / javaOptions += "-Dscalatorch.smollm.dir=/absolute/path/to/smollm2-135m";models / Test / testOnly scalatorch.models.TokenizerSuite'
```

Only `tokenizer.json` is required for this test. Chat formatting, tokenizer-config defaults, streaming text decoding, SentencePiece/Unigram and trained-model generation remain subsequent work.

Review found no blocking defects. Minor deferred schema validation: decoder flags (`add_prefix_space`, `trim_offsets`, `use_regex`) are accepted as metadata without checking their value types; they have no effect on ByteLevel decoding. The distinct file-byte/string-unit bounds above are deliberate documented limits of the current parser, not identical memory budgets.

Full verification at this checkpoint: **99 passed, 0 failed, 0 skipped**, with both public-reference gates and real CUDA tests enabled. Scala runtime dependencies still emit the existing Java 25 Unsafe deprecation warning.
