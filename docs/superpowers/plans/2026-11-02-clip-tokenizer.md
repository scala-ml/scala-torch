# Native CLIP fast-profile byte BPE

Continue inline under approved diffusion/NUI goal after a2e1972. Existing byte BPE
uses different normalization/pre-tokenization and has no CLIP word-end suffix.
Do not silently reinterpret an existing tokenizer configuration.

1. Implement explicit `ClipFastTokenizer.load(vocabPath,mergesPath,maxLength)`
   for the Transformers4.56.2 CLIPConverter profile: NFC, whitespace collapse,
   lowercase, CLIP Unicode split, byte alphabet, ranked BPE with </w>, standard
   start/end/unknown tokens. Validate vocabulary, merges and resource bounds.
2. Provide unpadded encode and fixed-length batch with attention mask; truncate
   only when explicitly enabled and always retain the final EOS. Preserve special
   token matching and exact logical batch shape, including empty batches.
3. Compare multilingual/Unicode/special-token IDs against tokenizers0.22.1 with
   independently constructed canonical profile using pinned public vocabulary and
   merges. Rust generates fixtures only, not a library dependency. Test malformed
   inputs, padding/truncation and frozen public CLIP conditioning.
4. Run affected tokenizer/public diffusion regressions, review, document, commit.

This explicitly implements the fast profile, not the slow CLIPTokenizer ftfy
repair/HTML-cleaning path used by NUI. Full NUI normalization parity remains a
separate requirement. Direct file arguments avoid silently ignoring tokenizer
config options; loader does not claim automatic config-driven selection.

## Outcome

All four steps completed. Independent15-case exact-ID reference passes after
fixing the observed contextual-Greek-lowercase mismatch. Public CLIP conditioning
and malformed/batch tests pass. Affected diffusion72tests and existing LLM
tokenizer5tests pass with no failures/skips; read-only review found no blockers.
The user identified the current local NUI during this work; its refreshed audit
is docs/research/nui-current-migration.md. Slow normalization and long-prompt
semantics remain open for that application; no NUI code was changed.
