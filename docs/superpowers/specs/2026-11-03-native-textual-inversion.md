# Native textual inversion design

## Intent

Allow the NUI native worker to use requested safetensors textual-inversion
embeddings during SD1.x and SDXL generation without Python. NUI already sends
the catalogued `embedding_paths` on generation requests; the Scala worker owns
file loading, CLIP token registration and embedding-row extension.

## Behavior

- Load only embeddings whose basename token occurs in the positive or negative
  prompt, using NUI's word-boundary match. Cache each attempted token until the
  checkpoint is unloaded, matching NUI's persistent pipeline behavior. Missing
  or inactive files are not cached.
- For SD1.x, accept one F16/BF16/F32 safetensors tensor with shape `(width,)` or
  `(vectors,width)`. For SDXL, require exactly the `clip_l` and `clip_g` tensors,
  each with the matching encoder width and the same vector count.
- Register the basename and its `_1`, `_2`, ... continuation tokens in each
  applicable CLIP tokenizer. Encode one prompt occurrence as all rows in order.
- Extend CLIP token-embedding matrices immutably. Apply each file transactionally
  and retain the extended pipeline and attempted-token cache until unload. An
  invalid file cannot leave half of its CLIP encoders changed; earlier valid
  embeddings remain loaded. Unsupported or model-incompatible files produce
  request warnings; missing and inactive files are skipped without caching.
- Preserve existing NUI EngineClient and JSON Lines schemas. No scala-torch
  model/tensor class or dependency is added to NUI shared/UI modules or assembly.

## Limits

The first implementation increment handled safetensors only. The follow-up
`2026-11-03-native-pytorch-embedding-reader.md` adds the bounded current-ZIP
`.pt` tensor subset; arbitrary globals, old archive formats and other payloads
remain explicit worker warnings. Embedding editing/inspection/merge,
model-keyword integration and Python removal remain open.

## Validation

- Tokenizer tests cover single/multi-vector expansion, longest-token matching,
  ordinary BPE preservation and token collision rejection.
- Encoder tests cover immutable row append, width/value validation and exact
  appended vectors.
- Worker tests cover SD1.x and SDXL safetensors layouts, `.pt` warning,
  persistence across requests, SDXL unload reset and deterministic output.
- NUI `Generator` integration sends `embedding_paths`, produces deterministic
  output with the token, and confirms the Python-free engine home is still used.
