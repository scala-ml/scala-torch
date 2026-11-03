# SD1.5 NUI prompt conditioning design

## Intent

Make scala-torch's native SD1.5 image path apply the prompt semantics NUI uses
for standard SD1.x CLIP conditioning. NUI currently parses attention weights,
supports `BREAK`, makes 75-content-token CLIP chunks, applies token weights,
and pads the shorter positive/negative embedding sequence with empty-prompt
embeddings. The native shared image pipeline currently bypasses these semantics
for its single-encoder path and encodes one fixed tokenizer batch instead.

## Goals

- Apply the existing `DiffusionPrompt` parser/chunker to the SD1.5 single-CLIP
  path without Python or extra runtime dependencies.
- Match NUI SD1.5 CLIP skip selection, final LayerNorm, emphasis mean correction,
  and positive/negative chunk padding.
- Make emphasis mean correction well-conditioned: when a global mean is within
  eight dtype epsilons of the tensor's mean absolute magnitude, retain the
  weighted tensor without rescaling. This avoids backend-dependent amplification
  of LayerNorm rounding noise; apply the same rule in NUI and scala-torch.
- Reuse common token chunking, emphasis and padding code from SDXL while retaining
  SDXL's dual encoders, pooled projection, chunk-count reconciliation and blank
  negative zero tensors.
- Preserve explicit `truncatePrompt` behavior as a single bounded chunk.
- Skip negative-prompt encoding entirely when classifier-free guidance is disabled.
- Verify embeddings against an independent NUI/PyTorch oracle for plain, weighted,
  chunked, negative-padding and clip-skip cases using a pinned tiny SD1.5 model.

## Non-goals

- SD2/OpenCLIP-specific layer selection.
- Slow-tokenizer ftfy/HTML cleanup, textual inversion, LoRA, aesthetic guidance,
  scheduled prompt edits, or composable AND semantics.
- Changing the NUI request protocol or removing the Python backend.

## Approaches

1. Keep SDXL's private helpers and add a separate single-encoder implementation.
   This limits immediate SDXL risk but duplicates chunking and weighting behavior.
2. Extract shared CLIP chunk encoding, emphasis and sequence padding, then build
   single-encoder conditioning on that shared code while preserving SDXL-specific
   dual-encoder and pooled behavior. **Chosen:** it makes later NUI prompt fixes
   apply consistently across the primary SD1.5 path and SDXL; fixture parity gates
   protect the established SDXL behavior.

## Constraints and acceptance

- Runtime remains Scala/JVM only and adds no dependency on PyTorch or Python.
- In NUI, backend selection/process startup stays in `engine`; shared protocol
  and UI modules do not acquire scala-torch tensor/model dependencies. The
  actual implementation stays in scala-torch's `models` worker classpath.
- Keep changes on `feat/tensor-foundation`; NUI-specific source remains in the
  NUI repository and its existing feature branch.
- Standard SD1.5 clip-skip 1 uses the final normalized CLIP output. For skip N>1,
  select the Nth hidden state from the end and apply final LayerNorm.
- Blank SD1.5 negative text is encoded like other text, then positive/negative
  chunk counts are padded with the empty prompt embedding, matching NUI.
- SDXL continues to return its current zero negative and pooled tensors for blank
  negative prompts.
- A fixture generated from the pinned tiny-model checkpoint and checked NUI
  `prompt.py` source must compare all output tensor shapes and values with a
  maximum absolute error below `3e-5`.
- NUI and scala-torch use the same dtype-relative near-zero mean rule for prompt
  emphasis; stable means continue to be preserved by the existing correction.
- No claim of complete NUI prompt parity is made while the listed non-goals remain.
