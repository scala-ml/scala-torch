# Native PyTorch textual-inversion reader

## Intent

Allow the NUI scala-torch worker to consume the common legacy `.pt` textual-
inversion embeddings it currently loads with `torch.load(weights_only=True)`.
This removes Python from one frequent model-asset path without changing the
NUI protocol or bringing Python/PyTorch into the JVM application.

## Approaches

1. **Launch Python/PyTorch for conversion.** This preserves broad PyTorch
   compatibility but retains the dependency this migration is removing.
2. **Use a general pickle library/unpickler.** This risks executing or
   constructing arbitrary classes and exceeds the narrow data NUI needs.
3. **Read the current ZIP-based tensor archive with an allowlisted pickle
   interpreter.** This supports the common tensor/container subset, rejects
   executable or unknown pickle globals, and keeps resource limits local. Use
   this approach.

## Scope

- Read current ZIP-based `torch.save` archives only. Reject the pre-ZIP legacy
  format and any archive/pickle feature outside the allowlist.
- Support contiguous CPU/GPU-saved Float, Half and BFloat16 tensors backed by
  ZIP storage records. Device tags are metadata; payloads are interpreted on
  CPU. Require tensor storage bounds and shape/stride metadata to be valid.
- Accept NUI textual-inversion payloads: a tensor; `emb_params`; legacy
  `string_to_param["*"]`; or the SDXL mapping with exactly `clip_l`/`clip_g`.
  NUI's scalar metadata fields `name`, `step`, `sd_checkpoint` and
  `sd_checkpoint_name` may accompany these payloads.
  For a single-tensor SD1 payload, a nested `string_to_param` may contain only
  the `*` entry. Reject unrelated tensor keys and all non-tensor objects except
  the primitive map/list/tuple/string/integer structures needed to represent
  those layouts.
- Keep archive, entry, pickle-opcode, tensor-count, element-count and allocation
  limits bounded. Reject duplicate ZIP paths, duplicate mapping keys, sparse or
  non-contiguous tensors, overlapping/out-of-range views and malformed data.
- Keep the implementation in scala-torch `io`/`models`; NUI continues to pass
  paths over JSON Lines and has no PyTorch/scala-torch compile dependency.
- Unsupported but syntactically valid `.pt` structures remain explicit worker
  warnings, matching the existing warning behavior; never fall back to Python.

## Non-goals

- General PyTorch checkpoint/model loading, arbitrary pickle compatibility,
  protocol-0 legacy files, Python class reconstruction, NumPy payloads, tensor
  subclasses, sparse/quantized tensors, storage views, editing/inspection/merge.
- Claiming Python removal: other NUI features and packaging remain open.

## Validation

- Synthetic `torch.save`-compatible ZIP fixtures cover protocol 2/4/5 where
  supported, tensor and mapping layouts, F32/F16/BF16, GPU-origin tags,
  deterministic rows, and strict limit/error paths.
- Adversarial pickle fixtures contain forbidden `GLOBAL`, `REDUCE`, persistent
  IDs, duplicate keys, malformed frames and oversized shapes; all are rejected
  without executing a callable or allocating from untrusted dimensions.
- Scala worker suites prove SD1/SDXL `.pt` vectors affect generation exactly as
  equivalent safetensors and that unsupported `.pt` files warn and do not
  poison the loaded checkpoint.
- NUI `Generator` integration exercises a `.pt` embedding through the child JVM,
  saves final PNG metadata, and confirms no Python environment is created.

