# Selective SD1 checkpoint VAE loading

Goal: restore a checkpoint's VAE without allocating the checkpoint's U-Net and CLIP tensor arrays, preserving validation, cancellation and the existing all-tensor API.

Measured using the official SD1.5 pruned EMA-only `.ckpt` and the current reader:
248 VAE tensors decode to 334,615,452 bytes; 686 U-Net tensors to 3,438,083,856 bytes;
197 CLIP tensors to 492,242,536 bytes. These count decoded primitive arrays,
not peak heap or process memory. The other 14 state entries are auxiliary tensors.

1. Add a restricted state-dictionary reader that parses tensor descriptors before materializing values. Preserve storage dtype, shape, offset, strides and the original global, nesting, ZIP and aggregate-element limits for every tensor. Do not infer a tensor's name from pickle stack position: resolve names only after the dictionary is constructed, including batched SETITEMS and memo references.
2. Return metadata for all dictionary tensors and materialize only selected VAE tensors. Validate all ZIP entries and actual lengths, including unselected storages. Preserve repeated storage views, signed integer metadata and interruption checks. Keep the existing public reader behavior unchanged.
3. Add fixtures for shared storage, duplicate references, multi-batch dictionaries, unselected malformed tensors, integer tensors, cancellation, and allocation accounting using a large unselected storage. Run the existing restricted-pickle test suite.
4. Use all metadata for the same SD1 structure checks, then pass only `first_stage_model.` values into the VAE loader. Preserve before/after file identity checks and avoid publishing a replacement VAE on failure.
5. Compare real `.ckpt` reset images before and after, and sample heap/working set under identical settings. Verify the NUI CUDA service and the 4 GB GPU path; report memory as measurements, not guaranteed requirements. Rebuild the distribution only after verification.
