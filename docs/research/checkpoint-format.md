# Training checkpoint format v1

API: `Checkpoint.save(model, optimizer, path, step)` and `Checkpoint.load(model, optimizer, path): Long` in `scalatorch.nn`.

This is a Scala training-state format, not a substitute for safetensors model interchange. It uses ordinary binary streams, no object deserialization or executable payloads. All multibyte numeric values use big-endian DataInput/DataOutput encoding.

Records, in order:

1. Eight ASCII bytes `STCKPT01` and a nonnegative signed Int64 caller-supplied training step.
2. Optimizer kind string (`sgd` or `adamw`), Int32 setting count, then setting-name string and Float64 value pairs in optimizer-defined order.
3. Int32 count of distinct model Parameter identities. Each parameter group follows in lexical order of its first alias; aliases within the group are sorted.
4. Per group: Int32 alias count and alias strings; trainable Boolean; Int32 rank and Int64 dimensions; exactly shape.numel raw Float32 words; optimized Boolean. Optimized groups then contain a has-moments Boolean. When present: positive Int64 per-parameter step, shape.numel Float64 first moments, then shape.numel Float64 second moments.
5. SHA-256 of every preceding payload byte, followed immediately by EOF.

Strings are Int32 byte length + strict UTF-8, capped at 65536 bytes. Counts, rank and dimensions must match the actual target model before tensor/moment allocation. Missing/extra aliases, changed tying, shapes, trainability, optimizer membership, kind or configuration reject. Floating weight bits, including signed zero and NaN payloads, are preserved. Optimizer moments must be finite and second moments nonnegative; absent moment slots remain absent.

## State and file lifecycle

Save writes a sibling temporary file and atomically replaces the destination only after successful encoding. There is no non-atomic fallback. Temporary files are cleaned on success or failure. This promises atomic replacement on supported filesystems, not power-loss durability.

Load stages values and optimizer slots, validates all metadata, checksum and exact EOF, then loads weights and restores optimizer state. Invalid input does not alter weights, gradients, or optimizer moments. On successful restore, gradients are cleared and previously saved graphs are invalidated. Optimizer parameter order can differ: slots are mapped by model names and identity groups.

Take checkpoints at optimizer boundaries. Gradient accumulation, random generator state, data-loader position and concurrent save/load/step are outside v1. A stochastic-training restart needs additional caller-managed state; exact resume has been verified for the deterministic tests, not claimed generally.

## Verified evidence

- AdamW uninterrupted 40-step regression training matches 17 steps + save/load + 23 steps exactly.
- SGD tied parameter restore, optimizer subsets, reordered AdamW parameter lists, and absent moment slots.
- Corrupted checksum, truncation, trailing bytes, invalid magic, oversized text metadata, configuration/alias mismatches reject without mutation; a subsequent training run proves rejected loads leave optimizer state pristine.
- Raw Float32 payload bits round-trip.
- Files with recomputed valid checksums but invalid Adam counters, nonfinite first moments or negative second moments reject without changing model, gradients or optimizer state.
- A failed atomic replacement preserves existing directory contents and removes the temporary file.
