# Byte-preserving embedding edit staging

1. Generate NUI reference source and edited safetensors with F16/BF16/F32/F64 embeddings and unrelated tensors/metadata. Add failing byte-for-byte and error/cancellation tests.
2. Implement exclusive, bounded staging: copy and hash the original; validate identity and tensor layout; patch only changed coordinates while preserving every other byte; delete only the newly created stage on failure.
3. Verify native results against the NUI reference and upstream safetensors, and test source-change/collision/overflow behavior. Keep editor publication and live activation explicitly pending.
