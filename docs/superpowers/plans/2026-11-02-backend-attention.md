# Backend rotary and causal attention

Continue the approved GPU transformer roadmap inline.

1. Extend FloatBackend with split-half RoPE and causal grouped-query attention, using (batch,heads,tokens,width), query offsets and explicit score-workspace limits matching existing autograd.
2. Implement CPU forward reference plus common checked shape/rotary metadata helpers. Verify hand calculations and existing autograd agreement.
3. CUDA RoPE applies Double host-precomputed sine/cosine coefficients to device activations. Only positional metadata crosses from CPU. CUDA attention computes Double dot products and stable causal softmax on device; approximate Float32 exponentials feed Double normalization/value accumulation. No activation downloads.
4. Bound Double scratch/coefficient memory through the session budget. Preserve empty shapes, GQA head mapping, nonzero query/position offsets and differing value widths. Validate failures before publication and clean temporary buffers.
5. Real-GPU numerical, masking, extreme-value, offset, empty and budget tests; full suites/review. Follow-on: prepared pretrained model execution on the backend, cache, performance/native libraries and complete NUI migration.
