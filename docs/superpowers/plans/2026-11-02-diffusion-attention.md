# Noncausal attention and diffusion projections

Continue inline under approved M5.

1. Factor existing CPU AttentionOps causal implementation through a shared internal visibility switch. Add Variable.scaledDotProductAttention using every key while preserving causalAttention's offset semantics, workspace limit, GQA and pullbacks.
2. Add DiffusionAttention with Q/K/V/output Linear projections, explicit heads/score budget, batched self/cross contexts and strict widths. No implicit masking/dropout or normalization.
3. Test analytical unequal-length attention, GQA gradients by finite differences, empty/workspace behavior; projected cross-attention parameter/input gradients and default self-attention.
4. Run affected autograd tests plus model/public/CUDA regression, review and commit. Backend-native dense attention, masks, transformer norms/feed-forward, pretrained diffusion and NUI remain required.
