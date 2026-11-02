# Noncausal attention and diffusion projections

`Variable.scaledDotProductAttention(key, value, maxScoreElements)` performs CPU scaled dot-product attention across every key. It shares the existing stable Double score/softmax and analytical pullbacks with causalAttention; the visibility count differs. Layout is (batch,heads,tokens,width), with grouped query heads and possibly unequal query/key sequence lengths and value widths.

`models.DiffusionAttention(toQ,toK,toV,toOut,heads,maxScoreElements)` projects (batch,tokens,channels) inputs, splits heads, performs noncausal attention and projects back to the query channel width. Optional context supplies separate cross-attention tokens/channels; omitting context performs self-attention. Channel/head layout and batch/width are validated, and score budget is explicit. No normalization, residual, masking or dropout is implicit.

Analytical tests prove every key contributes to zero-query attention and causal calls still exclude future keys. Finite differences cover all Q/K/V values with grouped heads and different sequence/value widths. Projected cross-attention tests cover all model parameters, query/context inputs, budget rejection, default/explicit self-context agreement and shared-input gradient accumulation.

This is CPU autograd execution. Backend-resident noncausal attention, padding/additive masks, pretrained diffusion attention fixtures, transformer feed-forward/normalization, full U-Net/VAE and complete NUI migration remain required. Existing causal backend kernels are unchanged.
