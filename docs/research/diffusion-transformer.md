# GEGLU and token diffusion Transformer

`Variable.narrow(dim,start,length)` selects a bounded logical axis range as a Tensor view. The pullback scatters into the original logical shape; overlapping branches accumulate through the existing graph machinery. Bounds are checked without start+length overflow, and empty selections produce zero gradients.

`DiffusionFeedForward(projection,output)` applies a combined Linear projection, splits its final feature axis into value then gate, multiplies value by Gaussian-error GELU(gate), and applies the output Linear. Leading dimensions are flattened/restored, with positive compatible projection widths enforced.

`DiffusionTransformerBlock` composes three pre-norm residual stages: LayerNorm/self-attention, LayerNorm/cross-attention, and LayerNorm/GEGLU. It validates common token widths, self-context projection width and feed-forward output width. Parameters remain ordinary nested derived Modules.

Tests cover strided/overlapping/empty selections and invalid bounds; an analytical GEGLU gate-order example; every input/context/parameter derivative of a complete small Transformer against central differences; and exact identity behavior/gradient with zeroed residual output branches.

This is a CPU, unmasked, zero-dropout token block with standard LayerNorm and GEGLU. Spatial normalization/projections around image tokenization, masks, adaptive conditioning variants, dropout, pretrained configuration/weight mapping, GPU execution and the complete diffusion/NUI migration remain required. The synthetic gradient tests do not establish pretrained image quality.
