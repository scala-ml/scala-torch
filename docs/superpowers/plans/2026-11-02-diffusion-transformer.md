# GEGLU and diffusion Transformer

Continue inline under approved M5.

1. Add differentiable Variable.narrow(dim,start,length), preserving Tensor views and scattering gradients into logical input coordinates. Validate bounds without overflow; empty selections yield zero gradients.
2. Compose DiffusionFeedForward: Linear to concatenated value/gate, split, value*GELU(gate), output Linear. Accept token tensors with arbitrary leading dimensions and preserve them.
3. Compose a standard pre-norm DiffusionTransformerBlock with self-attention residual, cross-attention residual and GEGLU residual. Validate all token widths and self-attention context widths at construction.
4. Test selection views/overlap/empty/bounds, analytic gate ordering, all-parameter Transformer finite differences and input/context gradients, zero residual identity and malformed widths.
5. Run affected autograd/nn/model tests with public/CUDA gates, review and commit. Masks, dropout/adaptive variants, spatial wrapper, pretrained mapping, CUDA and full NUI remain required.
