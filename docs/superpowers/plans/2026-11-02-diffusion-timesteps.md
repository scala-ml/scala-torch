# Diffusion timestep conditioning

Continue inline under approved M5 diffusion plan.

1. Add deterministic sinusoidal timestep features for rank-one CPU Float32 times, including fractional/negative values, configurable width/order/frequency shift/phase scale/max period, odd-width trailing zero and explicit output-element budget.
2. Add a derived two-Linear SiLU timestep MLP with optional biases and validated widths. Features remain constants; learned projection parameters use existing autograd.
3. Test analytical frequencies/order/odd widths, malformed/nonfinite values/budgets, empty batches and all learned-parameter finite differences. Compose the MLP output into DiffusionResnet2d and verify gradient flow.
4. Run targeted conditioning/residual tests and ordinary model regression without downloading assets. Document precise scope; scheduler timestep selection, CUDA/pretrained mapping and whole diffusion/NUI remain open.
