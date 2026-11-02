# Conditional U-Net graph

Continue inline under approved M5 diffusion/NUI scope.

1. Add typed down/up layer containers (conditioned residual plus optional spatial Transformer), a residual-attention-residual middle block and ConditionalUNet2d.
2. Connect sinusoidal features and timestep MLP; retain initial/residual/downsample skip tensors and consume them LIFO in up residual layers. Upsample to the next remaining skip's explicit spatial size, including odd dimensions.
3. Validate all channels/time/context widths and exact skip production/consumption at construction. Require paired down/up resolutions, sampling between stages only, shape-preserving input/output convolutions and a matching output norm.
4. Test a genuine two-resolution model with mixed attention stages, fractional/scalar/per-batch times, odd rectangular images, whole-graph directional derivatives covering all inputs/parameters, exact zero-output behavior and malformed topology rejection.
5. Run diffusion component suite, review and commit. This is graph assembly, not pretrained execution. Next gates are config/weight mapping and independent reference comparison; VAE/scheduler/CLIP/CUDA and complete NUI remain active.
