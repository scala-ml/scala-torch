# Spatial padding for diffusion tiling

Continue inline under approved M5/NUI scope. Previous milestone: CPU convolution committed and all244 tests passed.

1. Implement Float32 CPU NCHW spatial padding with explicit nonnegative left/right/top/bottom extents and independent horizontal/vertical Zero or Circular modes. Circular extents cannot exceed their input dimension, matching the pinned NUI operation's single-wrap contract. Validate dimensions/output size before allocation, including empty batches.
2. Expose Variable.pad2d with a scatter-add pullback using Double accumulation. No mutation of inputs or graph state.
3. Test analytical asymmetric/mixed-axis output, noncontiguous values, corner multiplicity, finite differences and convolution composition for all four tiling choices; malformed/extreme shapes reject.
4. Run targeted and affected regression suites, review and commit evidence. This supplies the numerical operation only; porting NUI patch installation/removal and application state remains required.
