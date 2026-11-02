# Published conditional U-Net loading

Continue inline under the approved diffusion/NUI goal.

1. Parse bounded, duplicate-free Diffusers UNet2DConditionModel config. Support the existing additive SiLU/GEGLU, convolutional spatial projection graph; reject unknown or unsupported semantic options explicitly. Legacy attention_head_dim denotes the head count. Bound dimensions and construction depth.
2. Build the graph from exact Diffusers safetensors names, validate shapes/dtypes and total parameter budget, reject missing/unused weights, freeze by default, expose opt-in training. Use the original diffusion_pytorch_model.safetensors filename.
3. Test malformed configs and archives plus pinned hf-internal-testing/tiny-sd-pipe revision 39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917. Compare a deterministic forward against independent Rust Candle 0.9.1; record checkpoint and fixture hashes. No Python model execution.
4. Run affected diffusion tests, independent review, documentation, commit. Loading this U-Net does not complete image generation or NUI migration. CLIP, VAE, scheduler and integrated image generation follow.

Review focus: unrecognized semantics, norm epsilon conventions, reversed skip channel mapping, resource limits before weight allocation, frozen inference, unused weights, and independent reference provenance.
