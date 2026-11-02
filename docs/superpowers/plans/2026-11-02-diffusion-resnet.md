# Diffusion residual block

Execute inline under approved M5 scope.

1. Add models.DiffusionResnet2d composed from norm1, conv1, optional time projection, norm2, conv2 and optional shortcut. Implement default additive timestep conditioning with SiLU activation and output scaling. Use parameter names matching the conceptual diffusers fields (explicit model loading remains separate).
2. Enforce shape-preserving 3x3 convolutions, 1x1 shortcut, channel consistency, valid scale, rank-four input, exact timestep batch/width and matching conditioning presence. Reject incompatible blocks before executing.
3. Test zero residual/shortcut analytical behavior, timestep-conditioned all-parameter/input/embedding finite differences, derived parameter naming and invalid construction/inputs. Watch absent API fail first.
4. Run model suite (public/CUDA gates enabled), review and document. Nonzero dropout, scale-shift conditioning, integrated resampling, pretrained block fixtures, CUDA execution and full U-Net/VAE/NUI remain required follow-ons.
