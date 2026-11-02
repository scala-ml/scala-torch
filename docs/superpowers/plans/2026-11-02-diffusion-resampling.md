# U-Net skip connections and resampling

Continue inline under approved diffusion scope.

1. Add Variable.cat(inputs,dim) with shape/bounds checks, logical-view copying, empty handling and sliced pullbacks. Repeated/tied inputs must accumulate through separate graph edges.
2. Add DiffusionDownsample2d for3x3 stride2 convolution, supporting symmetric pad1 or explicit bottom/right pad1 before a pad0 convolution (VAE-style geometry).
3. Add DiffusionUpsample2d with explicit target height/width, Floor nearest resize and3x3 pad1 convolution. Explicit targets support odd-size U-Net skip connections.
4. Test concatenation order/views/tied gradients/empty/invalid shapes and analytical odd-image resampling with gradients; run autograd plus diffusion component suites, review and commit.
5. Follow by conditional U-Net graph/config/weight mapping and independent pretrained comparisons; full VAE/schedulers/CLIP/CUDA/NUI remain open.
