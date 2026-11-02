# U-Net skip connections and spatial resampling

`Variable.cat(inputs,dim)` concatenates logical Tensor values along an explicit axis. Non-concatenated dimensions must match. Axis sums and output allocation are bounded before copying; empty inputs are supported. Pullbacks slice the upstream gradient, so repeated input identities accumulate through separate graph edges.

`DiffusionDownsample2d` wraps a3x3 stride-two convolution. Symmetric padding1 samples ceil-sized odd-image outputs. With convolution padding0 it first adds one zero row/column at bottom/right, providing floor-sized VAE-style downsampling.

`DiffusionUpsample2d` performs Floor nearest resizing to an explicit positive target size, then a3x3 stride-one/pad-one convolution. Explicit targets allow up-path outputs to match odd-size skip features.

Four new tests cover strided concatenation order, repeated-input gradients, unequal channel pullbacks, empty/invalid concatenations, analytical3x5 downsampling for both padding choices, explicit3x5 upsampling and exact input gradient multiplicities. Convolution weight gradients remain connected.

These primitives prepare down/mid/up U-Net graph assembly; they are not yet a pretrained U-Net. Config/weight mapping, independent model comparisons, CUDA versions, scheduler/CLIP/VAE completion and all NUI migration features remain open.
