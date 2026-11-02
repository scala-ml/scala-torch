# CPU convolution evidence

The approved diffusion milestone requires convolution for U-Net and VAE. This adds a deterministic CPU reference, differentiable operation and trainable module; it does not yet execute diffusion models.

- `Conv2dConfig` carries height/width stride, symmetric zero padding, dilation and groups.
- `Conv2dOps.forward` takes Float32 NCHW inputs and OIHW weights and returns NCHW outputs. It performs cross-correlation (no kernel reversal), with grouped channel partitions and Double accumulation.
- `Variable.conv2d` supplies first-order input and weight pullbacks. Frozen parents do not allocate unused gradient buffers. Existing immutable snapshots, noGrad and stale-parameter checks apply.
- `nn.Conv2d` derives parameter traversal and adds optional channel bias through differentiable broadcasting. Callers provide weights; random initialization convenience is not yet exposed.

The shape/group/stride/dilation convention follows [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html). This implementation supports explicit symmetric numeric padding only, rank-four batched inputs and positive channel/spatial dimensions; an empty batch is valid. Bounds and output sizes are checked before allocation. Noncontiguous inputs and weights are read in logical order. Padding is structural: products outside the image are skipped.

Analytical tests verify orientation, non-unit stride/dilation and grouped padded noncontiguous inputs. Central differences verify all input/weight/bias derivatives for a two-batch, four-channel grouped convolution with unequal stride/dilation/padding. Empty-batch gradients, noGrad, parameter replacement and malformed geometry are tested. These are analytical/numerical tests, not an independent PyTorch binary comparison.

The direct CPU kernel avoids im2col activation expansion and establishes a numerical reference. It is not an optimized production convolution backend. CUDA convolution, normalization/upsampling, pretrained CLIP/U-Net/VAE, schedulers, image-level comparisons and every NUI migration feature remain required.

NUI compatibility follow-on: pinned `python/sdfx/sampling_ext.py:384` implements `TilingPatch` by applying circular or constant padding separately per axis before convolution. The current zero-padding kernel does not replace that feature. The [spatial padding primitive](spatial-padding.md) now supplies differentiable per-axis circular behavior with convolution-composition tests. NUI patch installation/removal, state toggling and image-level application tests remain required before claiming tiling compatibility.
