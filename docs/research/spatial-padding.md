# Spatial padding for NUI tiling

`Padding2d(left, right, top, bottom, horizontal, vertical)` supports nonnegative asymmetric extents and independent `PaddingMode.Zero` / `PaddingMode.Circular` axes. `SpatialPadding.forward` operates on CPU Float32 NCHW tensors; `Variable.pad2d` adds a first-order scatter-add pullback. Double accumulation preserves contributions from repeated wrapped pixels before rounding to Float32.

This implements the numerical composition in pinned NUI revision8823aa4, `python/sdfx/sampling_ext.py`'s `TilingPatch`: horizontally pad with circular/constant mode, vertically pad with circular/constant mode, then convolve with zero internal padding. It can be used as:

```scala
val padded = input.pad2d(Padding2d(1, 1, 1, 1,
  horizontal = PaddingMode.Circular, vertical = PaddingMode.Zero))
val output = padded.conv2d(weight) // internal convolution padding is zero
```

Positive channel/spatial dimensions and Int indexing are required; empty batches are valid. Circular extents may wrap at most once on either side. Negative padding/cropping, reflection, replication and nonzero constant values are not yet implemented. Noncontiguous inputs are consumed in logical order. Explicit padding allocates its output; GPU/device-resident padding remains open.

Four new tests cover asymmetric mixed-mode output, two-axis corner wrapping, invalid/extreme extents, empty batches, all four tiling choices composed with convolution, input finite differences, noGrad and repeated backward accumulation. The composition oracle independently indexes the original image and kernel rather than reusing the padding implementation.

This is the numerical primitive, not a completed NUI feature port. Installing/removing model patches, toggling application state and testing the resulting image/UI workflows remain required alongside the complete diffusion pipeline.
