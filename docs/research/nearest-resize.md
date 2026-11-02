# Nearest spatial resize

`Resize2d.nearest(input, height, width, mode)` and `Variable.resizeNearest2d` resize CPU Float32 NCHW tensors to an explicit positive size. Modes are:

- `NearestMode.Floor` (default): source index floor(destination * inputSize / outputSize).
- `NearestMode.Exact`: source index floor((destination + 1/2) * inputSize / outputSize).

These are the floor/cell-center conventions in [PyTorch's upsampling source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSample.h). Index calculation uses exact Long integer ratios for bounded Int dimensions, rather than intermediate Float scales. No bitwise equivalence to every floating-point scale-factor implementation is claimed. A floating `scale_factor` API is not implemented.

Forward copies logical source values without interpolation arithmetic. Backward accumulates repeated contributions in Double; unsampled pixels receive zero gradients. Noncontiguous tensors, batches/channels and empty batches are supported. Positive channels/spatial dimensions and bounded output allocation are required.

Four new tests cover noninteger up/down ratios distinguishing the modes, strided inputs, plane isolation, invalid/oversized shapes, empty batches, explicit gradient multiplicities, all-input finite differences and noGrad. NUI's pinned `token_downsampling.py` selects `nearest-exact` with explicit target sizes; the numerical primitive supports that selection, while patch installation, attention masks and application lifecycle remain unported.

U-Net/VAE integration and resident CUDA resize remain open. Other NUI features also require bilinear/bicubic/area interpolation and their scale-factor semantics; this nearest-only primitive does not replace those.
