# Diffusion timestep conditioning

`DiffusionTimesteps.sinusoidal` maps rank-one CPU Float32 time values to (batch,width) constant features. It supports fractional/negative times, sine/cosine order flipping, frequency shift, phase scaling, configurable period and odd-width zero padding. Defaults follow the feature convention in [diffusers get_timestep_embedding](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py): sine first, frequency shift1, scale1 and max period10000. Model configurations must select their own required settings (often cosine first and shift0).

Frequencies are exp(-log(period)*i/(floor(width/2)-shift)); trigonometric calculation uses Double before Float32 output. This is a numerical reference, not a claim of bitwise equivalence to diffusers' Float32 intermediate operations. Width must be at least2 and the frequency denominator positive; period must be finite and at least1. Times and phase products must be finite. Output elements default to a16777216 budget and must fit Int indexing. Empty batches produce empty features without allocating a frequency table.

`DiffusionTimeEmbedding(linear1,linear2)` applies Linear→SiLU→Linear with derived parameter traversal and width validation. It accepts differentiable features, but the sinusoidal API deliberately treats scheduler times as constants. Additional conditioning projections, alternative activations/post-activations and trainable time gradients are not implemented.

Analytical tests cover fractional phases/frequencies, order/shift differences, odd padding, empty inputs and rejection bounds. Every learned MLP parameter passes central differences. The MLP is composed with DiffusionResnet2d and image loss sends finite nonzero gradients into its parameters.

Scheduler timestep selection, pretrained config/weight mapping, independent pretrained numerical fixtures, CUDA execution and full U-Net/VAE/NUI integration remain required.
