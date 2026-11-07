# Native TAESD preview

Implement NUI's optional TAESD preview with the pinned `madebyollin/taesd` and `madebyollin/taesdxl` safetensors models. Keep neural preview distinct from the existing approximate projection and never require Python at runtime.

1. Record the fixed architecture, safetensors keys, latent scaling and output conversion from the pinned models and the locally installed Diffusers reference. Generate deterministic independent SD1 and SDXL decoder fixtures, including extreme latent values.
2. Implement bounded, shape-checked TinyAutoencoder decoder components in scala-torch with numeric reference tests. Preserve cancellation and use existing convolution/resize primitives; do not allocate unbounded intermediate feature maps.
3. Add explicit local model discovery for both model families, lazy loading and fallback behavior matching NUI when weights are unavailable; cache only validated weights, isolate preview failures from generation, and preserve interruption. Surface which preview method was actually used.
4. Validate decode parity and worker preview events with SD1 and SDXL; run CUDA/4 GB and NUI paths with the real trained weights. Test absent and malformed weights, cancellation, and the existing approximate preview.
5. Document model placement, source/license and limitations; rebuild and validate the native-only package when the feature is complete.
