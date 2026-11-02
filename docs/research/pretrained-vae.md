# Pretrained VAE interoperability

The CPU Float32 loader reads the original AutoencoderKL config and all 124
weights from `hf-internal-testing/tiny-sd-pipe`, revision
`39f7e1819d772fbb8dd7f2aa3da8e2fc5a9bd917`, subdirectory `vae`.
`PretrainedVaeSuite` enforces SHA-256 hashes for the config, checkpoint and
reference outputs. Weights are downloaded separately and are not published
inside the library.

The API returns raw encoder moments and unscaled decoder images. Posterior
sampling, log-variance clipping, latent scaling and an image generation pipeline
are not implemented by this loader. Frozen inference is the default.

## Independent reference and batch discrepancy

The fixtures use Candle 0.9.1 CPU Float32. Rust is exclusively an offline fixture
generator; neither Scala builds nor library users need Rust. The generator,
Cargo manifest and dependency lock are under `models/src/test/resources/tiny-vae`.
Copy Candle's `src/models/stable_diffusion/vae.rs` to the generator's
`src/reference_vae.rs`, append `expose-moments.rs`, and use `reference.rs` as
`src/main.rs`. The pristine VAE source SHA-256 is
`5683ea2bce8e54c3c4a5f435b423f4539f5f0565f36f42ed14cd2765d1b86abd`.
The appended method exposes the existing encoder/quant-convolution output
before constructing a probability distribution; it changes no arithmetic.
Run the generator with the checkpoint directory and output directory arguments.

The initial direct Candle batch-of-two encoder reference disagreed with Scala
by 0.1852392, exceeding the unchanged absolute tolerance of 0.00002. Candle's own
batched output differed from independently evaluated rows by 0.18523908.
Fresh single-row inputs ruled out slicing offsets: row 1 matched the batched
path exactly after the input convolution and both down blocks, then differed
by 0.6204808 after the middle block. The precise underlying Candle operator
defect has not been isolated. No Candle kernel or model arithmetic was patched.

The committed generator therefore evaluates each image from a fresh single-row
tensor and concatenates the results for both encoder and decoder. Scala still
evaluates the actual batch of two. The six comparisons cover rectangular,
square and odd image dimensions and two-image batches. Observed maximum
absolute errors were below 0.0000009 for encoder moments and 0.0000030 for
decoder images. A separate Scala regression compares batched execution against
independent rows. This is evidence for the tested tiny checkpoint and CPU
Float32 path, not for all AutoencoderKL variants or CUDA execution.

## Loader validation

Synthetic archives exercise trainability, analytical encoder/decoder outputs,
missing and unused parameters, shape mismatch, unsupported BOOL dtype and the
aggregate parameter budget. F16 and BF16 archives materialize Float32 parameters.
Legacy query/key/value/proj_attn names with singleton convolution dimensions
load explicitly. Disabling quant, post-quant and middle attention removes their
parameters; leaving those weights in the archive rejects.

The affected public CLIP/U-Net/VAE and diffusion component suite passes
**57 tests, 0 failures, 0 skips**. The loader suite accounts for10 tests. No shared
core or autograd kernels changed. Read-only review found no correctness blockers.
Complete image generation and the Python-free NUI migration remain open.
