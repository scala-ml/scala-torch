# Python-free VAE Tools

NUI's existing VAE Tools protocol includes encode, decode and roundtrip, exact and quantized latent representations, batching, metadata, encrypted image handling, cancellation, reports and multiple VAE types. Preserve these behaviors and do not advertise the entire feature on the strength of one operation.

1. Implement the trained TAESD/TAESDXL encoder in scala-torch using the existing tiny residual block, with CPU and bounded CUDA paths. Validate all encoder weight names/shapes and compare independent Diffusers outputs for both even and odd dimensions; verify cancellation, recovery and a 512px CUDA input.
2. Implement bounded latent interchange, preserving L bit-exact representation, RGBA/CMYK quantization and safetensors metadata. Compare round trips and malformed inputs against NUI fixtures, retaining password and plaintext-consent rules.
3. Implement native VAE Tools request handling and per-item publication, errors, interruption and report semantics. Add full/approximate/TAESD paths as each is actually implemented; keep unavailable operations explicit. Validate tiling and raw decoder-input coordinates.
4. Exercise the UI service and request transport on real SD1.5 and SDXL with 4 GB CUDA, including multi-input interruption and password handling; then validate an extracted Python-free package. Document all remaining gaps.

## Progress

- The TAESD encoder is implemented for CPU and bounded CUDA. Trained SD1 and
  SDXL models match independent Diffusers encodes and encode/decode round trips
  for 16x24 and 17x19 inputs. A 512x512 CUDA encode and cancellation/recovery
  passed on the 4 GB GPU; one encode took 0.94 s excluding weight loading.
- TAESD/TAESDXL now use explicit environment paths first, then a NUI-local
  cache, and download pinned official weights only with permission. Both real
  official files passed SHA-256 verification and encoder/decoder loading. NUI
  service tests verified disabled download, opt-in download and cache reuse.
- RGBA/CMYK raw four-byte pixel packing and unpacking for four and sixteen
  channels matches independent NUI Python fixtures, including clipping and grid
  layout. RGBA PNG file reading/writing now preserves even the RGB channels of
  fully transparent pixels; 4- and 16-channel files were checked in both
  directions against NUI's Python implementation. Input dimensions and PNG
  format are checked before decoding. CMYK TIFF now preserves four/16-channel
  bytes in both directions against Python NUI. NUI service tests saved and
  decoded CMYK TIFF using a small fixture and real SD1.5 at 512px on CUDA;
  CMYK JPEG remains open.
- L-format PNG and in-memory safetensors now preserve Float32 bits. Native and
  Python NUI files were read in both directions for four and sixteen channels.
  Wider input floating types are converted to Float32 by the native reader.
- The worker-side TAESD adapter now converts RGB pixels to model input and
  model output to RGB, using the selected CPU or bounded CUDA execution. Tests
  ran trained TAESD and TAESDXL weights on CUDA through image -> exact PNG ->
  image, including a 17x19 source and cancellation.
- An initial `vae_tools` worker route handles TAESD with L/RGBA PNG, including
  batching, per-item errors, a report and transactional publication. The route
  requires a loaded checkpoint and explicitly configured local tiny weights.
  CUDA processing and denied-publication cleanup are tested directly; the loaded
  model route and actual NUI transport still need an end-to-end test. CMYK JPEG
  and TAESD tiled modes explicitly fail as unsupported.
- Approx NN now supports local official A1111 tensor-only checkpoints using
  a bounded restricted archive reader and CPU eight-layer inference. Official
  SD1 and SDXL weights match independent Python Float32 outputs within 1e-4;
  the handler test checks the doubled-resolution result. NUI service on real
  SD1.5 at 512px completed Full encode plus Approx NN decode in 10.96 seconds;
  a small-model GUI check also passed. Optional downloads are restricted to
  the two official release files, size bounded and SHA-256 checked, staged
  before publication and do not overwrite existing files. Actual SD1/SDXL
  downloads and NUI service opt-in download passed.
- Approx cheap decoding now uses the CPU latent preview factors, with Full VAE
  encoding fallback. SD1/SDXL output pixels match independent NUI references;
  a worker handler test verifies the latent-sized PNG output. The NUI service
  exercised encode/decode with Full fallback on the small fixture and real SD1.5
  at 512px with CUDA, confirming the report and output dimensions.
- NUI's `VaeToolsService` now has an integration test using a separately
  launched native worker: it loads a small checkpoint, verifies CUDA VAE
  placement, performs a Full L encode/decode and confirms no Python runtime
  was created. The same service test also passed with the real SD 1.5 checkpoint,
  512x512 gradient, CUDA VAE weight offloading, 4 GiB GPU and mean RGB
  reconstruction error 3.17/255. It also passed with real SDXL, the same
  image size and GPU, mean error 4.62/255. The initial test-only JVM launch
  exhausted its default heap; both an explicit 20 GiB heap and the production
  launch's `-XX:MaxRAMPercentage=50.0` passed on this 64 GiB RAM machine.
  A JavaFX GUI check also opened the actual VAE Tools tab, ran a Full L
  encode/decode through the native worker and displayed its preview using
  real SD1.5 at 512px and CUDA. Packaged release remains unverified.
- Full VAE encoding now samples the posterior from the seeded PyTorch CPU
  normal stream and supports the existing tiled encoder. A small CUDA fixture
  checks seed reproducibility, scaling and a handler encode/decode roundtrip.
  Python CUDA uses a different random stream, which is reported to the caller;
  multi-tile UI integration still needs end-to-end verification. On a GTX 1650 SUPER
  with 4 GiB VRAM, the real SD 1.5 VAE processed a 512x512 image to a 64x64
  latent and back in 16.85 seconds excluding weight load, using a 1.5 GiB CUDA
  session limit and weight offloading. The 512px fixture does not force multiple
  tiles at the checkpoint's configured tile size. Its synthetic gradient
  reconstruction has mean absolute RGB error 3.23 on a 0..255 scale. A separate
  768x512 synthetic gradient exercised multiple overlapping tiles for both
  encode and decode: 34.22 seconds excluding weight loading, mean absolute
  RGB error 4.69 on a 0..255 scale, same GPU and CUDA session budget.
- Full VAE decoding uses the loaded checkpoint and supports tiled execution;
  a small CUDA fixture tested diffusion scaling, cancellation, and a handler
  request with tiling enabled. Real multi-tile SD1.5 execution is tested above;
  NUI multi-tile transport and SDXL multi-tile remain unverified. Full recovery now sends saved decoder-input Float32 bits
  directly to the decoder, without a scaling round trip; a handler test checks
  the exact bits including negative zero.
- Safetensors input now validates its tensor and provenance, and converts
  decoder-input coordinates using the recorded scale, shift or mean/std before
  TAESD decoding. A CUDA test encodes an image, saves the native safetensors and
  decodes that file through the VAE Tools handler. Mean/std and external NUI
  recovery files still need independent integration fixtures.
- Encoded safetensors now include NUI-compatible `sdfx_vae_tools` provenance:
  model and VAE identity, model family, scale and coordinate. Python NUI read
  the native tensor and metadata successfully. L/RGBA encoded PNG, decoded PNG
  and display previews now carry NUI-compatible `VAE Tools` metadata without
  re-encoding pixels. Pillow reads it on open, and the encoded Float32 latent
  still matches the safetensors bit-for-bit. TIFF metadata remains open.
- Native pixel permutation and password-marker verification now match Python
  fixtures for scramble versions 1/2/3, including Unicode passwords and RGBA
  pixels beneath zero alpha. The protocol accepts VAE Tools password envelopes
  with redacted request diagnostics. Bounded protected PNG reading/writing now
  checks markers and passwords and interoperates with Python in both directions.
  VAE Tools now accepts protected RGB/RGBA input, protects output and previews,
  requires plaintext consent for encode and avoids saving intermediate latents
  for encode+decode. Handler and NUI service tests cover consent, wrong
  passwords and protected RGBA roundtrip; the latter passed with real SD1.5,
  512px input and CUDA on 4 GiB VRAM. Legacy v1 protected grayscale latent PNG
  also restores exact Float32 bits; files without password verifiers produce an
  unverified warning. GUI tests on real SD1.5 verify consent rejection,
  automatic consent reset and protected output/preview files; a small-model
  GUI test also verifies displayed preview pixels and clearing on lock.
- CMYK JPEG100 remains unsupported with JDK-only codecs; the request fails
  explicitly while CMYK TIFF remains supported. `vae_tools` is now advertised
  by the native worker and a NUI service integration test verifies a
  Python-free Full VAE encode/decode request. Rebuild the Windows ZIP to include
  this capability advertisement. Remaining parity work includes any
  unverified provenance/decoder-input cases, complete progress semantics and
  broader GUI/package acceptance; the plugin is not yet at full Python parity.
