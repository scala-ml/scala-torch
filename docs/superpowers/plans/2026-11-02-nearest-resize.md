# Nearest spatial resize

Continue inline under approved diffusion/NUI scope.

1. Add CPU Float32 NCHW Resize2d.nearest with explicit positive output height/width and NearestMode.Floor or Exact. Floor maps floor(dst*in/out); Exact maps floor((dst+1/2)*in/out). Use checked Int tensor bounds and Long rational index arithmetic to avoid floating boundary ambiguity. Empty batch valid.
2. Add Variable.resizeNearest2d and a Double scatter-add pullback. Preserve logical views and bit values; no resampling arithmetic in forward.
3. RED/GREEN tests distinguish modes for noninteger up/down ratios, verify multiple batches/channels, noncontiguous input, extreme/invalid shapes, explicit gradient multiplicity, finite differences and noGrad.
4. Run affected core/CPU/autograd/nn tests with serial core execution (known sbt parallel aggregate inconsistency), review, document and commit.
5. CUDA resize, bilinear/bicubic/area modes and scale-factor semantics remain required for broader NUI features. This does not complete token downsampling's patch/lifecycle migration or pretrained diffusion.
