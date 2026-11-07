# CUDA Gaussian-error GEGLU

`CudaSession.geglu` splits the final axis into equal value and gate halves.
It computes Gaussian-error GELU of the gate in Double, rounds that GELU to
Float32, then performs the separate Float32 multiplication used by CPU
`DiffusionFeedForward`. It does not substitute tanh GELU or QuickGELU.

The independently implemented PTX uses the relation between erfc and the upper
incomplete gamma function at a=1/2, documented in
[DLMF 7.11.2](https://dlmf.nist.gov/7.11.E2). With z=(abs(x)/sqrt(2))^2,
it uses the positive-term series below z=1.5
([DLMF 8.7.1](https://dlmf.nist.gov/8.7.E1)), and a contracted continued fraction
above that boundary ([DLMF 8.9.2](https://dlmf.nist.gov/8.9.E2)).
The series uses 40 terms after its initial term; the continued fraction stops
at a relative update below 2e-15 or 128 iterations. It reuses the existing
Double exponential helper. No external implementation code or coefficient table
was copied. Negative tails use erfc directly, avoiding subtraction from one.
Beyond +/-15 the Float32 GELU result is x or signed negative zero respectively.

Actual CUDA tests compare 30,721 evenly spaced gate inputs over [-15,15],
plus signed zeros, subnormals, maximum finite values, infinities and NaN against
the existing CPU Commons Numbers Erfc reference. Nonzero finite results agree
within two Float32 ULPs; zeros and infinities preserve their expected bits.
Tests also cover noncontiguous host uploads, leading dimensions, empty batches,
invalid widths, closed handles and allocation-budget cleanup.

This is an inference kernel, without a CUDA backward operation. The tests are
numerical evidence for the tested domain, not an exhaustive proof over all
Float32 inputs or a performance benchmark.
