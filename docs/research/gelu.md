# GELU activation

`Variable.gelu()` uses the Gaussian-error form; `gelu(GeluApproximation.Tanh)` explicitly selects the tanh approximation. Both return Float32 values with first-order gradients and preserve logical input shape/views.

The default computes x * Phi(x) using Phi(x)=0.5*erfc(-x/sqrt(2)), avoiding cancellation from 1+erf in the negative tail. Its derivative is Phi(x)+x*phi(x). The tanh option uses the conventional coefficient0.044715 and its analytical derivative. Double intermediates precede Float32 rounding. Negative/positive infinity use limiting output/derivative values; NaNs propagate. A zero local derivative suppresses upstream nonfinites consistently with the existing SiLU activation.

The autograd module pins [Apache Commons Numbers Gamma1.3](https://commons.apache.org/proper/commons-numbers/userguide/index.html) for the complementary error function. This is a JVM numerical dependency; activation composition and autograd remain Scala, with no Python or native runtime dependency added.

Three tests cover independent Gaussian output/derivative constants, nonzero negative-tail output at -10, distinct approximation outputs, finite differences for both modes, noncontiguous/empty tensors, noGrad and nonfinite limits. GPU GELU, GEGLU/feed-forward composition and pretrained diffusion comparisons remain open.
