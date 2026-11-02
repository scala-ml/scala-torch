package scalatorch.models

import scalatorch.Tensor

/** Actual decoder input, after diffusion scaling, retained for exact recovery. */
final class VaeDecodeFailure(val decoderInput: Tensor[Float],cause: Throwable)
  extends IllegalArgumentException("VAE decode failed",cause)
