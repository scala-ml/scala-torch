package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** Resident FP32 decoder snapshot. Input is unscaled latents; output is raw RGB,
  * with no pipeline scaling or pixel clamp. The session must outlive the decoder.
  */
final class CudaVaeDecoder private (runtime: CudaDiffusionRuntime,model: VaeDecoder)
    extends AutoCloseable:
  def apply(input: CudaTensor): CudaTensor=runtime.decode(input,model)
  def close(): Unit=runtime.close()

object CudaVaeDecoder:
  def upload(session: CudaSession,model: VaeDecoder): CudaVaeDecoder =
    new CudaVaeDecoder(CudaDiffusionRuntime.upload(session,model),model)
