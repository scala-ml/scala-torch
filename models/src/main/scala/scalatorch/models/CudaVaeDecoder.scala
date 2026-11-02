package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** FP32 decoder snapshot. Input is unscaled latents; output is raw RGB,
  * with no pipeline scaling or pixel clamp. The session must outlive the decoder.
  */
final class CudaVaeDecoder private (runtime: CudaDiffusionRuntime,model: VaeDecoder)
    extends AutoCloseable:
  def apply(input: CudaTensor): CudaTensor=runtime.decode(input,model)
  def close(): Unit=runtime.close()

object CudaVaeDecoder:
  /** offloadWeights keeps immutable CPU snapshots and scopes GPU weight storage
    * to each operation. Defaults to retaining all weights on the GPU.
    */
  def upload(session: CudaSession,model: VaeDecoder,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaVaeDecoder =
    new CudaVaeDecoder(CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights),model)
