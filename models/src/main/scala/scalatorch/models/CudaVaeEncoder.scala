package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** FP32 encoder snapshot returning raw mean/log-variance moments.
  * Sampling and diffusion scaling belong to the caller. Session must outlive encoder.
  */
final class CudaVaeEncoder private (runtime: CudaDiffusionRuntime,model: VaeEncoder)
    extends AutoCloseable:
  def apply(input: CudaTensor): CudaTensor=runtime.encode(input,model)
  def close(): Unit=runtime.close()

object CudaVaeEncoder:
  def upload(session: CudaSession,model: VaeEncoder,cancelled: () => Boolean=() => false,
      offloadWeights: Boolean=false): CudaVaeEncoder =
    new CudaVaeEncoder(CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights),model)
