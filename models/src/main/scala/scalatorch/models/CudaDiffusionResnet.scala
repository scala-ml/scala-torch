package scalatorch.models

import scalatorch.ffm.{CudaSession,CudaTensor}

/** Inference-only resident snapshot; the CUDA session must outlive this object and its outputs. */
final class CudaDiffusionResnet private (runtime: CudaDiffusionRuntime,model: DiffusionResnet2d)
    extends AutoCloseable:
  def apply(input: CudaTensor,timeEmbedding: Option[CudaTensor]=None): CudaTensor =
    runtime.resnet(input,model,timeEmbedding)
  def close(): Unit=runtime.close()

object CudaDiffusionResnet:
  def upload(session: CudaSession,model: DiffusionResnet2d): CudaDiffusionResnet =
    new CudaDiffusionResnet(CudaDiffusionRuntime.upload(session,model),model)
