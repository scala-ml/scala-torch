package scalatorch.models

import scalatorch.Tensor
import scalatorch.ffm.{CudaSession,CudaTensor}
import scala.util.Using

/** FP32 U-Net snapshot. Timesteps are CPU constants; learned/image
  * operations remain on CUDA. Caller owns inputs and independently owned outputs.
  */
final class CudaUNet private (session: CudaSession,runtime: CudaDiffusionRuntime,model: ConditionalUNet2d) extends AutoCloseable:
  def apply(sample: CudaTensor,timesteps: Tensor[Float],context: CudaTensor,
      added: Option[DiffusionAddedConditioning]=None): CudaTensor =
    val shape=session.shape(sample)
    require(shape.rank==4,"U-Net input must be NCHW")
    val features=model.addedFeatures(shape.dims.head,added)
    Using.Manager { use =>
      val uploaded=features.map(x => use(session.upload(x)))
      applyPrepared(sample,timesteps,context,uploaded)
    }.get
  private[models] def applyPrepared(sample: CudaTensor,timesteps: Tensor[Float],context: CudaTensor,
      features: Option[CudaTensor],control: Option[CudaControlNetResiduals]=None): CudaTensor =
    runtime.unet(sample,model,timesteps,context,features,control)
  def close(): Unit=runtime.close()

object CudaUNet:
  /** Runs cond and uncond as one batch of two; `SCALATORCH_CUDA_BATCH_GUIDANCE=0` switches it off. */
  val batchedGuidance: Boolean=sys.env.get("SCALATORCH_CUDA_BATCH_GUIDANCE").forall(v => v.trim!="0")
  private[models] def memoryFailure(error: Throwable): Boolean=
    val text=Iterator.iterate[Throwable](error)(_.getCause).takeWhile(_!=null).take(5).map(e => Option(e.getMessage).getOrElse("")).mkString(" ")
    text.contains("allocation budget exceeded") || text.toLowerCase(java.util.Locale.ROOT).contains("out of memory") || text.contains("CUDA_ERROR_OUT_OF_MEMORY")

  /** With offloadWeights, retain immutable CPU tensors and upload weights per
    * operation. Activations stay resident; transfers trade throughput for VRAM.
    */
  def upload(session: CudaSession,model: ConditionalUNet2d,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaUNet =
    new CudaUNet(session,CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights,half=CudaDiffusionRuntime.fp16Default),model)
