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
  private val batchedSamplingSetting: Option[String]=sys.env.get("SCALATORCH_BATCHED_SAMPLING").map(_.trim)
  /** Samples the images of one batch together, one U-Net pass per step. On by default with Float16 weights only: Float32 attention
    * over a batch of more than two leaves its cuBLAS path and the batch gets slower than the images one by one (measured 57 s
    * against 27 s for four SD1.5 images). `SCALATORCH_BATCHED_SAMPLING=0` switches it off, `=1` uses it with Float32 as well. */
  def batchedSampling: Boolean=batchedGuidance && (batchedSamplingSetting match
    case Some("0") => false
    case Some("1") => true
    case _ => CudaDiffusionRuntime.fp16Default)
  private[models] def memoryFailure(error: Throwable): Boolean=
    val text=Iterator.iterate[Throwable](error)(_.getCause).takeWhile(_!=null).take(5).map(e => Option(e.getMessage).getOrElse("")).mkString(" ")
    text.contains("allocation budget exceeded") || text.toLowerCase(java.util.Locale.ROOT).contains("out of memory") || text.contains("CUDA_ERROR_OUT_OF_MEMORY")

  /** With offloadWeights, retain immutable CPU tensors and upload weights per
    * operation. Activations stay resident; transfers trade throughput for VRAM.
    */
  def upload(session: CudaSession,model: ConditionalUNet2d,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaUNet =
    val half=CudaDiffusionRuntime.fp16Default
    val offload=offloadWeights || CudaDiffusionRuntime.shouldOffload(session,model,half,"U-Net")
    new CudaUNet(session,CudaDiffusionRuntime.upload(session,model,cancelled,offload,half=half),model)
