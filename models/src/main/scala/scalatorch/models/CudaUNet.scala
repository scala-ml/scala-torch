package scalatorch.models

import scalatorch.Tensor
import scalatorch.ffm.{CudaSession,CudaTensor}

/** FP32 U-Net snapshot. Timesteps are CPU constants; learned/image
  * operations remain on CUDA. Caller owns inputs and independently owned outputs.
  */
final class CudaUNet private (runtime: CudaDiffusionRuntime,model: ConditionalUNet2d) extends AutoCloseable:
  def apply(sample: CudaTensor,timesteps: Tensor[Float],context: CudaTensor): CudaTensor =
    runtime.unet(sample,model,timesteps,context)
  def close(): Unit=runtime.close()

object CudaUNet:
  /** With offloadWeights, retain immutable CPU tensors and upload weights per
    * operation. Activations stay resident; transfers trade throughput for VRAM.
    */
  def upload(session: CudaSession,model: ConditionalUNet2d,cancelled: () => Boolean=() => false,offloadWeights: Boolean=false): CudaUNet =
    new CudaUNet(CudaDiffusionRuntime.upload(session,model,cancelled,offloadWeights),model)
