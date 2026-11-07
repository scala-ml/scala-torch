package scalatorch.models.worker

import java.awt.image.BufferedImage
import scalatorch.*
import scalatorch.models.{TorchCpuNormal,VaeExecution,VaeModel,VaePosterior}
import scalatorch.autograd.Variable

/** Decode diffusion-scaled latent values using the loaded checkpoint's VAE. */
private[worker] final class NativeFullVaeTools(vae: VaeModel,execution: VaeExecution,tiled: Boolean)
    extends NativeVaeToolsCodec:
  def encode(image: BufferedImage,cancelled: () => Unit): Tensor[Float] =
    encodeSeeded(image,0,cancelled)
  override def encodeSeeded(image: BufferedImage,seed: Long,cancelled: () => Unit): Tensor[Float] =
    val input=NativeVaePixels.encodeInput(image,cancelled)
    val check=()=>{cancelled();false}
    val moments=if tiled then execution.encodeTiled(vae.encoder,input,vae.config.sampleHeight,vae.config.sampleWidth,check)
      else execution.encode(vae.encoder,input,check)
    cancelled()
    val posterior=VaePosterior(Variable(moments,false))
    val shape=posterior.mean.data.shape
    val noise=Tensor.fromFloats(new TorchCpuNormal(seed).normal(shape.numel.toInt),shape)
    val latent=vae.toDiffusionLatents(posterior.sample(noise)).data
    require(latent.toFloatArray.forall(_.isFinite),"VAE produced nonfinite latent values")
    cancelled()
    latent
  def decode(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
    cancelled()
    val raw=Tensor.fromFloats(latent.toFloatArray.map(_/vae.config.scalingFactor.toFloat),latent.shape)
    decodeRaw(raw,cancelled)
  override def decodeRaw(raw: Tensor[Float],cancelled: () => Unit): BufferedImage =
    cancelled()
    val check=()=>{cancelled();false}
    val decoded=if tiled then execution.decodeTiled(vae.decoder,raw,vae.config.sampleHeight,vae.config.sampleWidth,check)
      else execution.decode(vae.decoder,raw,check)
    cancelled()
    NativePreview.decodedRgb(decoded)
