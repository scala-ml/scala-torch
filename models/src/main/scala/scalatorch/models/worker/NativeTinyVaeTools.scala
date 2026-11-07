package scalatorch.models.worker

import java.awt.image.BufferedImage
import scalatorch.*
import scalatorch.models.{TinyAutoencoderDecoder,TinyAutoencoderEncoder,VaeExecution}

/** TAESD image conversion for VAE Tools; latent values use diffusion coordinates. */
private[worker] trait NativeVaeToolsCodec:
  def encode(image: BufferedImage,cancelled: () => Unit): Tensor[Float]
  def encodeSeeded(image: BufferedImage,seed: Long,cancelled: () => Unit): Tensor[Float] = encode(image,cancelled)
  def decode(latent: Tensor[Float],cancelled: () => Unit): BufferedImage
  def decodeRaw(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
    throw new NativeBackendError("unsupported","raw decoder input requires Full VAE")

private[worker] object NativeVaePixels:
  def encodeInput(image: BufferedImage,cancelled: () => Unit): Tensor[Float] =
    require(image!=null && image.getWidth>=8 && image.getHeight>=8 &&
      image.getWidth.toLong*image.getHeight<=16777216,"invalid VAE input image")
    val width=image.getWidth;val height=image.getHeight;val plane=width*height
    val pixels=new Array[Float](Math.multiplyExact(plane,3))
    for y<-0 until height do
      cancelled()
      for x<-0 until width do
        val rgb=image.getRGB(x,y)
        for c<-0 until 3 do pixels(c*plane+y*width+x)=((rgb>>(16-8*c))&255)/127.5f-1f
    Tensor.fromFloats(pixels,Shape.from(Seq(1,3,height,width)))

private[worker] final class NativeTinyVaeTools(encoder: TinyAutoencoderEncoder,
    decoder: TinyAutoencoderDecoder,execution: VaeExecution) extends NativeVaeToolsCodec:
  def encode(image: BufferedImage,cancelled: () => Unit): Tensor[Float] =
    val input=NativeVaePixels.encodeInput(image,cancelled)
    execution match
      case cuda: VaeExecution.Cuda => encoder.encodeCuda(input,cuda.deviceIndex,cuda.maxBytes,cancelled)
      case _ => encoder.encode(input,cancelled)

  def decode(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
    val output=execution match
      case cuda: VaeExecution.Cuda => decoder.decodeCuda(latent,cuda.deviceIndex,cuda.maxBytes,cancelled)
      case _ => decoder.decode(latent,cancelled)
    cancelled()
    NativePreview.decodedRgb(output)
