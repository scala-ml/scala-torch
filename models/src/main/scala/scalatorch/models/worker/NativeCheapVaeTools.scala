package scalatorch.models.worker

import java.awt.image.BufferedImage
import scalatorch.Tensor

/** Full VAE encoding paired with NUI's weight-free latent RGB approximation. */
private[worker] final class NativeCheapVaeTools(encoder: NativeVaeToolsCodec,sdxl: Boolean) extends NativeVaeToolsCodec:
  def encode(image: BufferedImage,cancelled: () => Unit): Tensor[Float] = encoder.encode(image,cancelled)
  override def encodeSeeded(image: BufferedImage,seed: Long,cancelled: () => Unit): Tensor[Float] =
    encoder.encodeSeeded(image,seed,cancelled)
  def decode(latent: Tensor[Float],cancelled: () => Unit): BufferedImage =
    cancelled()
    val image=NativePreview.rgb(latent,sdxl)
    cancelled()
    image
