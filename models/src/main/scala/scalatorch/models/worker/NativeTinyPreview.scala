package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.Tensor
import scalatorch.models.{TinyAutoencoderDecoder,VaeExecution}

/** Lazily loads explicitly configured TAESD weights; each backend owns its cache. */
private[worker] final class NativeTinyPreview(sd15: Option[Path],sdxl: Option[Path],execution: VaeExecution):
  private var cached=Map.empty[Boolean,(Path,Long,java.nio.file.attribute.FileTime,AnyRef,TinyAutoencoderDecoder)]
  def render(latent: Tensor[Float],xl: Boolean,cancelled: () => Unit): String =
    cancelled()
    val dir=(if xl then sdxl else sd15).getOrElse(
      throw IllegalArgumentException(if xl then "TAESDXL weights are not configured" else "TAESD weights are not configured"))
    val weights=dir.resolve("diffusion_pytorch_model.safetensors").toRealPath()
    val before=Files.readAttributes(weights,classOf[java.nio.file.attribute.BasicFileAttributes])
    val decoder=cached.get(xl) match
      case Some((path,size,modified,key,model)) if path==weights && size==before.size() &&
          modified==before.lastModifiedTime() && key==before.fileKey() => model
      case _ =>
        val model=TinyAutoencoderDecoder.load(weights.getParent)
        cancelled()
        val after=Files.readAttributes(weights,classOf[java.nio.file.attribute.BasicFileAttributes])
        require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
          before.fileKey()==after.fileKey(),"TAESD weights changed during loading")
        cached=cached.updated(xl,(weights,after.size(),after.lastModifiedTime(),after.fileKey(),model))
        model
    val decoded=execution match
      case cuda: VaeExecution.Cuda => decoder.decodeCuda(latent,cuda.deviceIndex,cuda.maxBytes,cancelled)
      case _ => decoder.decode(latent,cancelled)
    NativePreview.jpegImageBase64(NativePreview.decodedRgb(decoded))

private[worker] object NativeTinyPreview:
  def fromEnvironment(env: Map[String,String],execution: VaeExecution): NativeTinyPreview =
    def path(key: String)=env.get(key).filter(_.trim.nonEmpty).map(value=>Path.of(value.trim).toAbsolutePath.normalize())
    new NativeTinyPreview(path("SDFX_SCALATORCH_TAESD_DIR"),path("SDFX_SCALATORCH_TAESDXL_DIR"),execution)
