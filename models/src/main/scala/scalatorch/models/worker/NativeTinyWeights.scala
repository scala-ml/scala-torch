package scalatorch.models.worker

import java.net.URI
import java.nio.file.{Files,Path}

/** Explicit TAESD override, then a NUI-local cache with opt-in pinned download. */
private[worker] object NativeTinyWeights:
  private val sd15Revision="614f76814bbe30edbe2e627ace1c2234c81a2c0e"
  private val sdxlRevision="b20258aaef75ef61e659c1e0f14f251cf0ad153e"
  private val sd15Hash="db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549"
  private val sdxlHash="ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f"
  def official(target: Path,xl: Boolean,cancelled: () => Unit): Unit =
    val name=if xl then "taesdxl" else "taesd"
    val revision=if xl then sdxlRevision else sd15Revision
    val hash=if xl then sdxlHash else sd15Hash
    val url=s"https://huggingface.co/madebyollin/$name/resolve/$revision/diffusion_pytorch_model.safetensors"
    NativeApproxDownload.fetch(target,hash,()=>{
      val connection=URI.create(url).toURL.openConnection()
      connection.setConnectTimeout(30000);connection.setReadTimeout(30000)
      connection.getInputStream
    },cancelled)
  def resolve(env: Map[String,String],cache: Path,allowDownload: Boolean,xl: Boolean,cancelled: () => Unit,
      download: (Path,Boolean,() => Unit) => Unit = official): Path =
    val key=if xl then "SDFX_SCALATORCH_TAESDXL_DIR" else "SDFX_SCALATORCH_TAESD_DIR"
    val explicit=env.get(key).filter(_.trim.nonEmpty).map(value=>Path.of(value.trim))
    val directory=explicit.getOrElse(cache.resolve(if xl then "taesdxl" else "taesd"))
    val weights=directory.resolve("diffusion_pytorch_model.safetensors")
    if explicit.isEmpty && !Files.exists(weights) && allowDownload then download(weights,xl,cancelled)
    if !Files.isRegularFile(weights) then
      throw new NativeBackendError("unsupported",s"TAESD weights unavailable; configure $key or allow optional downloads")
    directory
