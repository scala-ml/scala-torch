package scalatorch.models

import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive,SafeTensorSet}
import scalatorch.nn.Parameter
import java.nio.file.{Files,Path}
import java.nio.file.attribute.BasicFileAttributes
import scala.util.Using
import scala.collection.mutable

/** Components loaded together before a worker publishes a replacement pipeline. */
private[models] final case class SingleFileSd1Model(
    clip: ClipTextModel,unet: ConditionalUNet2d,vae: VaeModel,linearTargets: Map[String,Parameter],
    family: String="sd1",clip2: Option[ClipTextModel]=None)

private[models] object SingleFileSd1Model:
  private[models] def isPyTorch(path: Path): Boolean =
    val name=path.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
    Vector(".ckpt",".pt",".pth",".bin").exists(name.endsWith)

  private def readPyTorch(path: Path,cancelled: () => Unit,vaeOnly: Boolean = false): mutable.Map[String,PyTorchTensorArchive.Value] =
    import PyTorchTensorArchive.*
    val limits=PyTorchArchiveLimits(maxArchiveBytes=8L*1024*1024*1024,maxEntries=5000,
      maxEntryBytes=256L*1024*1024,maxTotalUncompressedBytes=8L*1024*1024*1024,
      maxPickleBytes=16*1024*1024,maxPickleOps=4000000,maxTensors=4096,
      maxElements=64000000L,maxTotalTensorElements=1500000000L)
    val configured=PyTorchCompatibility.limits(limits)
    val value=(if vaeOnly then PyTorchTensorArchive.readStateDictSelected(path,configured,()=>{cancelled();false},
      _.startsWith("first_stage_model.")) else PyTorchTensorArchive.readStateDict(path,configured,()=>{cancelled();false})).value
    value match
      case DictValue(fields) if fields.contains("state_dict") => fields("state_dict") match
        case DictValue(state) => mutable.Map.from(state)
        case _ => throw IllegalArgumentException("SD1 state_dict must be a tensor dictionary")
      case DictValue(fields) => mutable.Map.from(fields)
      case _ => throw IllegalArgumentException("SD1 checkpoint must contain a tensor dictionary")

  private def take(values: mutable.Map[String,PyTorchTensorArchive.Value],prefix: String): Map[String,PyTorchTensorArchive.Value] =
    val names=values.keysIterator.filter(_.startsWith(prefix)).toVector
    names.map(name=>name->values.remove(name).get).toMap

  private def loadPyTorch(path: Path,maxScoreElements: Long,cancelled: () => Unit,
      maxVaeScoreElements: Long): SingleFileSd1Model =
    val values=readPyTorch(path,cancelled)
    val infos=SingleFileTensorView.pytorchInfos(values.toMap)
    if SingleFileSdxlConfig.isSdxl(infos) then
      val config=SingleFileSdxlConfig.unetConfigForTensors(infos)
      cancelled()
      val clip=PretrainedClip.loadSdxlView(SingleFileTensorView.pytorchSdxlClip(take(values,"conditioner.embedders.0.transformer.")),maxScoreElements)
      cancelled()
      val clip2=PretrainedClip.loadSdxlClip2View(SingleFileTensorView.pytorchSdxlClip2(take(values,"conditioner.embedders.1.model.")),maxScoreElements)
      cancelled()
      val unet=PretrainedUNet.loadSd1View(SingleFileTensorView.pytorchUnet(take(values,"model.diffusion_model.")),
        config,maxScoreElements,maxParameters=3000000000L)
      cancelled()
      val vae=PretrainedVae.loadSdxlView(SingleFileTensorView.pytorchVae(take(values,"first_stage_model.")),
        if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
      cancelled()
      SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++clip2.linearTargets++unet.linearTargets,"sdxl",Some(clip2.model))
    else if SingleFileSd2Config.isSd2(infos) then
      val config=SingleFileSd2Config.unetConfigForTensors(infos)
      cancelled()
      val clip=PretrainedClip.loadSd2View(SingleFileTensorView.pytorchSd2Clip(take(values,"cond_stage_model.model.")),maxScoreElements)
      cancelled()
      val unet=PretrainedUNet.loadSd1View(SingleFileTensorView.pytorchUnet(take(values,"model.diffusion_model.")),config,maxScoreElements)
      cancelled()
      val vae=PretrainedVae.loadSd1View(SingleFileTensorView.pytorchVae(take(values,"first_stage_model.")),
        if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
      cancelled()
      SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++unet.linearTargets,"sd2")
    else
      val config=SingleFileSd1Config.unetConfigForTensors(infos)
      cancelled()
      val clip=PretrainedClip.loadSd1View(SingleFileTensorView.pytorchClip(take(values,"cond_stage_model.")),maxScoreElements)
      cancelled()
      val unet=PretrainedUNet.loadSd1View(SingleFileTensorView.pytorchUnet(take(values,"model.diffusion_model.")),config,maxScoreElements)
      cancelled()
      val vae=PretrainedVae.loadSd1View(SingleFileTensorView.pytorchVae(take(values,"first_stage_model.")),
        if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
      cancelled()
      SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++unet.linearTargets)

  def loadVae(path: Path,maxScoreElements: Long,cancelled: () => Unit): VaeModel =
    val real=path.toRealPath()
    val before=Files.readAttributes(real,classOf[BasicFileAttributes])
    val values=readPyTorch(real,cancelled,vaeOnly=true)
    val infos=SingleFileTensorView.pytorchInfos(values.toMap)
    val sdxl=SingleFileSdxlConfig.isSdxl(infos)
    if sdxl then SingleFileSdxlConfig.inspectTensors(infos)
    else if SingleFileSd2Config.isSd2(infos) then SingleFileSd2Config.inspectTensors(infos)
    else SingleFileSd1Config.inspectTensors(infos)
    val vaeValues=take(values,"first_stage_model.")
    // Only the VAE is being restored. Release the decoded U-Net and CLIP arrays
    // before constructing it, so they can be reclaimed during allocation.
    values.clear()
    val vae=if sdxl then PretrainedVae.loadSdxlView(SingleFileTensorView.pytorchVae(vaeValues),maxScoreElements)
      else PretrainedVae.loadSd1View(SingleFileTensorView.pytorchVae(vaeValues),maxScoreElements)
    cancelled()
    val after=Files.readAttributes(real,classOf[BasicFileAttributes])
    require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
      before.fileKey()==after.fileKey(),"checkpoint changed during VAE loading")
    vae

  def load(path: Path,maxScoreElements: Long,cancelled: () => Unit = () => (),
      maxVaeScoreElements: Long = -1L): SingleFileSd1Model =
    val real=path.toRealPath()
    val before=Files.readAttributes(real,classOf[BasicFileAttributes])
    val model=if isPyTorch(real) then loadPyTorch(real,maxScoreElements,cancelled,maxVaeScoreElements) else
      Using.resource(SafeTensorSet.openFile(real)) { archive =>
        if SingleFileSdxlConfig.isSdxl(archive.tensors) then
          val prof=sys.env.get("SCALATORCH_LOAD_PROFILE").contains("1");var mark=System.nanoTime()
          def lap(label: String): Unit = if prof then { val n=System.nanoTime();System.err.println(f"[scala-torch load]   $label: ${(n-mark)/1e9}%.2f s");mark=n }
          SingleFileSdxlConfig.inspect(archive);lap("inspect")
          lap("pre");cancelled()
          val clip=PretrainedClip.loadSdxlView(SingleFileTensorView.sdxlClip(archive),maxScoreElements)
          lap("clip");cancelled()
          val clip2=PretrainedClip.loadSdxlClip2View(SingleFileTensorView.sdxlClip2(archive),maxScoreElements)
          lap("clip2");cancelled()
          val unet=PretrainedUNet.loadSd1View(SingleFileTensorView.unet(archive),
            SingleFileSdxlConfig.unetConfigFor(archive),maxScoreElements,maxParameters=3000000000L)
          lap("unet");cancelled()
          val vae=PretrainedVae.loadSdxlView(SingleFileTensorView.vae(archive),
            if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
          lap("vae");cancelled()
          SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++clip2.linearTargets++unet.linearTargets,"sdxl",Some(clip2.model))
        else if SingleFileSd2Config.isSd2(archive.tensors) then
          SingleFileSd2Config.inspect(archive)
          cancelled()
          val clip=PretrainedClip.loadSd2View(SingleFileTensorView.sd2Clip(archive),maxScoreElements)
          cancelled()
          val unet=PretrainedUNet.loadSd1View(SingleFileTensorView.unet(archive),
            SingleFileSd2Config.unetConfigFor(archive),maxScoreElements)
          cancelled()
          val vae=PretrainedVae.loadSd1View(SingleFileTensorView.vae(archive),
            if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
          cancelled()
          SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++unet.linearTargets,"sd2")
        else
          SingleFileSd1Config.inspect(archive)
          cancelled()
          val clip=PretrainedClip.loadSingleFile(archive,maxScoreElements=maxScoreElements)
          cancelled()
          val unet=PretrainedUNet.loadSingleFile(archive,maxScoreElements=maxScoreElements)
          cancelled()
          val vae=PretrainedVae.loadSingleFile(archive,maxScoreElements=if maxVaeScoreElements<0 then maxScoreElements else maxVaeScoreElements)
          cancelled()
          SingleFileSd1Model(clip.model,unet.model,vae,clip.linearTargets++unet.linearTargets)
      }
    val after=Files.readAttributes(real,classOf[BasicFileAttributes])
    require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
      before.fileKey()==after.fileKey(),"checkpoint changed during loading")
    model
