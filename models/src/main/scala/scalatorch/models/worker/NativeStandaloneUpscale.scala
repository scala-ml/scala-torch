package scalatorch.models.worker

import scalatorch.models.DiffusionImageInput
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.ImageIO
import NativeJson.*

/** Standalone NUI upscaling, independent of the loaded diffusion checkpoint. */
private[worker] object NativeStandaloneUpscale:
  private[worker] def canonicalUpscaler(value: String): Option[String] =
    val basename=value.trim.replace('\\','/').split('/').lastOption.getOrElse("").toLowerCase(java.util.Locale.ROOT)
    val stem=basename.lastIndexOf('.') match
      case i if i>0 => basename.substring(0,i)
      case _ => basename
    Option.when(basename=="r-esrgan 4x+" || stem=="realesrgan_x4plus" ||
      stem=="realesrgan_x4plus_anime_6b")("R-ESRGAN 4x+")

  def execute(p: Map[String,NativeJson],context: NativeWorkContext,device: String): Map[String,NativeJson] =
    val started=System.nanoTime()
    require(p.keySet.subsetOf(Set("image_path","output_dir","upscaler","scale","upscaler_paths","hardware_info")),
      "unsupported upscale parameters")
    def string(key: String,default: String): String=p.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val scale=p.get("scale") match
      case None => 2.0
      case Some(Number(value)) if value.toDouble.isFinite && value>0 && value<=16 => value.toDouble
      case _ => throw IllegalArgumentException("scale must be in (0, 16]")
    val hardware=NativeHardwareInfo.options(p.get("hardware_info"))
    val profile=hardware.map(_ => NativeHardwareInfo.profile(device))
    val paths=p.get("upscaler_paths") match
      case None | Some(Null) => Map.empty[String,String]
      case Some(Obj(fields)) => fields.map { (name,value) => value match
        case Str(path) => name -> path
        case _ => throw IllegalArgumentException(s"upscaler path for $name must be a string")
      }
      case _ => throw IllegalArgumentException("upscaler_paths must be an object")
    val upscaler=string("upscaler","Lanczos")
    val modelUpscaler=canonicalUpscaler(upscaler)
    val builtInUpscalers=Set("None","Lanczos","Nearest","LDSR","Waifu2x","waifu2x","waifu2x_anime_art_rgb","waifu2x_photo")
    if !builtInUpscalers(upscaler) && modelUpscaler.isEmpty then
      throw new NativeBackendError("unsupported",s"native upscaler '$upscaler' is not available")
    if modelUpscaler.nonEmpty then
      require(paths.get(upscaler).exists(path => Files.isRegularFile(Path.of(path))),"R-ESRGAN model path is missing or invalid")
    val imageName=string("image_path","")
    val outputName=string("output_dir","")
    require(imageName.nonEmpty && outputName.nonEmpty,"image_path and output_dir are required")
    val source=DiffusionImageInput.readRgb(Path.of(imageName))
    val output=Path.of(outputName).toAbsolutePath.normalize()
    require(Files.isDirectory(output),"output_dir must be an existing directory")
    def dimension(value: Int): Int =
      val result=math.max(1.0,math.rint(value.toDouble*scale))
      require(result<=16384,"upscale dimension exceeds limit")
      result.toInt
    val width=dimension(source.getWidth);val height=dimension(source.getHeight)
    require(width.toLong*height<=16777216L,"upscale output exceeds pixel limit")
    context.checkCancelled()
    val result=modelUpscaler match
      case None =>
        NativeImg2ImgHires.upscale(source,width,height,upscaler,()=>context.cancelled())
      case Some(modelName) =>
        var current=source
        var pass=0
        while (current.getWidth<width || current.getHeight<height) && pass<3 do
          context.checkCancelled()
          val nextWidth=current.getWidth.toLong*4;val nextHeight=current.getHeight.toLong*4
          require(nextWidth<=16384 && nextHeight<=16384 && nextWidth*nextHeight<=16777216L,
            "upscale intermediate exceeds pixel limit")
          current=NativeImg2ImgHires.upscale(current,nextWidth.toInt,nextHeight.toInt,modelName,
            paths.get(upscaler),()=>context.cancelled())
          pass+=1
        if current.getWidth==width && current.getHeight==height then current
        else NativeImg2ImgHires.upscale(current,width,height,"Lanczos",()=>context.cancelled())
    context.checkCancelled()
    val temp=Files.createTempFile(output,".native-upscale-",".png")
    try
      require(ImageIO.write(result,"png",temp.toFile),"PNG writer is unavailable")
      context.checkCancelled()
      val destination=output.resolve(s"native-upscale-${UUID.randomUUID()}.png")
      context.beginPublication()
      Files.move(temp,destination)
      Map[String,NativeJson]("path"->Str(destination.toString)) ++ hardware.map { remark =>
        "hardware_info"->NativeHardwareInfo.record(profile.get,remark,(System.nanoTime()-started)/1e9)
      }
    finally Files.deleteIfExists(temp)
