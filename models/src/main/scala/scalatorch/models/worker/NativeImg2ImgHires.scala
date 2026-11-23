package scalatorch.models.worker

import scalatorch.*
import scalatorch.models.{CudaRealEsrganX4Plus,DiffusionImageResize}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException

/** NUI's separate pixel-upscale then img2img refinement hook. */
private[worker] object NativeImg2ImgHires:
  import NativeJson.*

  final case class Config(ratio: Double,width: Int,height: Int,steps: Int,upscaler: String,upscalerPath: Option[String],prompt: String,
      negativePrompt: String,denoise: Double,sampler: String,scheduler: String,cfg: Double):
    def target(sourceWidth: Int,sourceHeight: Int): (Int,Int) =
      require(sourceWidth>0 && sourceHeight>0,"invalid img2img Hires source dimensions")
      val (requestedWidth,requestedHeight)=if width>0 && height>0 then (width,height)
        else ((sourceWidth*ratio).toInt,(sourceHeight*ratio).toInt)
      val targetWidth=math.max(64,requestedWidth/8*8)
      val targetHeight=math.max(64,requestedHeight/8*8)
      if targetWidth>16384 || targetHeight>16384 then
        throw new NativeBackendError("invalid_params","img2img Hires Fix output exceeds 16384 pixels per side")
      (targetWidth,targetHeight)

  private def fail(message: String): Nothing=throw new NativeBackendError("invalid_params",message)
  private def invalidUp(message: String): Nothing=throw new NativeBackendError("unsupported",message)

  def parse(value: NativeJson,upscalerPaths: Map[String,String]=Map.empty): Config = value match
    case Obj(fields) =>
      def number(key: String,default: Double,maximum: Double): Double=fields.get(key) match
        case None => default
        case Some(Number(value)) if value.toDouble.isFinite && value>=0 && value<=BigDecimal(maximum) => value.toDouble
        case _ => fail(s"img2img Hires Fix invalid $key")
      def integer(key: String,default: Int,maximum: Int,multipleOfEight: Boolean=false): Int=fields.get(key) match
        case None => default
        case Some(Number(value)) if value.isWhole && value>=0 && value<=maximum &&
            (!multipleOfEight || value.toInt%8==0) => value.toInt
        case _ => fail(s"img2img Hires Fix invalid $key")
      def string(key: String,default: String): String=fields.get(key) match
        case None => default
        case Some(Str(value)) if value.length<=4096 => value.trim
        case _ => fail(s"img2img Hires Fix invalid $key")
      val ratio=number("ratio",2.0,4.0)
      val width=integer("width",0,2048,multipleOfEight=true)
      val height=integer("height",0,2048,multipleOfEight=true)
      if ratio==0 && !(width>0 && height>0) then fail("img2img Hires Fix needs a positive ratio or both target dimensions")
      val upscaler=string("upscaler","R-ESRGAN 4x+")
      val selectedModel=NativeStandaloneUpscale.canonicalUpscaler(upscaler).nonEmpty
      val upscalerPath=if selectedModel then upscalerPaths.get(upscaler).orElse(upscalerPaths.get("R-ESRGAN 4x+")) else None
      val builtInPixelUpscalers=Set("Lanczos","Nearest","None","LDSR","Waifu2x","waifu2x","waifu2x_anime_art_rgb","waifu2x_photo")
      if !builtInPixelUpscalers(upscaler) && !selectedModel then
        invalidUp(s"img2img Hires Fix upscaler '$upscaler' is not available in the native worker")
      if selectedModel && upscalerPath.isEmpty then
        invalidUp(s"img2img Hires Fix requires a resolved model path for $upscaler")
      val denoise=number("denoise",.33,1.0)
      val steps=integer("steps",0,25)
      val cfg=number("cfg",0,30)
      val prompt=string("prompt","")
      val negative=string("negative_prompt","")
      val sampler=string("sampler","Use same sampler")
      val scheduler=string("scheduler","Use same scheduler")
      if sampler.isEmpty || scheduler.isEmpty then fail("img2img Hires Fix sampler and scheduler must be specified")
      if !Set("Use same sampler","Euler","Euler a","Heun","LMS","DPM2","DPM2 a","DPM++ 2S a","DPM++ SDE","DPM++ 2M","DPM++ 2M SDE","DPM++ 2M SDE Heun","DPM++ 3M SDE","DDIM","UniPC","LCM")(sampler) then
        invalidUp(s"img2img Hires Fix sampler '$sampler' is not available in the native worker")
      if !Set("Use same scheduler","Automatic","Uniform","Karras","Exponential","Polyexponential","Normal","Simple","SGM Uniform","Align Your Steps","KL Optimal","DDIM","Beta")(scheduler) then
        invalidUp(s"img2img Hires Fix scheduler '$scheduler' is not available in the native worker")
      val unknown=fields.keySet--Set("ratio","width","height","steps","upscaler","prompt","negative_prompt",
        "denoise","sampler","scheduler","cfg","enabled")
      fields.get("enabled").foreach {
        case Bool(true) => ()
        case _ => fail("img2img Hires Fix enabled must be true when configured")
      }
      if unknown.nonEmpty then fail(s"unsupported img2img Hires Fix options: ${unknown.toVector.sorted.mkString(", ")}")
      Config(ratio,width,height,steps,if selectedModel then "R-ESRGAN 4x+" else upscaler,upscalerPath,
        prompt,negative,denoise,sampler,scheduler,cfg)
    case _ => fail("img2img Hires Fix configuration must be an object")

  def upscale(source: BufferedImage,width: Int,height: Int,name: String,modelPath: Option[String],cancelled: () => Boolean,
      maxPixels: Long=16777216L): BufferedImage =
    require(width>0 && height>0 && width.toLong*height<=maxPixels,"img2img Hires target exceeds pixel limit")
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("img2img Hires upscale cancelled")
    name match
      case "Waifu2x" | "waifu2x" | "waifu2x_anime_art_rgb" | "waifu2x_photo" =>
        // real waifu2x upconv_7 (Anime/art, or Photo for waifu2x_photo), denoise level 2 like the tool's default
        NativeWaifu2x.upscale(source,width,height,if name=="waifu2x_photo" then "Photo" else "Anime",2,cancelled)
      case "Lanczos" | "None" | "LDSR" =>
        DiffusionImageResize(source,width,height,0,maxPixels)
      case "Nearest" =>
        val result=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
        for y <- 0 until height do
          if cancelled() || Thread.currentThread().isInterrupted then
            throw new CancellationException("img2img Hires upscale cancelled")
          val sy=math.min(source.getHeight-1,((2L*y+1)*source.getHeight/(2L*height)).toInt)
          for x <- 0 until width do
            val sx=math.min(source.getWidth-1,((2L*x+1)*source.getWidth/(2L*width)).toInt)
            result.setRGB(x,y,source.getRGB(sx,sy))
        result
      case "R-ESRGAN 4x+" =>
        val path=Path.of(modelPath.getOrElse(throw new NativeBackendError("invalid_params","R-ESRGAN model path is missing")))
          .toAbsolutePath.normalize()
        require(Files.isRegularFile(path),"R-ESRGAN checkpoint file is missing")
        val environment=sys.env
        val deviceOpt=environment.getOrElse("SDFX_SCALATORCH_UNET_DEVICE","cpu") match
          case "cuda" => Some(0)
          case value if value.matches("cuda:[0-9]+") => Some(value.drop(5).toInt)
          case "cpu" => None
          case _ => invalidUp("R-ESRGAN model upscaling requires a valid CPU or CUDA U-Net device")
        val maxBytes=environment.get("SDFX_SCALATORCH_UNET_MAX_BYTES").fold(1073741824L)(scalatorch.ffm.CudaSession.parseBudget)
        val upscaled=deviceOpt match
          case Some(device) => CudaRealEsrganX4Plus.upscale(source,path,cancelled,device,maxBytes)
          case None => CudaRealEsrganX4Plus.upscaleCpu(source,path,cancelled)
        DiffusionImageResize(upscaled,width,height,0,maxPixels)
      case other => invalidUp(s"img2img Hires Fix upscaler '$other' is not available in the native worker")

  def upscale(source: BufferedImage,width: Int,height: Int,name: String,cancelled: () => Boolean,
      maxPixels: Long): BufferedImage = upscale(source,width,height,name,None,cancelled,maxPixels)

  def upscale(source: BufferedImage,width: Int,height: Int,name: String,cancelled: () => Boolean): BufferedImage =
    upscale(source,width,height,name,None,cancelled)

  def tensor(image: BufferedImage,cancelled: () => Boolean): Tensor[Float] =
    val width=image.getWidth;val height=image.getHeight
    require(width>0 && height>0 && width.toLong*height<=16777216L,"invalid img2img Hires image size")
    val plane=width*height
    val values=new Array[Float](plane*3)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new CancellationException("img2img Hires image conversion cancelled")
      for x <- 0 until width do
        val rgb=image.getRGB(x,y);val offset=y*width+x
        values(offset)=(((rgb>>>16)&255)/127.5f)-1f
        values(plane+offset)=(((rgb>>>8)&255)/127.5f)-1f
        values(2*plane+offset)=((rgb&255)/127.5f)-1f
    Tensor.fromFloats(values,Shape.from(Seq(1L,3L,height.toLong,width.toLong)))
