package scalatorch.models.worker

import scalatorch.models.{DiffusionImageInput,DiffusionImageResize}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.ImageIO
import NativeJson.*

/** NUI's non-neural Lanczos plus contrast-adaptive sharpening image tool. */
private[worker] object NativeCasImageTool:
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    if p.get("tool").contains(Str("loopback_scaler")) then return NativeLoopbackImageTool.execute(p,context)
    if p.get("tool").contains(Str("color_correction")) then return NativeColorCorrection.execute(p,context)
    if p.get("tool").contains(Str("color_enhance")) then return NativeColorEnhance.execute(p,context)
    if p.get("tool").contains(Str("pixel_art")) then return NativePixelArtImageTool.execute(p,context)
    p.get("tool") match
      case Some(Str(t)) if Set("nsfw_checker", "aesthetic_score", "person_mask", "remove_background", "depth_mask", "waifu2x").contains(t) =>
        return NativeBatch18Services.executeImageTool(p, context)
      case _ => ()
    def string(key: String,default: String): String=p.get(key) match
      case None | Some(Null) => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val tool=string("tool","")
    if tool!="cas_upscale" then throw new NativeBackendError("unsupported",s"native image tool '$tool' is not implemented")
    require(p.keySet.subsetOf(Set("tool","image_path","output_dir","scale","sharpness")),"unsupported CAS parameters")
    def number(key: String,default: Double): Double=p.get(key) match
      case None => default
      case Some(Number(value)) if value.toDouble.isFinite => value.toDouble
      case _ => throw IllegalArgumentException(s"$key must be finite and numeric")
    val scale=number("scale",2.0)
    require(scale>0 && scale<=16,"CAS scale must be in (0, 16]")
    val sharpness=math.max(0.0,math.min(1.0,number("sharpness",.5)))
    val sourceName=string("image_path","");require(sourceName.nonEmpty,"image_path is required")
    val image=DiffusionImageInput.readRgb(Path.of(sourceName),compositeAlpha=false)
    val output=Path.of(string("output_dir",".")).toAbsolutePath.normalize()
    Files.createDirectories(output)
    def dimension(size: Int): Int =
      val result=math.max(1.0,math.rint(size*scale))
      require(result<=16384,"CAS dimension exceeds limit")
      result.toInt
    val width=dimension(image.getWidth);val height=dimension(image.getHeight)
    require(width.toLong*height<=16777216L,"CAS output exceeds pixel limit")
    context.checkCancelled()
    val up=DiffusionImageResize(image,width,height,0)
    val result=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    val peak=(-1.0/(8.0-3.0*sharpness)).toFloat
    for y <- 0 until height do
      context.checkCancelled()
      for x <- 0 until width do
        val center=up.getRGB(x,y)
        val north=up.getRGB(x,math.max(0,y-1));val south=up.getRGB(x,math.min(height-1,y+1))
        val west=up.getRGB(math.max(0,x-1),y);val east=up.getRGB(math.min(width-1,x+1),y)
        var rgb=0
        for shift <- Seq(16,8,0) do
          def value(pixel: Int): Float=((pixel>>>shift)&255)/255f
          val c=value(center);val n=value(north);val s=value(south);val w=value(west);val e=value(east)
          val minimum=math.min(c,math.min(math.min(n,s),math.min(w,e)))
          val maximum=math.max(c,math.max(math.max(n,s),math.max(w,e)))
          val amp=math.sqrt(math.max(0f,math.min(1f,math.min(minimum,1f-maximum)/math.max(maximum,1e-5f))).toDouble).toFloat
          val weight=amp*peak
          val v=math.max(0f,math.min(1f,(c+(n+s+w+e)*weight)/(1f+4f*weight)))
          rgb|=math.min(255,(v*255f+.5f).toInt)<<shift
        result.setRGB(x,y,rgb)
    context.checkCancelled()
    val temp=Files.createTempFile(output,".native-cas-",".png")
    try
      require(ImageIO.write(result,"png",temp.toFile),"PNG writer is unavailable")
      context.checkCancelled()
      val destination=output.resolve(s"native-cas-${UUID.randomUUID()}.png")
      context.beginPublication()
      Files.move(temp,destination)
      Map("path"->Str(destination.toString),"outputs"->Obj(Map.empty),"info"->Obj(Map.empty))
    finally Files.deleteIfExists(temp)
