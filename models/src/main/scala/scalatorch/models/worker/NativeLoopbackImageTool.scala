package scalatorch.models.worker

import scalatorch.models.DiffusionImageInput
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.ImageIO
import NativeJson.*

/** NUI Loopback Scaler's ordered Pillow-style RGB finishing operations. */
private[worker] object NativeLoopbackImageTool:
  private val smooth=Vector(1,1,1,1,5,1,1,1,1)
  private val detail=Vector(0,-1,0,-1,10,-1,0,-1,0)
  private val contour=Vector(-1,-1,-1,-1,8,-1,-1,-1,-1)
  private val blur=Vector.tabulate(25)(i => if i/5==0 || i/5==4 || i%5==0 || i%5==4 then 1 else 0)
  private def clamp(value: Int)=math.max(0,math.min(255,value))
  private def gray(rgb: Int): Int =
    (((rgb>>>16)&255)*19595+((rgb>>>8)&255)*38470+(rgb&255)*7471+32768)>>>16
  private def mapImage(image: BufferedImage,context: NativeWorkContext)(f: (Int,Int,Int)=>Int): BufferedImage =
    val out=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until image.getHeight do
      context.checkCancelled()
      for x <- 0 until image.getWidth do out.setRGB(x,y,f(x,y,image.getRGB(x,y)))
    out
  private def kernel(image: BufferedImage,weights: Vector[Int],size: Int,divisor: Int,offset: Int,
      context: NativeWorkContext): BufferedImage =
    val radius=size/2
    mapImage(image,context) { (x,y,pixel) =>
      if x<radius || y<radius || x>=image.getWidth-radius || y>=image.getHeight-radius then pixel
      else
        var result=0
        for shift <- Seq(16,8,0) do
          var sum=offset.toFloat+.5f
          for dy <- 0 until size; dx <- 0 until size do
            sum+=((image.getRGB(x+dx-radius,y+dy-radius)>>>shift)&255).toFloat*(weights(dy*size+dx).toFloat/divisor)
          val value=clamp(sum.toInt)
          result|=value<<shift
        result
    }
  private def blend(image: BufferedImage,base: BufferedImage,factor: Double,context: NativeWorkContext): BufferedImage =
    val alpha=factor.toFloat
    mapImage(image,context) { (x,y,pixel) =>
      val original=base.getRGB(x,y)
      Seq(16,8,0).foldLeft(0) { (rgb,shift) =>
        val a=(original>>>shift)&255;val b=(pixel>>>shift)&255
        rgb | (clamp((a+(b-a)*alpha).toInt)<<shift)
      }
    }
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    val factors=Vector("sharpness","brightness","color","contrast").map { key =>
      val value=p.get(key) match
        case None => 1.0
        case Some(Number(n)) => n.toDouble
        case _ => throw IllegalArgumentException(s"$key must be numeric")
      require(value.isFinite && value>=.1 && value<=2.0,s"$key must be in 0.1..2.0")
      value
    }
    val levels=Vector("detail","smooth","blur").map { key =>
      p.get(key) match
        case None | Some(Str("None")) => 0
        case Some(Str("Low")) => 1
        case Some(Str("Medium")) => 2
        case Some(Str("High")) => 3
        case _ => throw IllegalArgumentException(s"invalid $key filter level")
    }
    val useContour=p.get("contour") match
      case None => false
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException("contour must be boolean")
    def string(key: String,default: String): String=p.get(key) match
      case None | Some(Null) => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val source=string("image_path","");require(source.nonEmpty,"image_path is required")
    var image=DiffusionImageInput.readRgb(Path.of(source),compositeAlpha=false)
    val output=Path.of(string("output_dir",".")).toAbsolutePath.normalize()
    Files.createDirectories(output)
    if factors(0)!=1 then image=blend(image,kernel(image,smooth,3,13,0,context),factors(0),context)
    if factors(1)!=1 then
      val black=new BufferedImage(image.getWidth,image.getHeight,BufferedImage.TYPE_INT_RGB)
      image=blend(image,black,factors(1),context)
    if factors(2)!=1 then
      val grayscale=mapImage(image,context) { (_,_,p) => val v=gray(p);(v<<16)|(v<<8)|v }
      image=blend(image,grayscale,factors(2),context)
    if factors(3)!=1 then
      var sum=0L
      for y <- 0 until image.getHeight do
        context.checkCancelled()
        for x <- 0 until image.getWidth do sum+=gray(image.getRGB(x,y))
      val mean=math.floor(sum.toDouble/(image.getWidth.toLong*image.getHeight)+.5).toInt
      val base=mapImage(image,context)((_,_,_) => (mean<<16)|(mean<<8)|mean)
      image=blend(image,base,factors(3),context)
    for _ <- 0 until levels(0) do image=kernel(image,detail,3,6,0,context)
    for _ <- 0 until levels(1) do image=kernel(image,smooth,3,13,0,context)
    for _ <- 0 until levels(2) do image=kernel(image,blur,5,16,0,context)
    if useContour then image=kernel(image,contour,3,1,255,context)
    context.checkCancelled()
    val temp=Files.createTempFile(output,".native-loopback-",".png")
    try
      require(ImageIO.write(image,"png",temp.toFile),"PNG writer is unavailable")
      context.checkCancelled()
      val destination=output.resolve(s"native-loopback-${UUID.randomUUID()}.png")
      context.beginPublication()
      Files.move(temp,destination)
      Map("path"->Str(destination.toString),"outputs"->Obj(Map.empty),"info"->Obj(Map.empty))
    finally Files.deleteIfExists(temp)
