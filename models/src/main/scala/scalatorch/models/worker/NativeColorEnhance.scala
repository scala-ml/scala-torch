package scalatorch.models.worker

import scalatorch.models.DiffusionImageInput
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.ImageIO
import NativeJson.*

/** Chroma stretching in CIELAB for NUI's color-enhance image tool. */
private[worker] object NativeColorEnhance:
  // OpenCV's 33^3 fixed-point RGB-to-LAB interpolation grid; see opencv.txt.
  private lazy val labGrid: Array[Int] =
    val input=getClass.getResourceAsStream("/scalatorch/models/opencv-float-lab-grid.bin")
    require(input!=null,"OpenCV LAB interpolation grid is missing")
    val bytes=try input.readAllBytes() finally input.close()
    require(bytes.length==33*33*33*3*2,"invalid LAB interpolation grid")
    Array.tabulate(bytes.length/2)(i=>(bytes(i*2)&255)|((bytes(i*2+1)&255)<<8))
  private def floatLab(r: Int,g: Int,b: Int): (Double,Double,Double) =
    val values=Array(r,g,b).map(v=>math.rint(((v/255f)*16384f).toDouble).toInt)
    val base=values.map(_>>9);val fraction=values.map(v=>(v>>5)&15)
    val sum=new Array[Long](3)
    for dr <- 0 to 1; dg <- 0 to 1; db <- 0 to 1 do
      val weight=(if dr==0 then 16-fraction(0) else fraction(0)) *
        (if dg==0 then 16-fraction(1) else fraction(1)) *
        (if db==0 then 16-fraction(2) else fraction(2))
      val ir=math.min(32,base(0)+dr);val ig=math.min(32,base(1)+dg);val ib=math.min(32,base(2)+db)
      val at=((ir*33+ig)*33+ib)*3
      for c <- 0 until 3 do sum(c)+=labGrid(at+c).toLong*weight
    val l=((sum(0)+2048)>>12).toFloat*(100f/16384f)
    val a=((sum(1)+2048)>>12).toFloat/64f-128f
    val b1=((sum(2)+2048)>>12).toFloat/64f-128f
    (l.toDouble,a.toDouble,b1.toDouble)
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    require(p.keySet.subsetOf(Set("tool","image_path","output_dir","strength","percentile")),"unsupported color enhancement parameters")
    def str(key: String,default: String): String=p.get(key) match
      case None | Some(Null) => default
      case Some(Str(v)) => v
      case _ => throw IllegalArgumentException(s"$key must be a string")
    def number(key: String,default: Double): Double=p.get(key) match
      case None => default
      case Some(Number(v)) if v.toDouble.isFinite => v.toDouble
      case _ => throw IllegalArgumentException(s"$key must be finite and numeric")
    val strength=number("strength",1);val percentile=number("percentile",99.5)
    require(percentile>=0 && percentile<=100,"percentile must be between 0 and 100")
    val sourceName=str("image_path","");require(sourceName.nonEmpty,"image_path is required")
    val image=DiffusionImageInput.readRgb(Path.of(sourceName),compositeAlpha=false)
    val width=image.getWidth;val height=image.getHeight;val count=width*height
    val L=new Array[Double](count);val A=new Array[Double](count);val B=new Array[Double](count)
    val chroma=new Array[Double](count)
    for y <- 0 until height do
      context.checkCancelled()
      for x <- 0 until width do
        val i=y*width+x;val p=image.getRGB(x,y)
        val (l,a,b)=floatLab((p>>>16)&255,(p>>>8)&255,p&255)
        L(i)=l;A(i)=a;B(i)=b
        chroma(i)=math.sqrt((a.toFloat*a.toFloat+b.toFloat*b.toFloat).toFloat.toDouble).toFloat.toDouble
    val sorted=chroma.sorted
    val position=percentile/100*(count-1);val low=position.toInt;val high=math.min(count-1,low+1)
    val maximum=sorted(low)+(sorted(high)-sorted(low))*(position-low)
    val gain=if maximum<1e-3 then 1.0 else math.max(0,math.min(4,1+(100/maximum-1)*strength))
    val pixels=new Array[Int](count)
    def inverse(v: Double): Double =
      val cube=v*v*v
      if cube>216.0/24389 then cube else (v-16.0/116)/(841.0/108)
    def srgb(v: Double): Int =
      val value=math.max(0,math.min(1,v))
      val encoded=if value<=.0031308 then 12.92*value else 1.055*math.pow(value,1/2.4)-.055
      math.max(0,math.min(255,(encoded*255+.5).toInt))
    for y <- 0 until height do
      context.checkCancelled()
      for x <- 0 until width do
        val i=y*width+x
        if maximum<1e-3 then pixels(i)=image.getRGB(x,y)
        else
          val fy=(L(i)+16)/116;val xx=.950456*inverse(fy+A(i)*gain/500)
          val yy=inverse(fy);val zz=1.088754*inverse(fy-B(i)*gain/200)
          val r=srgb(3.240479*xx-1.537150*yy-.498535*zz)
          val g=srgb(-.969256*xx+1.875991*yy+.041556*zz)
          val b=srgb(.055648*xx-.204043*yy+1.057311*zz)
          pixels(i)=(r<<16)|(g<<8)|b
    val output=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    output.setRGB(0,0,width,height,pixels,0,width)
    val directory=Path.of(str("output_dir",".")).toAbsolutePath.normalize();Files.createDirectories(directory)
    val temp=Files.createTempFile(directory,".native-enhance-",".png")
    try
      require(ImageIO.write(output,"png",temp.toFile),"PNG writer unavailable")
      context.checkCancelled()
      val destination=directory.resolve(s"native-enhance-${UUID.randomUUID()}.png")
      context.beginPublication();Files.move(temp,destination)
      val info=if maximum<1e-3 then Map.empty[String,NativeJson] else
        Map("chroma_gain"->Number(BigDecimal(math.rint(gain*1000)/1000)))
      Map("path"->Str(destination.toString),"outputs"->Obj(Map.empty),"info"->Obj(info))
    finally Files.deleteIfExists(temp)
