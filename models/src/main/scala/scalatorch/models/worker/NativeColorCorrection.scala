package scalatorch.models.worker

import scalatorch.models.{DiffusionImageInput,DiffusionImageResize}
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import java.util.UUID
import javax.imageio.ImageIO
import NativeJson.*

/** NUI's RGB AdaIN and five-level wavelet color transfer. */
private[worker] object NativeColorCorrection:
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    require(p.keySet.subsetOf(Set("tool","image_path","reference_path","output_dir","method","strength","all_methods")),"unsupported color correction parameters")
    def string(key: String,default: String): String=p.get(key) match
      case None | Some(Null) => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val method=string("method","AdaIN")
    if !Set("A1111","AdaIN","Wavelet")(method) then throw new NativeBackendError("unsupported",s"color correction method '$method' is not implemented")
    val allMethods=p.get("all_methods") match
      case None | Some(Bool(false)) => false
      case Some(Bool(true)) => true
      case _ => throw IllegalArgumentException("all_methods must be boolean")
    val strength=p.get("strength") match
      case None => 1.0
      case Some(Number(n)) => n.toDouble
      case _ => throw IllegalArgumentException("strength must be numeric")
    require(strength.isFinite && strength>=0 && strength<=1,"strength must be finite and between 0 and 1")
    val sourceName=string("image_path","");val refName=string("reference_path","")
    require(sourceName.nonEmpty && refName.nonEmpty,"image_path and reference_path are required")
    if allMethods then
      val directory=Path.of(string("output_dir",".")).toAbsolutePath.normalize();Files.createDirectories(directory)
      val staging=Files.createTempDirectory(directory,".native-color-staging-")
      val moved=scala.collection.mutable.ArrayBuffer.empty[Path]
      var completed=false
      try
        val localContext=new NativeWorkContext(context.cancelled,()=>true,context.event)
        val methods=Vector(method)++Vector("A1111","AdaIN","Wavelet").filterNot(_==method)
        val staged=methods.map { name =>
          context.checkCancelled()
          val inner=p.updated("method",Str(name)).updated("all_methods",Bool(false)).updated("output_dir",Str(staging.toString))
          val result=execute(inner,localContext)
          name->Path.of(result("path").asInstanceOf[Str].value)
        }
        context.checkCancelled()
        context.beginPublication()
        val published=staged.map { (name,path) =>
          val destination=directory.resolve(s"native-color-${UUID.randomUUID()}.png")
          Files.move(path,destination);moved+=destination
          name->Str(destination.toString)
        }.toMap
        completed=true
        return Map("path"->published(method),"outputs"->Obj(published-method),
          "info"->Obj(Map("method"->Str(method),"strength"->Number(BigDecimal(strength)))))
      finally
        if !completed then moved.foreach(Files.deleteIfExists(_))
        val files=Files.list(staging)
        try files.forEach(path=>{Files.deleteIfExists(path);()}) finally files.close()
        Files.deleteIfExists(staging)
    val source=DiffusionImageInput.readRgb(Path.of(sourceName),compositeAlpha=false)
    val ref=DiffusionImageInput.readRgb(Path.of(refName),compositeAlpha=false)
    val width=source.getWidth;val height=source.getHeight;val count=width*height
    val sample=DiffusionImageResize.bicubic(ref,width,height)
    def values(image: BufferedImage): Array[Double] =
      val out=new Array[Double](count*3)
      for y <- 0 until height do
        context.checkCancelled()
        for x <- 0 until width do
          val pixel=image.getRGB(x,y);val i=(y*width+x)*3
          for c <- 0 until 3 do out(i+c)=((pixel>>>(16-8*c))&255).toDouble
      out
    val original=values(source);val target=values(sample)
    val output=new Array[Int](count)
    if method=="A1111" then
      def pixels(image: BufferedImage): Array[Int] =
        val out=new Array[Int](image.getWidth*image.getHeight*3)
        for y <- 0 until image.getHeight do
          context.checkCancelled()
          for x <- 0 until image.getWidth do
            val pixel=image.getRGB(x,y);val at=(y*image.getWidth+x)*3
            for c <- 0 until 3 do out(at+c)=(pixel>>>(16-8*c))&255
        out
      val sourcePixels=pixels(source)
      val sourceLab=NativeRgbLab.toLab(sourcePixels,context.cancelled)
      val referenceLab=NativeRgbLab.toLab(pixels(ref),context.cancelled)
      val matched=NativeHistogramMatch.rgbChannels(sourceLab,referenceLab,context.cancelled)
      val corrected=NativeRgbLab.toRgb(matched,context.cancelled)
      val blended=NativeLuminosity.blend(corrected,sourcePixels,context.cancelled)
      for i <- 0 until count do
        if i%width==0 then context.checkCancelled()
        var rgb=0
        for c <- 0 until 3 do
          val j=i*3+c
          val value=if strength==1 then blended(j) else
            (sourcePixels(j)+(blended(j)-sourcePixels(j))*strength.toFloat).toInt
          rgb|=math.max(0,math.min(255,value))<<(16-8*c)
        output(i)=rgb
    else if method=="AdaIN" then
      val mean=new Array[Double](3);val targetMean=new Array[Double](3)
      for i <- original.indices do { mean(i%3)+=original(i)/255;targetMean(i%3)+=target(i)/255 }
      for c <- 0 until 3 do { mean(c)/=count;targetMean(c)/=count }
      val variance=new Array[Double](3);val targetVariance=new Array[Double](3)
      for i <- original.indices do
        val c=i%3;variance(c)+=math.pow(original(i)/255-mean(c),2);targetVariance(c)+=math.pow(target(i)/255-targetMean(c),2)
      val divisor=if count>1 then count-1 else 1
      for c <- 0 until 3 do
        variance(c)=math.sqrt(variance(c)/divisor+1e-5);targetVariance(c)=math.sqrt(targetVariance(c)/divisor+1e-5)
      for i <- 0 until count do
        if i%width==0 then context.checkCancelled()
        var rgb=0
        for c <- 0 until 3 do
          val value=original(i*3+c)/255
          val adjusted=(value-mean(c))/variance(c)*targetVariance(c)+targetMean(c)
          val blended=(value+strength*(adjusted-value))*255+1e-10
          rgb|=math.max(0,math.min(255,blended.toInt))<<(16-c*8)
        output(i)=rgb
    else
      def coarse(input: Array[Double]): Array[Double] =
        var current=input.clone()
        for radius <- Seq(1,2,4,8,16); vertical <- Seq(true,false) do
          val next=new Array[Double](current.length)
          for y <- 0 until height do
            context.checkCancelled()
            for x <- 0 until width; c <- 0 until 3 do
              val index=(y*width+x)*3+c
              val before=if vertical then (math.max(0,y-radius)*width+x)*3+c else (y*width+math.max(0,x-radius))*3+c
              val after=if vertical then (math.min(height-1,y+radius)*width+x)*3+c else (y*width+math.min(width-1,x+radius))*3+c
              next(index)=(current(before)+2*current(index)+current(after))/4
          current=next
        current
      val sourceCoarse=coarse(original);val targetCoarse=coarse(target)
      for i <- 0 until count do
        if i%width==0 then context.checkCancelled()
        var rgb=0
        for c <- 0 until 3 do
          val j=i*3+c
          val value=math.rint(math.max(0,math.min(255,original(j)+strength*(targetCoarse(j)-sourceCoarse(j))))).toInt
          rgb|=value<<(16-c*8)
        output(i)=rgb
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    image.setRGB(0,0,width,height,output,0,width)
    val directory=Path.of(string("output_dir",".")).toAbsolutePath.normalize();Files.createDirectories(directory)
    val temp=Files.createTempFile(directory,".native-color-",".png")
    try
      require(ImageIO.write(image,"png",temp.toFile),"PNG writer unavailable")
      context.checkCancelled()
      val destination=directory.resolve(s"native-color-${UUID.randomUUID()}.png")
      context.beginPublication();Files.move(temp,destination)
      Map("path"->Str(destination.toString),"outputs"->Obj(Map.empty),
        "info"->Obj(Map("method"->Str(method),"strength"->Number(BigDecimal(strength)))))
    finally Files.deleteIfExists(temp)
