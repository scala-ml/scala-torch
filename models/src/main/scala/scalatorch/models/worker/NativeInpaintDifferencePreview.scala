package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.DiffusionImageInput
import NativeJson.*

/** Model-free preview for NUI's difference mask; artifacts belong to the caller. */
private[worker] object NativeInpaintDifferencePreview:
  def execute(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    def str(key: String,default: String): String=p.get(key) match
      case None => default
      case Some(Str(v)) => v
      case _ => throw IllegalArgumentException(s"$key must be a string")
    def bool(key: String,default: Boolean): Boolean=p.get(key) match
      case None => default
      case Some(Bool(v)) => v
      case _ => throw IllegalArgumentException(s"$key must be boolean")
    val base=str("base_path","");val alteredPath=str("altered_path","");val output=str("output_dir","")
    require(Seq(base,alteredPath,output).forall(v=>v.nonEmpty && v.length<=32768),"image paths and output_dir are required")
    val settings=NativeInpaintDifference.parse(Obj(p.filter((k,_)=>Set("base_path","threshold","contours","dilation","erosion")(k))))
    val blur=p.get("blur") match
      case None => 0
      case Some(Number(v)) if v.isValidInt && v>=0 && v<=64 => v.toInt
      case _ => throw IllegalArgumentException("blur must be an integer in 0..64")
    val color=str("color","#ffffff");require(color.matches("#[0-9a-fA-F]{6}"),"color must be #RRGGBB")
    val mode=str("preview_mode","source");require(Set("source","effective")(mode),"invalid preview_mode")
    val show=bool("show_under",true);val invert=bool("invert",false)
    val first=DiffusionImageInput.readRgb(Path.of(base),maxBytes=200000000L)
    val altered=DiffusionImageInput.readRgb(Path.of(alteredPath),maxBytes=200000000L)
    val w=altered.getWidth;val h=altered.getHeight
    val mask=NativeInpaintDifference.mask(first,altered,settings.threshold,settings.dilation,settings.erosion,context.cancelled,settings.contours)
    val displayed=if mode=="effective" then
      NativeInpaint.gaussianBytes(if invert then mask.map(255-_) else mask,w,h,blur,context.cancelled).map(_.toDouble)
    else sourceBlur(mask,w,h,blur,settings.contours,settings.dilation!=0 || settings.erosion!=0,context)
    val maskImage=new BufferedImage(w,h,BufferedImage.TYPE_BYTE_GRAY)
    val preview=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB)
    val rgb=Array(Integer.parseInt(color.substring(1,3),16),Integer.parseInt(color.substring(3,5),16),Integer.parseInt(color.substring(5,7),16))
    for y <- 0 until h do
      context.checkCancelled()
      for x <- 0 until w do
        val i=y*w+x;maskImage.getRaster.setSample(x,y,0,mask(i))
        val level=displayed(i);val pixel=altered.getRGB(x,y)
        def channel(c: Int,shift: Int): Int =
          val overlay=level*(rgb(c)/255.0)
          val value=if show then ((pixel>>>shift)&255)*(1-level/255.0)+overlay*(level/255.0) else overlay
          math.max(0,math.min(255,value.toInt))
        preview.setRGB(x,y,(channel(0,16)<<16)|(channel(1,8)<<8)|channel(2,0))
    val root=Path.of(output).toAbsolutePath.normalize();Files.createDirectories(root)
    val directory=Files.createTempDirectory(root,"inpaint-difference-")
    val maskPath=directory.resolve("mask.png");val previewPath=directory.resolve("preview.png")
    var published=false
    try
      require(ImageIO.write(maskImage,"png",maskPath.toFile),"PNG writer unavailable")
      context.checkCancelled()
      require(ImageIO.write(preview,"png",previewPath.toFile),"PNG writer unavailable")
      context.checkCancelled();context.beginPublication();published=true
      Map("artifact_dir"->Str(directory.toString),"mask_path"->Str(maskPath.toString),"preview_path"->Str(previewPath.toString),"width"->Number(w),"height"->Number(h))
    finally
      if !published then
        Files.deleteIfExists(maskPath);Files.deleteIfExists(previewPath);Files.deleteIfExists(directory)

  private def sourceBlur(mask: Array[Int],w: Int,h: Int,blur: Int,contours: Boolean,integer: Boolean,context: NativeWorkContext): Array[Double] =
    if blur==0 then mask.map(_.toDouble)
    else
      val radius=(2.5*blur+0.5).toInt
      val weights=(-radius to radius).map(i=>math.exp(-i.toDouble*i/(2.0*blur*blur))).toArray
      val sum=weights.sum
      for i <- weights.indices do weights(i)/=sum
      def reflect(i: Int,n: Int): Int =
        if n==1 then 0 else
          val period=2*n-2;val v=Math.floorMod(i,period)
          if v<n then v else period-v
      val tmp=new Array[Double](mask.length);val result=new Array[Double](mask.length)
      for y <- 0 until h do
        context.checkCancelled()
        for x <- 0 until w do
          var value=0.0;var k= -radius
          while k<=radius do
            value+=mask(y*w+reflect(x+k,w))*weights(k+radius);k+=1
          tmp(y*w+x)=if contours && !integer then value.toFloat.toDouble else value
      for y <- 0 until h do
        context.checkCancelled()
        for x <- 0 until w do
          var value=0.0;var k= -radius
          while k<=radius do
            value+=tmp(reflect(y+k,h)*w+x)*weights(k+radius);k+=1
          result(y*w+x)=if integer then math.rint(value) else if contours then value.toFloat.toDouble else value
      result
