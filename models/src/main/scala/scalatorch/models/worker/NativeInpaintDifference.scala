package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.util.concurrent.CancellationException
import scalatorch.models.DiffusionImageResize

/** Pixel-space difference masks for NUI's Inpaint Difference workflow. */
private[worker] object NativeInpaintDifference:
  private val maxPixels=4096L*4096
  final case class Settings(basePath: String,threshold: Double,dilation: Int,erosion: Int,contours: Boolean)
  def parse(value: NativeJson): Settings =
    import NativeJson.*
    val fields=value match
      case Obj(values) => values
      case _ => throw IllegalArgumentException("inpaint difference must be an object")
    require((fields.keySet--Set("enabled","base_path","threshold","contours","dilation","erosion")).isEmpty,
      "unknown inpaint difference option")
    val path=fields.get("base_path") match
      case Some(Str(value)) if value.nonEmpty && value.length<=32768 => value
      case _ => throw IllegalArgumentException("inpaint difference base_path is required")
    val threshold=fields.get("threshold") match
      case None => 1.0
      case Some(Number(value)) => value.toDouble
      case _ => throw IllegalArgumentException("invalid difference threshold")
    require(threshold.isFinite && threshold>=0 && threshold<=1,"difference threshold must be in 0..1")
    val contours=fields.get("contours") match
      case None => false
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException("contours must be boolean")
    def amount(key: String): Int=fields.get(key) match
      case None => 0
      case Some(Number(value)) if value.isValidInt && value>=0 && value<=100 => value.toInt
      case _ => throw IllegalArgumentException(s"invalid difference $key")
    Settings(path,threshold,amount("dilation"),amount("erosion"),contours)

  def mask(base: BufferedImage,altered: BufferedImage,threshold: Double,dilation: Int,erosion: Int,
      cancelled: () => Boolean,contours: Boolean=false): Array[Int] =
    require(base!=null && altered!=null && cancelled!=null,"difference images and cancellation callback are required")
    require(threshold.isFinite && threshold>=0 && threshold<=1,"difference threshold must be in 0..1")
    require(dilation>=0 && dilation<=100 && erosion>=0 && erosion<=100,"difference morphology must be in 0..100")
    val width=altered.getWidth;val height=altered.getHeight
    require(width>0 && height>0 && width.toLong*height<=maxPixels && base.getWidth>0 && base.getHeight>0 &&
      base.getWidth.toLong*base.getHeight<=maxPixels,"difference image dimensions exceed limit")
    val aligned=if base.getWidth==width && base.getHeight==height then base
      else DiffusionImageResize.bicubic(base,width,height,maxPixels)
    def check(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("difference mask cancelled")
    var pixels=new Array[Int](width*height)
    for y <- 0 until height do
      check()
      for x <- 0 until width do
        val a=aligned.getRGB(x,y);val b=altered.getRGB(x,y)
        val sum=math.abs((a&255)-(b&255))+math.abs(((a>>>8)&255)-((b>>>8)&255))+
          math.abs(((a>>>16)&255)-((b>>>16)&255))
        val mean=sum/3
        pixels(y*width+x)=if mean/255.0>1.0-threshold then 255 else 0
    if contours then
      val boundary=new Array[Int](pixels.length)
      for y <- 1 until height-1 do
        check()
        for x <- 1 until width-1 do
          val i=y*width+x
          if pixels(i)!=0 && (pixels(i-1)==0 || pixels(i+1)==0 || pixels(i-width)==0 || pixels(i+width)==0) then
            boundary(i)=255
      pixels=boundary
    def morph(expand: Boolean,iterations: Int): Unit =
      if iterations>0 then
        // Repeated 3x3 binary morphology equals a square window of this radius.
        // Outside pixels are neutral (black for dilation, white for erosion).
        // A sliding count makes even the maximum radius linear in image size.
        def pass(input: Array[Int],horizontal: Boolean): Array[Int] =
          val output=new Array[Int](input.length)
          val lines=if horizontal then height else width
          val count=if horizontal then width else height
          for line <- 0 until lines do
            check()
            def index(pos: Int): Int=if horizontal then line*width+pos else pos*width+line
            def marked(pos: Int): Int =
              if (input(index(pos))!=0)==expand then 1 else 0
            var found=0
            var i=0
            while i<=math.min(count-1,iterations) do
              found+=marked(i);i+=1
            var pos=0
            while pos<count do
              output(index(pos))=if found>0 then (if expand then 255 else 0) else (if expand then 0 else 255)
              val remove=pos-iterations;val add=pos+iterations+1
              if remove>=0 then found-=marked(remove)
              if add<count then found+=marked(add)
              pos+=1
          output
        pixels=pass(pass(pixels,horizontal=true),horizontal=false)
    morph(true,dilation)
    morph(false,erosion)
    pixels
