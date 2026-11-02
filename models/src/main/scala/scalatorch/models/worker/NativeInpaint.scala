package scalatorch.models.worker

import scalatorch.*
import scalatorch.models.{DiffusionImageInput,DiffusionImageResize}
import java.awt.image.BufferedImage

/** Bounded mask preprocessing for the basic four-channel inpaint path. */
private[worker] object NativeInpaint:
  final case class Prepared(image: Tensor[Float],originalImage: Tensor[Float],mask: Tensor[Float])

  private def grayRgb(source: BufferedImage): BufferedImage =
    val result=new BufferedImage(source.getWidth,source.getHeight,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until source.getHeight; x <- 0 until source.getWidth do
      val pixel=source.getRGB(x,y)
      val gray=(.299*((pixel>>>16)&255)+.587*((pixel>>>8)&255)+.114*(pixel&255)).round.toInt
      result.setRGB(x,y,(gray<<16)|(gray<<8)|gray)
    result

  private def rgbTensor(image: BufferedImage,cancelled: () => Boolean): Tensor[Float] =
    val width=image.getWidth;val height=image.getHeight;val plane=width*height
    val values=new Array[Float](plane*3)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
      for x <- 0 until width do
        val rgb=image.getRGB(x,y);val i=y*width+x
        values(i)=(((rgb>>>16)&255)/127.5f)-1f
        values(plane+i)=(((rgb>>>8)&255)/127.5f)-1f
        values(plane*2+i)=((rgb&255)/127.5f)-1f
    Tensor.fromFloats(values,Shape.from(Seq(1L,3L,height.toLong,width.toLong)))

  private def boxBlur(input: Array[Float],width: Int,height: Int,radius: Int,horizontal: Boolean,
      cancelled: () => Boolean): Array[Float] =
    if radius==0 then input
    else
      val output=new Array[Float](input.length)
      val lines=if horizontal then height else width
      val length=if horizontal then width else height
      val window=radius*2+1
      for line <- 0 until lines do
        if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
        def index(position: Int): Int =
          val clamped=math.max(0,math.min(length-1,position))
          if horizontal then line*width+clamped else clamped*width+line
        var sum=0.0
        for i <- -radius to radius do sum+=input(index(i))
        for position <- 0 until length do
          output(index(position))=(sum/window).toFloat
          sum-=input(index(position-radius))
          sum+=input(index(position+radius+1))
      output

  private def blur(input: Array[Float],width: Int,height: Int,radius: Int,cancelled: () => Boolean): Array[Float] =
    if radius==0 then input
    else
      // Three separable box passes approximate a Gaussian with linear work in pixel count.
      var output=input
      for _ <- 0 until 3 do output=boxBlur(output,width,height,radius,horizontal=true,cancelled)
      for _ <- 0 until 3 do output=boxBlur(output,width,height,radius,horizontal=false,cancelled)
      output

  def prepare(source: BufferedImage,mask: BufferedImage,width: Int,height: Int,resizeMode: Int,blurRadius: Int,
      invert: Boolean,cancelled: () => Boolean): Prepared =
    require(width>0 && height>0 && width.toLong*height<=16777216L,"invalid inpaint output size")
    require(blurRadius>=0 && blurRadius<=64,"mask_blur must be between 0 and 64")
    if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
    val original=DiffusionImageResize(source,width,height,resizeMode)
    var maskRgb=grayRgb(mask)
    if invert then
      for y <- 0 until maskRgb.getHeight; x <- 0 until maskRgb.getWidth do
        val v=255-(maskRgb.getRGB(x,y)&255)
        maskRgb.setRGB(x,y,(v<<16)|(v<<8)|v)
    if maskRgb.getWidth!=source.getWidth || maskRgb.getHeight!=source.getHeight then
      maskRgb=DiffusionImageResize(maskRgb,source.getWidth,source.getHeight,0)
    maskRgb=DiffusionImageResize(maskRgb,width,height,resizeMode)
    val plane=width*height
    val levels=Array.tabulate(plane) { i => (maskRgb.getRGB(i%width,i/width)&255)/255f }
    val softened=blur(levels,width,height,blurRadius,cancelled)
    val image=rgbTensor(original,cancelled)
    Prepared(image,image,Tensor.fromFloats(softened,Shape.from(Seq(1L,1L,height.toLong,width.toLong))))
