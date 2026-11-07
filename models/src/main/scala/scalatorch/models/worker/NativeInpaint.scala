package scalatorch.models.worker

import scalatorch.*
import scalatorch.models.{DiffusionImageInput,DiffusionImageResize,DiffusionImageIO}
import java.awt.image.BufferedImage

/** Bounded mask preprocessing and canvas restoration for the four-channel inpaint path. */
private[worker] object NativeInpaint:
  private final class PythonRandom(seed: Long):
    private val n=624
    private val state=new Array[Int](n)
    private var index=n
    private def initialize(seed: Long): Unit =
      var value=BigInt(seed).abs
      val key=scala.collection.mutable.ArrayBuffer.empty[Int]
      while value>0 do
        key+= (value & BigInt("ffffffff",16)).toLong.toInt
        value=value>>32
      if key.isEmpty then key+=0
      state(0)=19650218
      for i <- 1 until n do state(i)=1812433253*(state(i-1)^(state(i-1)>>>30))+i
      var i=1;var j=0
      var k=math.max(n,key.size)
      while k>0 do
        state(i)=(state(i)^((state(i-1)^(state(i-1)>>>30))*1664525))+key(j)+j
        i+=1;j+=1
        if i>=n then { state(0)=state(n-1);i=1 }
        if j>=key.size then j=0
        k-=1
      k=n-1
      while k>0 do
        state(i)=(state(i)^((state(i-1)^(state(i-1)>>>30))*1566083941))-i
        i+=1
        if i>=n then { state(0)=state(n-1);i=1 }
        k-=1
      state(0)=0x80000000
      index=n
    initialize(seed)
    private def word(): Int =
      if index>=n then
        for i <- 0 until n do
          val y=(state(i)&0x80000000)|(state((i+1)%n)&0x7fffffff)
          state(i)=state((i+397)%n)^(y>>>1)^(if (y&1)!=0 then 0x9908b0df else 0)
        index=0
      var y=state(index);index+=1
      y^=y>>>11
      y^=(y<<7)&0x9d2c5680
      y^=(y<<15)&0xefc60000
      y^=y>>>18
      y
    def bytes(length: Int,cancelled: () => Boolean=() => false): Array[Byte] =
      require(length>=0 && length<=50331648,"inpaint noise exceeds allocation limit")
      val output=new Array[Byte](length)
      var offset=0
      while offset<length do
        if (offset&65535)==0 && (cancelled() || Thread.currentThread().isInterrupted) then
          throw new java.util.concurrent.CancellationException("inpainting noise generation cancelled")
        val count=math.min(4,length-offset)
        val bits=if count==4 then word() else word() >>> (32-count*8)
        for byte <- 0 until count do output(offset+byte)=(bits >>> (byte*8)).toByte
        offset+=count
      output

  private[worker] def pythonRandomBytes(seed: Long,length: Int): Array[Byte]=PythonRandom(seed).bytes(length)

  def maskedImage(image: Tensor[Float],mask: Tensor[Float],cancelled: () => Boolean): Tensor[Float] =
    if cancelled() || Thread.currentThread().isInterrupted then
      throw new java.util.concurrent.CancellationException("masked image preparation cancelled")
    val dims=image.shape.dims
    require(image.device==Device.Cpu && image.dtype==DType.Float32 && dims.size==4 &&
      dims(0)==1 && dims(1)==3 && dims(2)>0 && dims(3)>0,"masked image must be single-image NCHW RGB Float32")
    require(mask.device==Device.Cpu && mask.dtype==DType.Float32 &&
      mask.shape==Shape.from(Seq(1L,1L,dims(2),dims(3))),"masked image mask shape/device mismatch")
    val pixels=(dims(2)*dims(3)).toInt
    val source=image.toFloatArray
    val levels=mask.toFloatArray
    require(source.forall(x => x.isFinite && x>= -1f && x<=1f),"masked image pixels must be finite and in [-1,1]")
    require(levels.forall(x => x.isFinite && x>=0f && x<=1f),"masked image mask must be finite and in [0,1]")
    val output=new Array[Float](source.length)
    for y <- 0 until dims(2).toInt do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("masked image preparation cancelled")
      for x <- 0 until dims(3).toInt do
        val pixel=y*dims(3).toInt+x
        if levels(pixel)<.5f then
          for channel <- 0 until 3 do output(channel*pixels+pixel)=source(channel*pixels+pixel)
    Tensor.fromFloats(output,image.shape)

  final case class Prepared(image: Tensor[Float],originalImage: Tensor[Float],mask: Tensor[Float],
      pasteBox: Option[(Int,Int,Int,Int)]=None,fullOriginal: Option[BufferedImage]=None,
      fullOverlayMask: Option[BufferedImage]=None):
    def pasteBack(generated: BufferedImage,cancelled: () => Boolean=() => false): BufferedImage = pasteBox match
      case None => generated
      case Some((x1,y1,x2,y2)) =>
        val original=fullOriginal.getOrElse(throw IllegalStateException("full-resolution original image missing"))
        val mask=fullOverlayMask.getOrElse(throw IllegalStateException("full-resolution overlay mask missing"))
        require(generated.getWidth>0 && generated.getHeight>0 &&
          mask.getWidth==original.getWidth && mask.getHeight==original.getHeight,"invalid full-resolution inpaint result")
        val patch=DiffusionImageResize(generated,x2-x1,y2-y1,0)
        val output=new BufferedImage(original.getWidth,original.getHeight,BufferedImage.TYPE_INT_RGB)
        val pixels=new Array[Int](original.getWidth*original.getHeight)
        for y <- 0 until original.getHeight; x <- 0 until original.getWidth do
          if x==0 && (cancelled() || Thread.currentThread().isInterrupted) then
            throw new java.util.concurrent.CancellationException("inpainting paste-back cancelled")
          val base=original.getRGB(x,y)
          val edited=if x>=x1 && x<x2 && y>=y1 && y<y2 then patch.getRGB(x-x1,y-y1) else base
          val amount=mask.getRGB(x,y)&255
          def mix(shift: Int): Int=((((base>>>shift)&255)*(255-amount)+((edited>>>shift)&255)*amount+127)/255)
          pixels(y*original.getWidth+x)=0xff000000|(mix(16)<<16)|(mix(8)<<8)|mix(0)
        output.setRGB(0,0,original.getWidth,original.getHeight,pixels,0,original.getWidth)
        output

  def sourceForVae(prepared: Prepared,factor: Int,cancelled: () => Boolean): (Tensor[Float],Tensor[Float]) =
    val dims=prepared.image.shape.dims
    require(factor>0 && dims.size==4 && dims(0)==1 && dims(1)==3,"invalid VAE source")
    val sourceHeight=dims(2).toInt;val sourceWidth=dims(3).toInt
    val width=sourceWidth/factor*factor;val height=sourceHeight/factor*factor
    require(width>0 && height>0,"latent upscale source is smaller than the VAE scale")
    if width==sourceWidth && height==sourceHeight then (prepared.image,prepared.mask)
    else
      val image=DiffusionImageResize(DiffusionImageIO.toBufferedImage(prepared.image),width,height,0)
      val mask=prepared.mask.toFloatArray
      val resized=Array.tabulate(width*height) { i =>
        val x=i%width;val y=i/width
        val sx=((2L*x+1)*sourceWidth/(2L*width)).toInt
        val sy=((2L*y+1)*sourceHeight/(2L*height)).toInt
        mask(sy*sourceWidth+sx)
      }
      (rgbTensor(image,cancelled),Tensor.fromFloats(resized,Shape.from(Seq(1L,1L,height.toLong,width.toLong))))

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

  private[worker] def gaussianBytes(input: Array[Int],width: Int,height: Int,radius: Int,cancelled: () => Boolean): Array[Int] =
    if radius==0 then input
    else
      // Pillow's three fractional box passes, including byte rounding at every pass.
      // See Pillow BoxBlur.c; its license is in META-INF/licenses/pillow.txt.
      val sigma2=radius.toFloat*radius.toFloat/3f
      val length=math.sqrt(12.0*sigma2+1.0).toFloat
      val whole=math.floor((length-1.0)/2.0).toFloat
      val fraction=((2f*whole+1f)*(whole*(whole+1f)-3f*sigma2))/(6f*(sigma2-(whole+1f)*(whole+1f)))
      val boxRadius=(whole+fraction).toFloat
      val integerRadius=boxRadius.toInt
      val scale=1L<<24
      val weight=(scale.toFloat/(boxRadius*2f+1f)).toLong
      val edgeWeight=(scale-(integerRadius*2+1)*weight)/2
      var pixels=input
      def pass(horizontal: Boolean): Unit =
        val output=new Array[Int](pixels.length)
        val lines=if horizontal then height else width
        val count=if horizontal then width else height
        for line <- 0 until lines do
          if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
          def index(position: Int): Int =
            val at=math.max(0,math.min(count-1,position))
            if horizontal then line*width+at else at*width+line
          var sum=0L
          for position <- -integerRadius to integerRadius do sum+=pixels(index(position))
          for position <- 0 until count do
            val far=pixels(index(position-integerRadius-1))+pixels(index(position+integerRadius+1))
            output(index(position))=((sum*weight+far.toLong*edgeWeight+(1L<<23))>>24).toInt
            sum+=pixels(index(position+integerRadius+1))-pixels(index(position-integerRadius))
        pixels=output
      for _ <- 0 until 3 do pass(horizontal=true)
      for _ <- 0 until 3 do pass(horizontal=false)
      pixels

  private def blur(input: Array[Float],width: Int,height: Int,radius: Int,cancelled: () => Boolean): Array[Float] =
    if radius==0 then input
    else gaussianBytes(input.map(value => math.max(0,math.min(255,math.round(value*255f)))),
      width,height,radius,cancelled).map(_/255f)

  private def cropRegion(mask: Array[Float],width: Int,height: Int,pad: Int): Option[(Int,Int,Int,Int)] =
    var minX=width;var minY=height;var maxX= -1;var maxY= -1
    for y <- 0 until height; x <- 0 until width do
      if mask(y*width+x)>0f then
        minX=math.min(minX,x);minY=math.min(minY,y);maxX=math.max(maxX,x);maxY=math.max(maxY,y)
    if maxX<0 then None
    else Some((math.max(0,minX-pad),math.max(0,minY-pad),math.min(width,maxX+1+pad),math.min(height,maxY+1+pad)))

  private def expandCrop(box: (Int,Int,Int,Int),width: Int,height: Int,targetWidth: Int,targetHeight: Int): (Int,Int,Int,Int) =
    var (x1,y1,x2,y2)=box
    val cropWidth=x2-x1;val cropHeight=y2-y1
    if cropWidth.toDouble/cropHeight>targetWidth.toDouble/targetHeight then
      val desired=(cropWidth.toDouble/(targetWidth.toDouble/targetHeight)).toInt
      val diff=desired-cropHeight
      y1-=diff/2;y2+=diff-diff/2
      if y2>=height then { val d=y2-height;y2-=d;y1-=d }
      if y1<0 then { y2-=y1;y1=0 }
      if y2>=height then y2=height
    else
      val desired=(cropHeight.toDouble*(targetWidth.toDouble/targetHeight)).toInt
      val diff=desired-cropWidth
      x1-=diff/2;x2+=diff-diff/2
      if x2>=width then { val d=x2-width;x2-=d;x1-=d }
      if x1<0 then { x2-=x1;x1=0 }
      if x2>=width then x2=width
    (x1,y1,x2,y2)

  private def grayImage(values: Array[Float],width: Int,height: Int): BufferedImage =
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    val pixels=values.map { value =>
      val gray=math.max(0,math.min(255,(value*255f).round.toInt))
      0xff000000|(gray<<16)|(gray<<8)|gray
    }
    image.setRGB(0,0,width,height,pixels,0,width)
    image

  /** Pillow-compatible multiscale premultiplied-alpha fill. */
  private def blurredColorFill(image: BufferedImage,mask: BufferedImage,cancelled: () => Boolean): BufferedImage =
    require(image.getWidth==mask.getWidth && image.getHeight==mask.getHeight,"inpaint fill image/mask size mismatch")
    val width=image.getWidth;val height=image.getHeight;val count=width*height
    val pixels=image.getRGB(0,0,width,height,null,0,width)
    val levels=mask.getRGB(0,0,width,height,null,0,width)
    val premultiplied=Array.fill(4)(new Array[Int](count))
    def div255(value: Int): Int =
      val rounded=value+128
      ((rounded>>>8)+rounded)>>>8
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("inpainting blur-fill preparation cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val alpha=255-(levels(i)&255)
        premultiplied(0)(i)=div255(((pixels(i)>>>16)&255)*alpha)
        premultiplied(1)(i)=div255(((pixels(i)>>>8)&255)*alpha)
        premultiplied(2)(i)=div255((pixels(i)&255)*alpha)
        premultiplied(3)(i)=alpha
    val result=Array.fill(4)(new Array[Int](count))
    val radii=Vector((256,1),(64,1),(16,2),(4,4),(2,2),(0,1))
    for (radius,repeats) <- radii do
      val blurred=if radius==0 then premultiplied else premultiplied.map(channel => gaussianBytes(channel,width,height,radius,cancelled))
      for _ <- 0 until repeats do
        for y <- 0 until height do
          if cancelled() || Thread.currentThread().isInterrupted then
            throw new java.util.concurrent.CancellationException("inpainting blur-fill cancelled")
          for x <- 0 until width do
            val i=y*width+x
            val alpha=blurred(3)(i)
            if alpha>0 then
              val outAlpha255=alpha*255+result(3)(i)*(255-alpha)
              val foreground=alpha*255*255*128/outAlpha255
              val background=255*128-foreground
              for channel <- 0 until 3 do
                val value=if alpha==255 then blurred(channel)(i) else math.min(255,blurred(channel)(i)*255/alpha)
                val sum=value*foreground+result(channel)(i)*background+(128<<7)
                result(channel)(i)=(((sum>>>8)+sum)>>>8)>>>7
              val sum=outAlpha255+128
              result(3)(i)=((sum>>>8)+sum)>>>8
    val output=new Array[Int](count)
    for y <- 0 until height do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("inpainting blur-fill compositing cancelled")
      for x <- 0 until width do
        val i=y*width+x
        val amount=levels(i)&255
        def channel(index: Int,shift: Int): Int =
          val original=(pixels(i)>>>shift)&255
          div255(result(index)(i)*amount+original*(255-amount))
        output(i)=0xff000000|(channel(0,16)<<16)|(channel(1,8)<<8)|channel(2,0)
    val filled=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    filled.setRGB(0,0,width,height,output,0,width)
    filled

  private def applyFill(image: BufferedImage,mask: BufferedImage,mode: Int,seed: Long,
      cancelled: () => Boolean): BufferedImage =
    require(Set(0,1,2,3)(mode),"invalid inpainting fill mode")
    if mode==0 then blurredColorFill(image,mask,cancelled)
    else if mode==1 then image
    else
      require(image.getWidth==mask.getWidth && image.getHeight==mask.getHeight,"inpaint fill image/mask size mismatch")
      val width=image.getWidth;val height=image.getHeight
      val count=width*height
      val source=image.getRGB(0,0,width,height,null,0,width)
      val levels=mask.getRGB(0,0,width,height,null,0,width)
      val noise=if mode==2 then PythonRandom(seed).bytes(count*3,cancelled) else new Array[Byte](0)
      val output=new Array[Int](count)
      for y <- 0 until height do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("inpainting fill cancelled")
        for x <- 0 until width do
          val i=y*width+x
          if (levels(i)&255)>127 then
            if mode==2 then
              val j=i*3
              output(i)=0xff000000|((noise(j)&255)<<16)|((noise(j+1)&255)<<8)|(noise(j+2)&255)
            else output(i)=0xff7f7f7f
          else output(i)=source(i)
      val result=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
      result.setRGB(0,0,width,height,output,0,width)
      result

  def prepare(source: BufferedImage,mask: BufferedImage,width: Int,height: Int,resizeMode: Int,blurRadius: Int,
      invert: Boolean,cancelled: () => Boolean,fullRes: Boolean=false,fullResPadding: Int=32,
      fill: Int=1,seed: Long=0): Prepared =
    require(width>0 && height>0 && width.toLong*height<=16777216L,"invalid inpaint output size")
    require(blurRadius>=0 && blurRadius<=64,"mask_blur must be between 0 and 64")
    if cancelled() || Thread.currentThread().isInterrupted then throw new java.util.concurrent.CancellationException("inpainting cancelled")
    lazy val original=DiffusionImageResize(source,width,height,resizeMode)
    var maskRgb=grayRgb(mask)
    if invert then
      for y <- 0 until maskRgb.getHeight; x <- 0 until maskRgb.getWidth do
        val v=255-(maskRgb.getRGB(x,y)&255)
        maskRgb.setRGB(x,y,(v<<16)|(v<<8)|v)
    if maskRgb.getWidth!=source.getWidth || maskRgb.getHeight!=source.getHeight then
      maskRgb=DiffusionImageResize(maskRgb,source.getWidth,source.getHeight,0)
    val fullResPrepared=if fullRes then
      require(fullResPadding>=0 && fullResPadding<=256,"inpaint_full_res_padding must be between 0 and 256")
      val fullWidth=source.getWidth;val fullHeight=source.getHeight
      val fullPlane=fullWidth*fullHeight
      val fullLevels=Array.tabulate(fullPlane) { i => (maskRgb.getRGB(i%fullWidth,i/fullWidth)&255)/255f }
      val blurredFull=blur(fullLevels,fullWidth,fullHeight,blurRadius,cancelled)
      val fullMaskImage=grayImage(blurredFull,fullWidth,fullHeight)
      val box=cropRegion(blurredFull,fullWidth,fullHeight,fullResPadding)
        .map(expandCrop(_,fullWidth,fullHeight,width,height))
      box.map { (x1,y1,x2,y2) =>
        val imageCrop=source.getSubimage(x1,y1,x2-x1,y2-y1)
        val maskCrop=fullMaskImage.getSubimage(x1,y1,x2-x1,y2-y1)
        val resizedImage=DiffusionImageResize(imageCrop,width,height,2)
        val resizedMask=DiffusionImageResize(maskCrop,width,height,2)
        val cropPlane=width*height
        val cropMask=Array.tabulate(cropPlane) { i => (resizedMask.getRGB(i%width,i/width)&255)/255f }
        val modelMaskImage=grayImage(cropMask,width,height)
        val modelImage=rgbTensor(applyFill(resizedImage,modelMaskImage,fill,seed,cancelled),cancelled)
        val originalImage=rgbTensor(resizedImage,cancelled)
        val modelMask=Tensor.fromFloats(cropMask,Shape.from(Seq(1L,1L,height.toLong,width.toLong)))
        val fullCopy=new BufferedImage(fullWidth,fullHeight,BufferedImage.TYPE_INT_RGB)
        val fullPixels=source.getRGB(0,0,fullWidth,fullHeight,null,0,fullWidth)
        fullCopy.setRGB(0,0,fullWidth,fullHeight,fullPixels,0,fullWidth)
        Prepared(modelImage,originalImage,modelMask,Some((x1,y1,x2,y2)),Some(fullCopy),Some(fullMaskImage))
      }
    else None
    fullResPrepared.getOrElse {
      maskRgb=DiffusionImageResize(maskRgb,width,height,resizeMode)
      val plane=width*height
      val levels=Array.tabulate(plane) { i => (maskRgb.getRGB(i%width,i/width)&255)/255f }
      val softened=blur(levels,width,height,blurRadius,cancelled)
      val modelMaskImage=grayImage(softened,width,height)
      val modelImage=rgbTensor(applyFill(original,modelMaskImage,fill,seed,cancelled),cancelled)
      val originalImage=rgbTensor(original,cancelled)
      Prepared(modelImage,originalImage,Tensor.fromFloats(softened,Shape.from(Seq(1L,1L,height.toLong,width.toLong))))
    }
