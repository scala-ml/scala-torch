package scalatorch.models.worker

import scalatorch.*
import java.awt.image.BufferedImage
import javax.imageio.{ImageIO,IIOImage,ImageWriteParam}

/** Single-image cheap approximation, matching the SD1 and SDXL latent channel factors. */
object NativePreview:
  private[worker] def emitSafely(context: NativeWorkContext,step: Int)(render: => String): Unit =
    try
      context.checkCancelled()
      val encoded=render
      context.checkCancelled()
      context.event("preview",Map("step"->NativeJson.Number(step),"jpeg_b64"->NativeJson.Str(encoded)))
    catch
      case e: java.util.concurrent.CancellationException => throw e
      case scala.util.control.NonFatal(e) =>
        context.checkCancelled()
        context.event("log",Map("level"->NativeJson.Str("warn"),
          "message"->NativeJson.Str(s"Preview unavailable: ${e.getMessage}")))
  private val sd15Factors=Array(Array(.298f,.207f,.208f),Array(.187f,.286f,.173f),
    Array(-.158f,.189f,.264f),Array(-.184f,-.271f,-.473f))
  private val sdxlFactors=Array(Array(.3448f,.4168f,.4395f),Array(-.1953f,-.0290f,.0250f),
    Array(.1074f,.0886f,-.0163f),Array(-.3730f,-.2499f,-.2088f))
  def rgb(latent: Tensor[Float],sdxl: Boolean = false): BufferedImage=
    val factors=if sdxl then sdxlFactors else sd15Factors
    val dims=latent.shape.dims
    require(latent.device==Device.Cpu && latent.dtype==DType.Float32 && dims.size==4 &&
      dims(0)==1 && dims(1)==4 && dims(2)>0 && dims(3)>0,"preview requires one CPU NCHW four-channel latent")
    val pixels=BigInt(dims(2))*dims(3)
    require(pixels<=16777216,"preview exceeds allocation limit")
    val values=latent.toFloatArray
    require(values.forall(_.isFinite),"preview has nonfinite values")
    val width=dims(3).toInt;val height=dims(2).toInt;val plane=pixels.toInt
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    for i <- 0 until plane do
      var packed=0
      for channel <- 0 until 3 do
        var sum=0f
        for l <- 0 until 4 do sum+=values(l*plane+i)*factors(l)(channel)
        val value=(math.max(0f,math.min(1f,(sum+1f)/2f))*255f+.5f).toInt
        packed=(packed<<8)|value
      image.setRGB(i%width,i/width,packed)
    image

  def jpegBase64(latent: Tensor[Float],sdxl: Boolean = false): String=
    jpegImageBase64(rgb(latent,sdxl))

  private[worker] def decodedRgb(decoded: Tensor[Float]): BufferedImage =
    val dims=decoded.shape.dims
    require(decoded.device==Device.Cpu && decoded.dtype==DType.Float32 && dims.size==4 &&
      dims(0)==1 && dims(1)==3 && dims(2)>0 && dims(3)>0,"TAESD preview requires one RGB tensor")
    val plane=Math.multiplyExact(dims(2),dims(3))
    require(plane<=16777216,"TAESD preview exceeds allocation limit")
    val values=decoded.toFloatArray
    require(values.forall(_.isFinite),"TAESD preview contains nonfinite values")
    val image=new BufferedImage(dims(3).toInt,dims(2).toInt,BufferedImage.TYPE_INT_RGB)
    for i<-0 until plane.toInt do
      var packed=0
      for c<-0 until 3 do
        val v=values(c*plane.toInt+i)
        packed=(packed<<8)|((math.max(-1f,math.min(1f,v))+1f)*127.5f).toInt
      image.setRGB(i%dims(3).toInt,i/dims(3).toInt,packed)
    image

  private[worker] def jpegImageBase64(source: BufferedImage): String=
    val scale=256.0/math.max(source.getWidth,source.getHeight)
    val width=math.max(1,math.rint(source.getWidth*scale).toInt)
    val height=math.max(1,math.rint(source.getHeight*scale).toInt)
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    val graphics=image.createGraphics()
    try
      graphics.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      graphics.drawImage(source,0,0,width,height,null)
    finally graphics.dispose()
    val bytes=new java.io.ByteArrayOutputStream()
    val stream=ImageIO.createImageOutputStream(bytes)
    val writer=ImageIO.getImageWritersByFormatName("jpeg").next()
    try
      writer.setOutput(stream)
      val settings=writer.getDefaultWriteParam
      settings.setCompressionMode(ImageWriteParam.MODE_EXPLICIT)
      settings.setCompressionQuality(.7f)
      writer.write(null,new IIOImage(image,null,null),settings)
      stream.flush()
      java.util.Base64.getEncoder.encodeToString(bytes.toByteArray)
    finally
      writer.dispose()
      stream.close()
