package scalatorch.models.worker

import scalatorch.*
import java.awt.image.BufferedImage
import javax.imageio.{ImageIO,IIOImage,ImageWriteParam}

/** Single-image SD1.x cheap approximation, matching NUI's latent channel factors. */
object NativePreview:
  private val factors=Array(Array(.298f,.207f,.208f),Array(.187f,.286f,.173f),
    Array(-.158f,.189f,.264f),Array(-.184f,-.271f,-.473f))
  def rgb(latent: Tensor[Float]): BufferedImage=
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

  def jpegBase64(latent: Tensor[Float]): String=
    val source=rgb(latent)
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
