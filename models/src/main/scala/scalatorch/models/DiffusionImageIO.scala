package scalatorch.models

import scalatorch.*
import java.awt.image.BufferedImage
import java.nio.file.{Files,Path,StandardOpenOption}
import javax.imageio.ImageIO
import scala.util.Using

/** Raw VAE [-1,1] RGB images to PNG. Caller owns staging/publication and parent-directory creation. */
object DiffusionImageIO:
  def toBufferedImage(images: Tensor[Float],index: Int=0,maxPixels: Long=16777216L): BufferedImage =
    val dims=images.shape.dims
    require(images.device==Device.Cpu && images.dtype==DType.Float32 && dims.size==4 && dims(1)==3 &&
      dims(2)>0 && dims(3)>0 && index>=0 && index<dims(0),"PNG export requires a valid NCHW RGB image")
    val pixels=BigInt(dims(2))*dims(3)
    require(maxPixels>=0 && pixels<=maxPixels && pixels*3<=Int.MaxValue,"PNG exceeds allocation limit")
    val values=images.slice(0,index,index+1L).toFloatArray
    require(values.forall(_.isFinite),"PNG image has nonfinite values")
    val width=dims(3).toInt;val height=dims(2).toInt;val plane=pixels.toInt
    val image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB)
    def channel(v: Float): Int =
      val normalized=math.max(0f,math.min(1f,v*.5f+.5f))
      math.rint((normalized*255f).toDouble).toInt
    var i=0
    while i<plane do
      image.setRGB(i%width,i/width,(channel(values(i))<<16)|(channel(values(plane+i))<<8)|channel(values(2*plane+i)))
      i+=1
    image

  def writePng(images: Tensor[Float],path: Path,index: Int=0,maxPixels: Long=16777216L): Unit =
    writePngImage(toBufferedImage(images,index,maxPixels),path)

  def writePngImage(image: BufferedImage,path: Path): Unit =
    // Open outside the cleanup block so an existing caller-owned file can never be deleted.
    val stream=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try Using.resource(stream) { output => require(ImageIO.write(image,"png",output),"PNG writer unavailable") }
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
