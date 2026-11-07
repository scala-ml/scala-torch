package scalatorch.models.worker

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.util.Using

/** NUI-compatible protected PNGs. Pixel scrambling is not cryptographic encryption. */
private[worker] object NativeProtectedPng:
  final case class Decoded(image: BufferedImage,verified: Option[Boolean],scrambled: Boolean)
  def read(path: Path,password: Option[String],maxPixels: Long=16777216L): Decoded =
    require(maxPixels>0 && maxPixels<=67174400L,"invalid image pixel limit")
    require(Files.isRegularFile(path) && Files.size(path)<=268435456L,"image missing or too large")
    val signature=Using.resource(Files.newInputStream(path))(_.readNBytes(8))
    require(signature.sameElements(Array[Byte](-119,80,78,71,13,10,26,10)),"protected image must be PNG")
    val input=ImageIO.createImageInputStream(path.toFile)
    require(input!=null,"image stream unavailable")
    val image=Using.resource(input) { stream =>
      val readers=ImageIO.getImageReaders(stream)
      require(readers.hasNext,"invalid PNG")
      val reader=readers.next()
      try
        reader.setInput(stream,false,false)
        val width=reader.getWidth(0);val height=reader.getHeight(0)
        require(width>0 && height>0 && width.toLong*height<=maxPixels,"image exceeds pixel limit")
        reader.read(0)
      finally reader.dispose()
    }
    NativePreprocess.sourceParameters(path,"Encrypt") match
      case None => Decoded(image,None,false)
      case Some(tag) =>
        val version=Map("pixel_shuffle"->1,"pixel_shuffle_2"->2,"pixel_shuffle_3"->3).getOrElse(tag,
          throw IllegalArgumentException("unsupported image scramble marker"))
        val pass=password.filter(_.nonEmpty).getOrElse(throw IllegalArgumentException("image password required"))
        val verified=NativePreprocess.sourceParameters(path,"EncryptPwdSha").map(marker=>NativePixelScramble.verifyPassword(pass,marker))
        require(!verified.contains(false),"incorrect image password or invalid marker")
        Decoded(NativePixelScramble.decode(image,pass,version),verified,true)
  def write(image: BufferedImage,path: Path,password: String,texts: Vector[(String,String)]): Unit =
    require(!texts.exists((key,_)=>Set("Encrypt","EncryptPwdSha")(key)),"reserved image metadata")
    val scrambled=NativePixelScramble.encode(image,password)
    Files.createFile(path)
    try NativePreprocess.write(scrambled,path,texts++Vector("Encrypt"->"pixel_shuffle_3",
      "EncryptPwdSha"->NativePixelScramble.passwordMarker(password)))
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
