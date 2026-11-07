package scalatorch.models.worker

import java.awt.color.ColorSpace
import java.awt.image.{BufferedImage,IndexColorModel}
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** NUI pixel permutation compatibility; this is not cryptographic encryption. */
private[worker] object NativePixelScramble:
  private def digest(value: String): String =
    java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)))
  def passwordMarker(password: String): String = digest(digest(password)+"Encrypt")
  def verifyPassword(password: String,marker: String): Boolean =
    marker!=null && marker.matches("[0-9a-f]{64}") && MessageDigest.isEqual(
      passwordMarker(password).getBytes(StandardCharsets.US_ASCII),marker.getBytes(StandardCharsets.US_ASCII))
  private def indices(length: Int,key: String,version: Int): Array[Int] =
    val seed=digest(key);val doubled=seed+seed
    val values=Array.tabulate(length)(identity)
    for i<-values.indices do
      val target=(java.lang.Long.parseLong(doubled.substring(i%64,i%64+8),16)%(length-i)).toInt
      val source=if version==3 then length-i-1 else i
      val temp=values(source);values(source)=values(target);values(target)=temp
    values
  private def transform(image: BufferedImage,password: String,version: Int,reverse: Boolean): BufferedImage =
    require(image!=null && password!=null && password.nonEmpty && version>=1 && version<=3,"invalid pixel scramble input")
    val color=image.getColorModel
    val rgb= !color.isInstanceOf[IndexColorModel] && color.getColorSpace.getType==ColorSpace.TYPE_RGB &&
      color.getNumColorComponents==3 && Set(3,4)(color.getNumComponents) && color.getComponentSize.forall(b=>b==8 || b==16)
    val grayAlpha= !color.isInstanceOf[IndexColorModel] && color.getColorSpace.getType==ColorSpace.TYPE_GRAY &&
      color.hasAlpha && color.getNumComponents==2 && color.getComponentSize.forall(_==8)
    val legacy=reverse && version==1 && image.getRaster.getNumBands==1 &&
      (color.isInstanceOf[IndexColorModel] || color.getColorSpace.getType==ColorSpace.TYPE_GRAY &&
        color.getNumComponents==1 && color.getComponentSize.forall(b=>b==8 || b==16))
    require(rgb || grayAlpha || legacy,"unsupported pixel scramble image format")
    val width=image.getWidth;val height=image.getHeight
    require(width>0 && height>0 && width.toLong*height<=67174400L,"image exceeds pixel limit")
    val key=digest(password);val xs=indices(width,key,version);val ys=indices(height,digest(key),version)
    val positions=Array.tabulate(width*height)(identity)
    def swap(a: Int,b: Int): Unit =
      val tmp=positions(a);positions(a)=positions(b);positions(b)=tmp
    def xOrder=if reverse then width-1 to 0 by -1 else 0 until width
    def yOrder=if reverse then height-1 to 0 by -1 else 0 until height
    version match
      case 1 => for x<-xOrder; y<-yOrder do swap(y*width+x,ys(y)*width+xs(x))
      case 2 =>
        def rows(): Unit=for y<-yOrder; x<-0 until width do swap(y*width+x,ys(y)*width+x)
        def columns(): Unit=for x<-xOrder; y<-0 until height do swap(y*width+x,y*width+xs(x))
        if reverse then { columns();rows() } else { rows();columns() }
      case 3 =>
        for y<-0 until height; x<-0 until width do
          if reverse then positions(ys(y)*width+xs(x))=y*width+x
          else positions(y*width+x)=ys(y)*width+xs(x)
    val input=image.getRaster
    val output=input.createCompatibleWritableRaster(width,height)
    var element: Object=null
    for y<-0 until height; x<-0 until width do
      val source=positions(y*width+x)
      element=input.getDataElements(source%width,source/width,element)
      output.setDataElements(x,y,element)
    new BufferedImage(color,output,color.isAlphaPremultiplied,null)
  def encode(image: BufferedImage,password: String): BufferedImage = transform(image,password,3,false)
  def decode(image: BufferedImage,password: String,version: Int): BufferedImage = transform(image,password,version,true)
