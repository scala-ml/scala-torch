package scalatorch.models

import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import java.io.RandomAccessFile
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.util.Using

/** Bounded ImageIO decoding, EXIF orientation and NUI RGB/white-alpha conversion. */
object DiffusionImageInput:
  private def tiffOrientation(bytes: Array[Byte]): Int =
    require(bytes.length>=8,"truncated EXIF header")
    val little=bytes(0)==73 && bytes(1)==73
    require(little || (bytes(0)==77 && bytes(1)==77),"invalid EXIF byte order")
    val data=ByteBuffer.wrap(bytes).order(if little then ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN)
    require((data.getShort(2)&65535)==42,"invalid EXIF TIFF marker")
    val offset=Integer.toUnsignedLong(data.getInt(4))
    require(offset>=8 && offset+2<=bytes.length,"EXIF directory outside metadata budget")
    val count=data.getShort(offset.toInt)&65535
    require(offset+2+count.toLong*12<=bytes.length,"truncated EXIF directory")
    var result=1
    for i <- 0 until count do
      val start=offset.toInt+2+i*12
      if (data.getShort(start)&65535)==274 then
        require((data.getShort(start+2)&65535)==3 && data.getInt(start+4)==1,"invalid EXIF orientation field")
        val value=data.getShort(start+8)&65535
        if value>=1 && value<=8 then result=value
    result

  private def orientation(path: Path): Int = Using.resource(new RandomAccessFile(path.toFile,"r")) { file =>
    val length=file.length()
    def bytes(size: Int): Array[Byte] =
      require(size>=0 && size<=1048576 && file.getFilePointer+size<=length,"image metadata exceeds bounds")
      val data=new Array[Byte](size);file.readFully(data);data
    if length<8 then 1
    else
      val signature=bytes(8)
      if (signature(0)&255)==255 && (signature(1)&255)==216 then
        file.seek(2)
        var found=1;var done=false
        while !done && file.getFilePointer<length do
          require(file.readUnsignedByte()==255,"invalid JPEG marker")
          var marker=file.readUnsignedByte()
          while marker==255 do marker=file.readUnsignedByte()
          if marker==218 || marker==217 then done=true
          else if marker!=1 && !(marker>=208 && marker<=215) then
            val size=file.readUnsignedShort()-2
            require(size>=0 && file.getFilePointer+size<=length,"truncated JPEG segment")
            if marker==225 then
              val block=bytes(size)
              if block.length>=6 && block.take(6).sameElements(Array[Byte](69,120,105,102,0,0)) then
                found=tiffOrientation(block.drop(6));done=true
            else file.seek(file.getFilePointer+size)
        found
      else if signature.sameElements(Array[Byte](-119,80,78,71,13,10,26,10)) then
        var found=1;var done=false
        while !done && file.getFilePointer+12<=length do
          val size=Integer.toUnsignedLong(file.readInt())
          val kind=file.readInt()
          require(file.getFilePointer+size+4<=length,"truncated PNG chunk")
          if kind==0x65584966 then
            require(size<=1048576,"EXIF exceeds metadata limit")
            found=tiffOrientation(bytes(size.toInt));done=true
          else if kind==0x49454e44 then done=true
          else file.seek(file.getFilePointer+size+4)
        found
      else if (signature(0)==73 && signature(1)==73) || (signature(0)==77 && signature(1)==77) then
        file.seek(0);tiffOrientation(bytes(math.min(length,1048576).toInt))
      else 1
  }

  def readRgb(path: Path,maxPixels: Long=16777216L,maxBytes: Long=268435456L): BufferedImage =
    require(maxPixels>=0 && maxBytes>=0 && Files.isRegularFile(path),"invalid input image path or limits")
    require(Files.size(path)<=maxBytes,"encoded image exceeds byte limit")
    val turn=orientation(path)
    val stream=ImageIO.createImageInputStream(path.toFile)
    require(stream!=null,"image stream unavailable")
    Using.resource(stream) { input =>
      val readers=ImageIO.getImageReaders(input)
      require(readers.hasNext,"unsupported or invalid image format")
      val reader=readers.next()
      try
        reader.setInput(input,false,false)
        val width=reader.getWidth(0);val height=reader.getHeight(0)
        require(width>0 && height>0 && width.toLong*height<=math.min(maxPixels,Int.MaxValue.toLong),"decoded image exceeds pixel limit")
        val source=reader.read(0)
        val output=new BufferedImage(if turn>=5 then height else width,if turn>=5 then width else height,BufferedImage.TYPE_INT_RGB)
        val gray=source.getColorModel.getColorSpace.getType==ColorSpace.TYPE_GRAY
        for y <- 0 until height; x <- 0 until width do
          val pixel=if gray then
            val level=math.min(255,source.getRaster.getSample(x,y,0))
            0xff000000 | (level<<16) | (level<<8) | level
          else source.getRGB(x,y)
          val alpha=(pixel >>> 24)&255
          var rgb=0
          for shift <- Seq(16,8,0) do
            val channel=(((pixel >>> shift)&255)*alpha+255*(255-alpha)+127)/255
            rgb|=channel << shift
          val (dx,dy)=turn match
            case 2 => (width-1-x,y)
            case 3 => (width-1-x,height-1-y)
            case 4 => (x,height-1-y)
            case 5 => (y,x)
            case 6 => (height-1-y,x)
            case 7 => (height-1-y,width-1-x)
            case 8 => (y,width-1-x)
            case _ => (x,y)
          output.setRGB(dx,dy,rgb)
        output
      finally reader.dispose()
    }
