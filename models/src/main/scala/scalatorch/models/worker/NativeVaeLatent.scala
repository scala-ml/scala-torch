package scalatorch.models.worker

import scalatorch.*
import scalatorch.io.{ReadLimits,SafeDType,SafeTensorMemory,SafeTensors}
import java.awt.image.BufferedImage
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path,StandardOpenOption}
import javax.imageio.ImageIO
import scala.util.Using

/** NUI's four-byte-per-pixel latent layout shared by RGBA and CMYK containers. */
private[worker] object NativeVaeLatent:
  private val maxElements=16L*1024*1024
  final case class Quantized(width: Int,height: Int,bytes: Array[Byte],clipped: Int)

  private val maxExactBytes=maxElements*4+65536
  /** Uncompressed CMYK TIFF keeps the four quantized channels unchanged. */
  def writeCmykTiff(latent: Tensor[Float],path: Path): Quantized =
    val packed=packQuantized(latent)
    val tags=11;val bitsOffset=8+2+tags*12+4;val pixelOffset=bitsOffset+8
    val out=ByteBuffer.allocate(Math.addExact(pixelOffset,packed.bytes.length)).order(ByteOrder.LITTLE_ENDIAN)
    out.put('I'.toByte).put('I'.toByte).putShort(42.toShort).putInt(8).putShort(tags.toShort)
    def field(tag: Int,kind: Int,count: Int,value: Int): Unit =
      out.putShort(tag.toShort).putShort(kind.toShort).putInt(count).putInt(value)
    field(256,4,1,packed.width);field(257,4,1,packed.height)
    field(258,3,4,bitsOffset);field(259,3,1,1);field(262,3,1,5)
    field(273,4,1,pixelOffset);field(277,3,1,4);field(278,4,1,packed.height)
    field(279,4,1,packed.bytes.length);field(284,3,1,1);field(332,3,1,1)
    out.putInt(0)
    for _<-0 until 4 do out.putShort(8.toShort)
    out.put(packed.bytes)
    Files.write(path,out.array(),StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    packed

  def readCmykTiff(path: Path,channels: Int): Tensor[Float] =
    require(Files.isRegularFile(path) && Files.size(path)<=maxElements*4+65536,"latent TIFF is missing or too large")
    val input=ImageIO.createImageInputStream(path.toFile)
    require(input!=null,"TIFF input unavailable")
    Using.resource(input) { stream =>
      val readers=ImageIO.getImageReaders(stream)
      require(readers.hasNext,"invalid TIFF")
      val reader=readers.next()
      try
        require(Set("tif","tiff")(reader.getFormatName.toLowerCase(java.util.Locale.ROOT)),"latent image must be TIFF")
        reader.setInput(stream,false,false)
        val width=reader.getWidth(0);val height=reader.getHeight(0)
        require(width>0 && height>0 && width.toLong*height*4<=maxElements,"latent TIFF exceeds limit")
        val directory=javax.imageio.plugins.tiff.TIFFDirectory.createFromMetadata(reader.getImageMetadata(0))
        def scalar(tag: Int,default: Int): Int=Option(directory.getTIFFField(tag)).map(_.getAsInt(0)).getOrElse(default)
        val bits=directory.getTIFFField(258)
        require(scalar(262,-1)==5 && scalar(277,1)==4 && scalar(332,1)==1 && bits!=null &&
          bits.getCount==4 && (0 until 4).forall(i=>bits.getAsInt(i)==8),"latent TIFF must be 8-bit CMYK")
        val image=reader.read(0)
        require(image.getColorModel.getColorSpace.getType==java.awt.color.ColorSpace.TYPE_CMYK,"TIFF reader did not preserve CMYK")
        val raster=image.getRaster
        require(raster.getNumBands==4,"invalid CMYK channels")
        val bytes=new Array[Byte](width*height*4)
        val row=new Array[Int](width*4)
        for y<-0 until height do
          raster.getPixels(0,y,width,1,row)
          for i<-row.indices do bytes(y*row.length+i)=row(i).toByte
        unpackQuantized(width,height,bytes,channels)
      finally reader.dispose()
    }

  /** Reads NUI tensor files and converts saved decoder-input coordinates to diffusion coordinates. */
  def readSafetensors(path: Path,channels: Int): Tensor[Float] =
    readSafetensorsWithCoordinate(path,channels,false)._1

  def readSafetensorsWithCoordinate(path: Path,channels: Int,preserveDecoderInput: Boolean): (Tensor[Float],Boolean) =
    require(Files.isRegularFile(path) && Files.size(path)<=maxExactBytes,"latent safetensors missing or too large")
    Using.resource(SafeTensors.open(path,ReadLimits(maxHeaderBytes=65536,maxTensors=1,maxRank=4,maxTensorBytes=maxElements*8))) { archive =>
      require(archive.tensors.keySet==Set("tensor"),"expected a single latent tensor")
      val info=archive.tensors("tensor")
      require(info.shape.numel<=maxElements,"latent exceeds element limit")
      val latent=info.dtype match
        case SafeDType.F16 | SafeDType.BF16 | SafeDType.F32 => archive.readFloat("tensor")
        case SafeDType.F64 =>
          val value=archive.readDouble("tensor")
          Tensor.fromFloats(value.toDoubleArray.map(_.toFloat),value.shape)
        case _ => throw IllegalArgumentException("latent tensor must be floating point")
      validateExact(latent,channels)
      val metadata=archive.metadata.get("sdfx_vae_tools").map { value =>
        NativeJson.parse(value) match
          case NativeJson.Obj(fields) => fields
          case _ => throw IllegalArgumentException("invalid VAE provenance")
      }.getOrElse(Map.empty[String,NativeJson])
      metadata.getOrElse("coordinate",NativeJson.Str("diffusion_scaled")) match
        case NativeJson.Str("diffusion_scaled") => (latent,false)
        case NativeJson.Str("vae_decoder_input") if preserveDecoderInput => (latent,true)
        case NativeJson.Str("vae_decoder_input") =>
          val scale=metadata.get("scaling_factor") match
            case Some(NativeJson.Number(value)) if value.toDouble.isFinite && value>0 => value.toFloat
            case _ => throw IllegalArgumentException("decoder-input latent needs a positive scaling factor")
          require(scale.isFinite && scale>0,"invalid scaling factor")
          def vector(key: String): Option[Array[Float]]=metadata.get(key) match
            case None | Some(NativeJson.Null) => None
            case Some(NativeJson.Arr(values)) if values.size==channels => Some(values.map {
              case NativeJson.Number(value) if value.toFloat.isFinite => value.toFloat
              case _ => throw IllegalArgumentException(s"invalid $key")
            }.toArray)
            case _ => throw IllegalArgumentException(s"invalid $key")
          val mean=vector("latents_mean");val std=vector("latents_std")
          val shift=metadata.get("shift_factor") match
            case None | Some(NativeJson.Null) => 0f
            case Some(NativeJson.Number(value)) if value.toFloat.isFinite => value.toFloat
            case _ => throw IllegalArgumentException("invalid shift factor")
          val dims=latent.shape.dims;val plane=(dims(2)*dims(3)).toInt
          val values=latent.toFloatArray
          for i<-values.indices do
            val c=i/plane
            values(i)=(mean,std) match
              case (Some(m),Some(s)) =>
                require(s(c)!=0f,"invalid latent standard deviation")
                (values(i)-m(c))/s(c)*scale
              case _ => (values(i)-shift)*scale
          val result=Tensor.fromFloats(values,latent.shape)
          validateExact(result,channels)
          (result,false)
        case _ => throw IllegalArgumentException("unknown latent coordinate convention")
    }
  private def validateExact(latent: Tensor[Float],channels: Int): Unit =
    val dims=latent.shape.dims
    require(latent.device==Device.Cpu && latent.dtype==DType.Float32 && dims.size==4 && dims(0)==1 &&
      Set(4,16)(channels) && dims(1)==channels && dims(2)>0 && dims(3)>0 &&
      latent.shape.numel<=maxElements && latent.toFloatArray.forall(_.isFinite),"invalid exact latent")

  /** NUI's lossless L PNG stores a big-endian length followed by safetensors bytes. */
  def writeExactPng(latent: Tensor[Float],path: Path): Unit =
    val dims=latent.shape.dims
    require(dims.size==4 && Set(4L,16L)(dims(1)),"invalid latent shape")
    validateExact(latent,dims(1).toInt)
    val payload=SafeTensorMemory.writeFloat(latent)
    val count=Math.addExact(payload.length,4)
    val side=math.ceil(math.sqrt(count.toDouble)).toInt
    require(side.toLong*side<=maxExactBytes,"latent image exceeds limit")
    val pixels=new Array[Byte](side*side)
    ByteBuffer.wrap(pixels).order(ByteOrder.BIG_ENDIAN).putInt(payload.length).put(payload)
    val image=new BufferedImage(side,side,BufferedImage.TYPE_BYTE_GRAY)
    val row=new Array[Int](side)
    for y<-0 until side do
      for x<-0 until side do row(x)=pixels(y*side+x)&255
      image.getRaster.setPixels(0,y,side,1,row)
    val output=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try
      try require(ImageIO.write(image,"png",output),"PNG writer unavailable")
      finally output.close()
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error

  def readExactPng(path: Path,channels: Int): Tensor[Float] =
    require(Files.isRegularFile(path) && Files.size(path)<=maxExactBytes+1048576,"latent PNG is missing or too large")
    val header=Using.resource(Files.newInputStream(path))(_.readNBytes(29))
    require(header.length==29 && header.take(8).sameElements(Array[Byte](137.toByte,80,78,71,13,10,26,10)),"invalid latent PNG")
    val data=ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
    require(data.getInt(8)==13 && data.getInt(12)==0x49484452,"invalid PNG header")
    val width=data.getInt(16);val height=data.getInt(20)
    require(width>0 && height>0 && width.toLong*height<=maxExactBytes &&
      (header(24)&255)==8 && (header(25)&255)==0,"latent PNG must be bounded 8-bit grayscale")
    val image=Using.resource(Files.newInputStream(path))(ImageIO.read)
    require(image!=null && image.getWidth==width && image.getHeight==height,"invalid grayscale latent image")
    readExactImage(image,channels)

  def readExactImage(image: BufferedImage,channels: Int): Tensor[Float] =
    require(image!=null && image.getWidth>0 && image.getHeight>0 && image.getWidth.toLong*image.getHeight<=maxExactBytes &&
      image.getColorModel.getColorSpace.getType==java.awt.color.ColorSpace.TYPE_GRAY &&
      !image.getColorModel.hasAlpha && image.getRaster.getNumBands==1 && image.getSampleModel.getSampleSize.forall(_==8),
      "invalid grayscale latent image")
    val width=image.getWidth;val height=image.getHeight
    val pixels=new Array[Byte](width*height)
    val row=new Array[Int](width)
    for y<-0 until height do
      image.getRaster.getPixels(0,y,width,1,row)
      for x<-0 until width do pixels(y*width+x)=row(x).toByte
    require(pixels.length>=4,"truncated latent PNG")
    val size=ByteBuffer.wrap(pixels).order(ByteOrder.BIG_ENDIAN).getInt()&0xffffffffL
    require(size>=8 && size<=maxExactBytes && size<=pixels.length-4,"invalid latent payload length")
    val latent=SafeTensorMemory.readFloat(java.util.Arrays.copyOfRange(pixels,4,4+size.toInt),
      ReadLimits(maxHeaderBytes=65536,maxTensors=1,maxRank=4,maxTensorBytes=maxElements*4))
    validateExact(latent,channels)
    latent

  /** Writes a new RGBA PNG without premultiplying or replacing an existing file. */
  def writeRgbaPng(latent: Tensor[Float],path: Path): Quantized =
    val packed=packQuantized(latent)
    val image=new BufferedImage(packed.width,packed.height,BufferedImage.TYPE_4BYTE_ABGR)
    val raster=image.getRaster
    val row=new Array[Int](packed.width*4)
    for y<-0 until packed.height do
      for i<-row.indices do row(i)=packed.bytes(y*row.length+i)&255
      raster.setPixels(0,y,packed.width,1,row)
    val output=Files.newOutputStream(path,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE)
    try
      try require(ImageIO.write(image,"png",output),"PNG writer unavailable")
      finally output.close()
      packed
    catch
      case error: Throwable =>
        try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error

  /** Reads exact 8-bit RGBA channels without converting through RGB or dropping alpha. */
  def readRgbaPng(path: Path,channels: Int): Tensor[Float] =
    require(Files.isRegularFile(path) && Files.size(path)<=maxElements*4+65536,"latent PNG is missing or too large")
    val header=Using.resource(Files.newInputStream(path))(_.readNBytes(29))
    require(header.length==29 && header.take(8).sameElements(Array[Byte](137.toByte,80,78,71,13,10,26,10)),"invalid latent PNG")
    val data=ByteBuffer.wrap(header).order(ByteOrder.BIG_ENDIAN)
    require(data.getInt(8)==13 && data.getInt(12)==0x49484452,"invalid PNG header")
    val width=data.getInt(16);val height=data.getInt(20)
    require(width>0 && height>0 && BigInt(width)*height*4<=maxElements &&
      (header(24)&255)==8 && (header(25)&255)==6,"latent PNG must be bounded 8-bit RGBA")
    require(Set(4,16)(channels) && (channels==4 || (width%2==0 && height%2==0)),"invalid latent channels or grid")
    val image=Using.resource(Files.newInputStream(path))(ImageIO.read)
    require(image!=null && image.getWidth==width && image.getHeight==height &&
      image.getRaster.getNumBands==4 && image.getSampleModel.getSampleSize.forall(_==8) &&
      !image.isAlphaPremultiplied,"invalid RGBA latent image")
    val bytes=new Array[Byte](width*height*4)
    val row=new Array[Int](width*4)
    for y<-0 until height do
      image.getRaster.getPixels(0,y,width,1,row)
      for i<-row.indices do bytes(y*row.length+i)=row(i).toByte
    unpackQuantized(width,height,bytes,channels)

  def packQuantized(latent: Tensor[Float]): Quantized =
    val dims=latent.shape.dims
    require(latent.device==Device.Cpu && latent.dtype==DType.Float32 && dims.size==4 && dims(0)==1 &&
      Set(4L,16L)(dims(1)) && dims(2)>0 && dims(3)>0 && latent.shape.numel<=maxElements,
      "latent must be one bounded CPU Float32 image with 4 or 16 channels")
    val values=latent.toFloatArray
    require(values.forall(_.isFinite),"latent contains nonfinite values")
    val channels=dims(1).toInt;val height=dims(2).toInt;val width=dims(3).toInt
    val grid=if channels==16 then 2 else 1
    val outWidth=Math.multiplyExact(width,grid);val outHeight=Math.multiplyExact(height,grid)
    val bytes=new Array[Byte](Math.multiplyExact(Math.multiplyExact(outWidth,outHeight),4))
    var clipped=0
    for c<-0 until channels; y<-0 until height; x<-0 until width do
      val value=values((c*height+y)*width+x)
      if math.abs(value)>10f then clipped+=1
      val scaled=(value+10f)*12.75f
      val quantized=math.max(0d,math.min(255d,math.rint(scaled.toDouble))).toInt
      val tile=c/4
      val destination=(((tile/2)*height+y)*outWidth+(tile%2)*width+x)*4+c%4
      bytes(destination)=quantized.toByte
    Quantized(outWidth,outHeight,bytes,clipped)

  def unpackQuantized(width: Int,height: Int,bytes: Array[Byte],channels: Int): Tensor[Float] =
    require(width>0 && height>0 && Set(4,16)(channels) && bytes!=null,
      "invalid latent image dimensions or channels")
    val grid=if channels==16 then 2 else 1
    require(width%grid==0 && height%grid==0,"latent grid dimensions are invalid")
    val byteCount=BigInt(width)*height*4
    require(byteCount<=maxElements && byteCount==bytes.length,"latent image byte count exceeds its limit or is invalid")
    val latentWidth=width/grid;val latentHeight=height/grid
    val result=new Array[Float](byteCount.toInt)
    for c<-0 until channels; y<-0 until latentHeight; x<-0 until latentWidth do
      val tile=c/4
      val source=(((tile/2)*latentHeight+y)*width+(tile%2)*latentWidth+x)*4+c%4
      result((c*latentHeight+y)*latentWidth+x)=((bytes(source)&255).toFloat/255f)*20f-10f
    Tensor.fromFloats(result,Shape.from(Seq(1,channels,latentHeight,latentWidth)))
