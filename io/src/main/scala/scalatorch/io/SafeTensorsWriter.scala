package scalatorch.io

import scalatorch.{Shape,Tensor}
import com.fasterxml.jackson.core.JsonFactory
import java.io.ByteArrayOutputStream
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files,Path,StandardCopyOption,StandardOpenOption}
import java.util.concurrent.CancellationException

/** Explicit on-disk dtype: half and bfloat inputs are held as Float32 in memory. */
sealed trait SafeTensorWrite:
  def shape: Shape
  def dtype: SafeDType

object SafeTensorWrite:
  /** Exact on-disk tensor data; caller retains ownership and must not mutate during writing. */
  final case class Raw(dtype: SafeDType,shape: Shape,bytes: Array[Byte]) extends SafeTensorWrite:
    require(dtype!=null && shape!=null && bytes!=null &&
      BigInt(shape.numel)*dtype.bytes==bytes.length,"raw tensor byte length mismatch")
  final case class F32(tensor: Tensor[Float]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.F32
  final case class F16(tensor: Tensor[Float]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.F16
  final case class BF16(tensor: Tensor[Float]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.BF16
  final case class F64(tensor: Tensor[Double]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.F64
  final case class I64(tensor: Tensor[Long]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.I64
  final case class Bool(tensor: Tensor[Boolean]) extends SafeTensorWrite:
    def shape= tensor.shape
    def dtype=SafeDType.BOOL

/** Header metadata plus a deferred tensor materializer, invoked once in sorted key order. */
final case class SafeTensorPlan(dtype: SafeDType,shape: Shape,load: () => SafeTensorWrite)

/** Deterministic safetensors output, staged beside the destination before publication. */
object SafeTensorsWriter:
  private val factory=new JsonFactory()
  private def unicode(value: String): Boolean =
    if value==null then false else
      var i=0;var valid=true
      while i<value.length && valid do
        val c=value.charAt(i)
        if Character.isHighSurrogate(c) then
          if i+1>=value.length || !Character.isLowSurrogate(value.charAt(i+1)) then valid=false else i+=1
        else if Character.isLowSurrogate(c) then valid=false
        i+=1
      valid
  private def check(cancelled: () => Boolean): Unit =
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("safetensors write cancelled")
  private def bfloat(value: Float): Short =
    val bits=java.lang.Float.floatToRawIntBits(value)
    if (bits&0x7f800000)==0x7f800000 && (bits&0x007fffff)!=0 then ((bits>>>16)|0x40).toShort
    else ((bits+0x7fff+((bits>>>16)&1))>>>16).toShort

  def write(path: Path,tensors: Map[String,SafeTensorWrite],metadata: Map[String,String]=Map.empty,
      overwrite: Boolean=false,cancelled: () => Boolean=() => false,limits: ReadLimits=ReadLimits()): Unit =
    require(tensors!=null,"invalid safetensors tensors")
    writePlanned(path,tensors.map { (name,value) =>
      require(value!=null,"invalid tensor value")
      name->SafeTensorPlan(value.dtype,value.shape,()=>value)
    },metadata,overwrite,cancelled,limits)

  /** Materializes only the current tensor and publishes after all planned entries succeed. */
  def writePlanned(path: Path,tensors: Map[String,SafeTensorPlan],metadata: Map[String,String]=Map.empty,
      overwrite: Boolean=false,cancelled: () => Boolean=() => false,limits: ReadLimits=ReadLimits(),
      beforePublish: () => Unit=() => ()): Unit =
    require(path!=null && tensors!=null && metadata!=null && cancelled!=null && limits!=null && beforePublish!=null,"invalid safetensors write arguments")
    require(tensors.size<=limits.maxTensors,"too many safetensors entries")
    require(tensors.forall((name,value)=>unicode(name) && name.nonEmpty && name!="__metadata__" && value!=null &&
      value.dtype!=null && value.shape!=null && value.load!=null),"invalid tensor name or plan")
    require(metadata.forall((key,value)=>unicode(key) && unicode(value)),"invalid safetensors metadata")
    val entries=tensors.toVector.sortBy(_._1)
    val descriptions=entries.map { (name,value) =>
      require(value.shape.rank<=limits.maxRank,"safetensors rank exceeds limit")
      val size=try Math.multiplyExact(value.shape.numel,value.dtype.bytes.toLong)
        catch case _: ArithmeticException => throw IllegalArgumentException("safetensors payload size overflow")
      require(size<=limits.maxTensorBytes,"safetensors tensor exceeds byte limit")
      (name,value,size)
    }
    val headerBytes=new ByteArrayOutputStream()
    val bounded=new java.io.OutputStream:
      override def write(value: Int): Unit =
        require(headerBytes.size().toLong+1<=limits.maxHeaderBytes,"safetensors header exceeds byte limit")
        headerBytes.write(value)
      override def write(bytes: Array[Byte],offset: Int,length: Int): Unit =
        require(headerBytes.size().toLong+length<=limits.maxHeaderBytes,"safetensors header exceeds byte limit")
        headerBytes.write(bytes,offset,length)
    val json=factory.createGenerator(bounded)
    try
      json.writeStartObject()
      if metadata.nonEmpty then
        json.writeObjectFieldStart("__metadata__")
        metadata.toVector.sortBy(_._1).foreach((key,value)=>json.writeStringField(key,value))
        json.writeEndObject()
      var offset=0L
      descriptions.foreach { (name,value,size) =>
        val end=try Math.addExact(offset,size) catch case _: ArithmeticException => throw IllegalArgumentException("safetensors payload size overflow")
        json.writeObjectFieldStart(name)
        json.writeStringField("dtype",value.dtype.toString)
        json.writeArrayFieldStart("shape");value.shape.dims.foreach(json.writeNumber);json.writeEndArray()
        json.writeArrayFieldStart("data_offsets");json.writeNumber(offset);json.writeNumber(end);json.writeEndArray()
        json.writeEndObject();offset=end
      }
      json.writeEndObject()
    finally json.close()
    val header=headerBytes.toByteArray
    val padding=(8-header.length%8)%8
    require(header.length.toLong+padding<=limits.maxHeaderBytes,"safetensors header exceeds byte limit")
    val target=path.toAbsolutePath.normalize()
    val parent=target.getParent
    require(parent!=null && Files.isDirectory(parent),"safetensors destination directory does not exist")
    if !overwrite && Files.exists(target) then throw new java.nio.file.FileAlreadyExistsException(target.toString)
    check(cancelled)
    val temp=Files.createTempFile(parent,".scalatorch-", ".safetensors.tmp")
    try
      val channel=FileChannel.open(temp,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)
      try
        def writeBuffer(buffer: ByteBuffer): Unit =
          buffer.flip()
          while buffer.hasRemaining do
            check(cancelled)
            if channel.write(buffer)<=0 then throw new java.io.IOException("safetensors write made no progress")
          buffer.clear()
        val prefix=ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);prefix.putLong(header.length.toLong+padding);writeBuffer(prefix)
        val encoded=ByteBuffer.allocate(header.length+padding);encoded.put(header);for _ <- 0 until padding do encoded.put(32.toByte)
        writeBuffer(encoded)
        val buffer=ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN)
        def emit[A](values: Array[A],width: Int)(append: A => Unit): Unit =
          var i=0
          while i<values.length do
            if buffer.remaining()<width then writeBuffer(buffer)
            append(values(i));i+=1
          if buffer.position()>0 then writeBuffer(buffer)
        def emitFloat(tensor: Tensor[Float],mode: Int): Unit =
          val storage=tensor.storage.data
          val layout=tensor.layout
          val contiguous=layout.isContiguous
          val offset=layout.offset.toInt
          val count=tensor.shape.numel.toInt
          val width=if mode==0 then 4 else 2
          var i=0
          while i<count do
            if buffer.remaining()<width then writeBuffer(buffer)
            val value=storage(if contiguous then offset+i else layout.index(i))
            if mode==0 then buffer.putFloat(value)
            else if mode==1 then buffer.putShort(java.lang.Float.floatToFloat16(value))
            else buffer.putShort(bfloat(value))
            i+=1
          if buffer.position()>0 then writeBuffer(buffer)
        descriptions.foreach { (_,plan,_) =>
          check(cancelled)
          val value=plan.load()
          require(value!=null && value.dtype==plan.dtype && value.shape==plan.shape,"materialized tensor differs from its plan")
          value match
            case SafeTensorWrite.Raw(_,_,bytes) => emit(bytes,1)(buffer.put)
            case SafeTensorWrite.F32(tensor) => emitFloat(tensor,0)
            case SafeTensorWrite.F16(tensor) => emitFloat(tensor,1)
            case SafeTensorWrite.BF16(tensor) => emitFloat(tensor,2)
            case SafeTensorWrite.F64(tensor) => emit(tensor.toDoubleArray,8)(x=>buffer.putDouble(x))
            case SafeTensorWrite.I64(tensor) => emit(tensor.toLongArray,8)(x=>buffer.putLong(x))
            case SafeTensorWrite.Bool(tensor) => emit(tensor.toBooleanArray,1)(x=>buffer.put(if x then 1.toByte else 0.toByte))
        }
      finally channel.close()
      check(cancelled)
      beforePublish()
      if overwrite then Files.move(temp,target,StandardCopyOption.REPLACE_EXISTING)
      else Files.move(temp,target)
    finally Files.deleteIfExists(temp)
