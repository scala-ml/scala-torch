package scalatorch.io

import scalatorch.*
import java.io.{EOFException, IOException}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Path, StandardOpenOption}

enum SafeDType(val bytes: Int):
  case F16 extends SafeDType(2)
  case BF16 extends SafeDType(2)
  case F32 extends SafeDType(4)
  case F64 extends SafeDType(8)
  case I64 extends SafeDType(8)
  case I32 extends SafeDType(4)
  case BOOL extends SafeDType(1)

final case class TensorInfo(dtype: SafeDType, shape: Shape, begin: Long, end: Long)
final case class ReadLimits(maxHeaderBytes: Int = 100000000, maxTensors: Int = 100000, maxRank: Int = 64,
    maxTensorBytes: Long = 1L << 30):
  require(maxHeaderBytes >= 2 && maxTensors >= 0 && maxRank >= 0 && maxTensorBytes >= 0, "invalid safetensors limits")

/** Metadata is indexed at open; payloads are copied on demand. Do not modify an open archive externally. */
final class SafeTensors private (
    channel: FileChannel, dataStart: Long, val tensors: Map[String, TensorInfo],
    val metadata: Map[String, String], limits: ReadLimits
) extends AutoCloseable:
  private def entry(name: String, allowed: Set[SafeDType], outputBytes: Int): TensorInfo =
    if !channel.isOpen then throw IllegalStateException("safetensors archive is closed")
    val info = tensors.getOrElse(name, throw IllegalArgumentException(s"tensor not found: $name"))
    require(allowed(info.dtype), s"dtype ${info.dtype} is unsupported by this typed read")
    require(info.shape.numel <= Int.MaxValue && Math.multiplyExact(info.shape.numel, outputBytes.toLong) <= limits.maxTensorBytes,
      "tensor materialization exceeds configured allocation limit")
    info

  private def chunks(info: TensorInfo)(decode: (ByteBuffer, Int) => Unit): Unit =
    val buffer = ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN)
    var consumed = 0L
    val total = info.end - info.begin
    while consumed < total do
      val n = math.min(buffer.capacity().toLong, total - consumed).toInt
      buffer.clear().limit(n)
      SafeTensors.readFully(channel, buffer, dataStart + info.begin + consumed)
      buffer.flip()
      decode(buffer, (consumed / info.dtype.bytes).toInt)
      consumed += n

  // Fresh arrays never escape before ownership passes to Tensor; no second full-size copy needed.
  private def owned[A](values: Array[A], shape: Shape, dtype: DType[A]): Tensor[A] =
    new Tensor(new Storage(values, dtype), Layout.contiguous(shape, values.length))

  /** Exact on-disk bytes, including integer dtypes unsupported by typed tensors. */
  def readRaw(name: String): Array[Byte] = synchronized {
    if !channel.isOpen then throw IllegalStateException("safetensors archive is closed")
    val info=tensors.getOrElse(name,throw IllegalArgumentException(s"tensor not found: $name"))
    val size=info.end-info.begin
    require(size<=limits.maxTensorBytes && size<=Int.MaxValue,"raw tensor exceeds configured allocation limit")
    val data=new Array[Byte](size.toInt)
    SafeTensors.readFully(channel,ByteBuffer.wrap(data),dataStart+info.begin)
    data
  }

  def readFloat(name: String): Tensor[Float] = synchronized {
    val info = entry(name, Set(SafeDType.F32, SafeDType.F16, SafeDType.BF16), 4)
    val out = new Array[Float](info.shape.numel.toInt)
    chunks(info) { (buffer, start) =>
      var i = start
      while buffer.hasRemaining do
        out(i) = info.dtype match
          case SafeDType.F32 => buffer.getFloat()
          case SafeDType.F16 => java.lang.Float.float16ToFloat(buffer.getShort())
          case SafeDType.BF16 => java.lang.Float.intBitsToFloat((buffer.getShort().toInt & 0xffff) << 16)
          case _ => throw IllegalStateException("invalid float decoder")
        i += 1
    }
    owned(out, info.shape, DType.Float32)
  }
  def readDouble(name: String): Tensor[Double] = synchronized {
    val info = entry(name, Set(SafeDType.F64), 8)
    val out = new Array[Double](info.shape.numel.toInt)
    chunks(info) { (buffer, start) =>
      var i = start
      while buffer.hasRemaining do { out(i) = buffer.getDouble(); i += 1 }
    }
    owned(out, info.shape, DType.Float64)
  }
  def readLong(name: String): Tensor[Long] = synchronized {
    val info = entry(name, Set(SafeDType.I64), 8)
    val out = new Array[Long](info.shape.numel.toInt)
    chunks(info) { (buffer, start) =>
      var i = start
      while buffer.hasRemaining do { out(i) = buffer.getLong(); i += 1 }
    }
    owned(out, info.shape, DType.Int64)
  }
  def readBoolean(name: String): Tensor[Boolean] = synchronized {
    val info = entry(name, Set(SafeDType.BOOL), 1)
    val out = new Array[Boolean](info.shape.numel.toInt)
    chunks(info) { (buffer, start) =>
      var i = start
      while buffer.hasRemaining do { out(i) = buffer.get() != 0; i += 1 }
    }
    owned(out, info.shape, DType.Bool)
  }
  override def close(): Unit = synchronized { channel.close() }

object SafeTensors:
  private[io] def readFully(channel: FileChannel, buffer: ByteBuffer, position: Long): Unit =
    var offset = position
    while buffer.hasRemaining do
      val n = channel.read(buffer, offset)
      if n < 0 then throw EOFException("truncated safetensors file")
      if n == 0 then throw IOException("safetensors file read made no progress")
      offset += n

  def open(path: Path, limits: ReadLimits = ReadLimits()): SafeTensors =
    val channel = FileChannel.open(path, StandardOpenOption.READ)
    try
      val size = channel.size()
      require(size >= 8, "safetensors file lacks header length")
      val prefix = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
      readFully(channel, prefix, 0)
      prefix.flip()
      val length = prefix.getLong()
      require(length >= 2 && length <= limits.maxHeaderBytes && length <= size - 8, "invalid or excessive header length")
      val buffer = ByteBuffer.allocate(length.toInt)
      readFully(channel, buffer, 8)
      val start = 8 + length
      val header = SafeTensorsHeader.parse(buffer.array(), size - start, limits)
      new SafeTensors(channel, start, header.tensors, header.metadata, limits)
    catch case error: Throwable =>
      try channel.close()
      catch case cleanup: Throwable => error.addSuppressed(cleanup)
      error match
        case e: IOException => throw IllegalArgumentException("invalid safetensors header", e)
        case e: ArithmeticException => throw IllegalArgumentException("safetensors metadata overflow", e)
        case _ => throw error
