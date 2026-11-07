package scalatorch.io

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files,Path,StandardOpenOption}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import scala.util.control.NonFatal

/** Creates an exclusive edit stage while preserving the source header and unedited tensor bytes. */
object SafeTensorsEditCopy:
  private val maxFile=512L*1024*1024
  private val limits=ReadLimits(maxHeaderBytes=16*1024*1024,maxTensors=100000,maxRank=64,maxTensorBytes=maxFile)
  private def check(cancelled: () => Boolean): Unit =
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("embedding edit cancelled")
  private def identity(path: Path): BasicFileAttributes=Files.readAttributes(path,classOf[BasicFileAttributes])
  private def same(a: BasicFileAttributes,b: BasicFileAttributes): Boolean =
    a.size()==b.size() && a.lastModifiedTime()==b.lastModifiedTime() && a.creationTime()==b.creationTime() &&
      java.util.Objects.equals(a.fileKey(),b.fileKey())
  private def owns(path: Path,link: Path): Boolean =
    try link!=null && Files.isSameFile(path,link)
    catch case NonFatal(_) => false
  private def writeFully(channel: FileChannel,buffer: ByteBuffer,position: Long): Unit =
    var pos=position
    while buffer.hasRemaining do
      val count=channel.write(buffer,pos)
      if count<=0 then throw new java.io.IOException("embedding edit write made no progress")
      pos+=count
  private def half(value: Double): Short =
    val sign=if (java.lang.Double.doubleToRawLongBits(value)&Long.MinValue)!=0 then 0x8000 else 0
    val magnitude=math.abs(value)
    require(magnitude<65520.0,"embedding value outside dtype range")
    if magnitude<math.pow(2.0,-14) then
      (sign | math.rint(Math.scalb(magnitude,24)).toInt).toShort
    else
      var exponent=Math.getExponent(magnitude)
      var significand=math.rint(Math.scalb(magnitude,10-exponent)).toInt
      if significand==2048 then { exponent+=1;significand=1024 }
      require(exponent<=15,"embedding value outside dtype range")
      (sign | ((exponent+15)<<10) | (significand-1024)).toShort
  private def encoded(value: Double,dtype: SafeDType): ByteBuffer =
    require(value.isFinite,"embedding value must be finite")
    val result=ByteBuffer.allocate(dtype.bytes).order(ByteOrder.LITTLE_ENDIAN)
    dtype match
      case SafeDType.F64 => result.putDouble(value)
      case SafeDType.F32 =>
        val converted=value.toFloat
        require(converted.isFinite,"embedding value outside dtype range")
        result.putFloat(converted)
      case SafeDType.F16 =>
        result.putShort(half(value))
      case SafeDType.BF16 =>
        val converted=value.toFloat
        require(converted.isFinite,"embedding value outside dtype range")
        val bits=java.lang.Float.floatToRawIntBits(converted)
        val rounded=((bits+0x7fff+((bits>>>16)&1))>>>16)&0xffff
        require((rounded&0x7f80)!=0x7f80,"embedding value outside dtype range")
        result.putShort(rounded.toShort)
      case _ => throw IllegalArgumentException(s"unsupported editable dtype: $dtype")
    result.flip();result

  /** The caller owns the returned stage and any later backup, replacement and live reload. */
  def build(source: Path,destination: Path,vectors: Map[String,Vector[Vector[Double]]],expectedSha256: String,
      baseline: Option[Map[String,Vector[Vector[Double]]]]=None,cancelled: () => Boolean=() => false): Unit =
    require(source!=null && destination!=null && vectors!=null && vectors.nonEmpty && cancelled!=null,
      "invalid embedding edit arguments")
    require(expectedSha256!=null && expectedSha256.matches("(?i)[0-9a-f]{64}"),"invalid source SHA-256")
    require(baseline!=null && baseline.forall(b=>b!=null && b.keySet==vectors.keySet),"baseline encoder set differs")
    check(cancelled)
    val before=identity(source)
    require(before.isRegularFile && before.size()>=8 && before.size()<=maxFile,"unsupported safetensors file size")
    var created=false;var complete=false;var identityLink: Path=null
    try
      val input=FileChannel.open(source,StandardOpenOption.READ)
      try
        val parent=destination.toAbsolutePath.normalize().getParent
        identityLink=Files.createTempFile(parent,".scalatorch-edit-", ".identity")
        val out=FileChannel.open(identityLink,StandardOpenOption.WRITE,StandardOpenOption.TRUNCATE_EXISTING)
        try
          Files.createLink(destination,identityLink)
          created=true
          val digest=MessageDigest.getInstance("SHA-256")
          val buffer=ByteBuffer.allocate(1024*1024)
          var reading=true
          while reading do
            check(cancelled)
            buffer.clear()
            val n=input.read(buffer)
            if n<0 then reading=false
            else if n==0 then throw new java.io.IOException("embedding edit read made no progress")
            else
              buffer.flip();digest.update(buffer.asReadOnlyBuffer())
              while buffer.hasRemaining do
                check(cancelled)
                if out.write(buffer)<=0 then throw new java.io.IOException("embedding edit write made no progress")
          require(same(before,identity(source)),"source file changed while copying")
          require(java.util.HexFormat.of().formatHex(digest.digest()).equalsIgnoreCase(expectedSha256),"source file changed")
        finally out.close()
      finally input.close()
      check(cancelled)
      require(owns(destination,identityLink),"embedding edit stage changed")
      val archive=SafeTensors.open(identityLink,limits)
      val entries=try archive.tensors finally archive.close()
      val channel=FileChannel.open(identityLink,StandardOpenOption.READ,StandardOpenOption.WRITE)
      try
        val prefix=ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        SafeTensors.readFully(channel,prefix,0);prefix.flip()
        val dataStart=8+prefix.getLong()
        for (name,rows) <- vectors.toVector.sortBy(_._1) do
          check(cancelled)
          val info=entries.getOrElse(name,throw IllegalArgumentException(s"embedding tensor not found: $name"))
          require(Set(SafeDType.F16,SafeDType.BF16,SafeDType.F32,SafeDType.F64)(info.dtype),s"unsupported editable dtype: ${info.dtype}")
          val (count,width)=info.shape.dims match
            case Vector(w) => (1L,w)
            case Vector(n,w) => (n,w)
            case _ => throw IllegalArgumentException("embedding tensor must have rank one or two")
          require(count>0 && width>0 && count<=Int.MaxValue && width<=Int.MaxValue,"invalid embedding shape")
          def validate(value: Vector[Vector[Double]]): Unit =
            require(value!=null && value.size==count && value.forall(row=>row!=null && row.size==width && row.forall(_.isFinite)),
              "embedding vector shape or values are invalid")
          validate(rows)
          val prior=baseline.map(_(name));prior.foreach(validate)
          for row <- rows.indices do
            check(cancelled)
            for column <- rows(row).indices do
              val value=rows(row)(column)
              if prior.forall(_(row)(column)!=value) then
                val index=row.toLong*width+column
                val position=dataStart+info.begin+index*info.dtype.bytes
                writeFully(channel,encoded(value,info.dtype),position)
      finally channel.close()
      check(cancelled)
      require(owns(destination,identityLink),"embedding edit stage changed")
      complete=true
    finally
      if created && !complete && owns(destination,identityLink) then Files.deleteIfExists(destination)
      if identityLink!=null then Files.deleteIfExists(identityLink)
