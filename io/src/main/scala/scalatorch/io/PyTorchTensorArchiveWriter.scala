package scalatorch.io

import java.io.{ByteArrayOutputStream,DataOutputStream}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,StandardCopyOption}
import java.util.zip.{ZipEntry,ZipOutputStream}
import java.util.concurrent.CancellationException
import PyTorchTensorArchive.*

/** Writes the restricted tensor/dictionary subset understood by the native reader and torch.load. */
object PyTorchTensorArchiveWriter:
  def write(path: Path,value: Value,overwrite: Boolean=false,cancelled: () => Boolean=() => false): Unit =
    require(path!=null && value!=null && cancelled!=null,"invalid PyTorch archive arguments")
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("PyTorch write cancelled")
    val bytes=new ByteArrayOutputStream
    val out=new DataOutputStream(bytes)
    val storages=scala.collection.mutable.ArrayBuffer.empty[Array[Byte]]
    def op(value: Int): Unit=out.writeByte(value)
    def raw(value: String): Unit=out.write(value.getBytes(UTF_8))
    def global(module: String,name: String): Unit={ op('c');raw(module);op('\n');raw(name);op('\n') }
    def unicode(value: String): Unit =
      require(value!=null && value.length<=1048576,"invalid PyTorch string")
      val encoded=value.getBytes(UTF_8);op('X');out.writeInt(Integer.reverseBytes(encoded.length));out.write(encoded)
    def integer(value: Long): Unit =
      if value>=Int.MinValue && value<=Int.MaxValue then { op('J');out.writeInt(Integer.reverseBytes(value.toInt)) }
      else
        val data=BigInt(value).toByteArray.reverse
        op(0x8a);op(data.length);out.write(data)
    def tuple(values: Vector[Long]): Unit={ op('(');values.foreach(integer);op('t') }
    def tensor(value: StoredTensor): Unit =
      require(value!=null && value.shape.size<=64 && value.shape.forall(_>=0),"invalid PyTorch tensor shape")
      val count=value.shape.foldLeft(1L)(Math.multiplyExact)
      require(count==value.values.length && count<=4L*1024*1024,"invalid PyTorch tensor size")
      require(value.values.forall(_.isFinite),"PyTorch tensor contains nonfinite values")
      require(storages.size<64,"too many PyTorch tensors")
      val key=storages.size.toString
      val storage=ByteBuffer.allocate(Math.toIntExact(count*value.dtype.bytes)).order(ByteOrder.LITTLE_ENDIAN)
      value.values.foreach { x =>
        value.dtype match
          case StorageDType.Float32 => storage.putFloat(x)
          case StorageDType.Float16 =>
            val encoded=java.lang.Float.floatToFloat16(x)
            require((encoded & 0x7c00)!=0x7c00,"PyTorch tensor value exceeds Float16 range")
            storage.putShort(encoded)
          case StorageDType.BFloat16 =>
            val bits=java.lang.Float.floatToRawIntBits(x)
            val encoded=((bits+0x7fff+((bits>>>16)&1))>>>16).toShort
            require((encoded & 0x7f80)!=0x7f80,"PyTorch tensor value exceeds BFloat16 range")
            storage.putShort(encoded)
          case _ => throw IllegalArgumentException("floating-point tensor requires a floating-point dtype")
      }
      storages+=storage.array()
      val storageType=value.dtype match
        case StorageDType.Float32 => "FloatStorage"
        case StorageDType.Float16 => "HalfStorage"
        case StorageDType.BFloat16 => "BFloat16Storage"
        case _ => throw IllegalArgumentException("floating-point tensor requires a floating-point dtype")
      global("torch._utils","_rebuild_tensor_v2");op('(');op('(')
      unicode("storage");global("torch",storageType);unicode(key);unicode("cpu");integer(count)
      op('t');op('Q');integer(0);tuple(value.shape)
      val strides=value.shape.indices.map(i=>value.shape.drop(i+1).foldLeft(1L)(Math.multiplyExact)).toVector
      tuple(strides);op(0x89);global("collections","OrderedDict");op(')');op('R');op('t');op('R')
    def emit(value: Value,depth: Int): Unit =
      check();require(depth<=32,"PyTorch data is too deeply nested")
      value match
        case TensorValue(t) => tensor(t)
        case DictValue(entries) =>
          require(entries.size<=4096,"too many PyTorch dictionary entries")
          op('}');entries.toVector.sortBy(_._1).foreach { (key,item) => unicode(key);emit(item,depth+1);op('s') }
        case StringValue(s) => unicode(s)
        case IntegerValue(i) => integer(i)
        case FloatValue(d) => op('G');out.writeLong(java.lang.Double.doubleToRawLongBits(d))
        case BooleanValue(b) => op(if b then 0x88 else 0x89)
        case NoneValue => op('N')
        case _ => throw IllegalArgumentException("unsupported PyTorch archive value")
    op(0x80);op(2);emit(value,0);op('.');out.flush()
    require(bytes.size()<=1024*1024,"PyTorch pickle too large")
    val target=path.toAbsolutePath.normalize();val parent=target.getParent
    require(parent!=null && Files.isDirectory(parent),"PyTorch output directory does not exist")
    check()
    val stage=Files.createTempFile(parent,".scalatorch-pytorch-",".tmp")
    try
      val zip=new ZipOutputStream(Files.newOutputStream(stage))
      try
        def entry(name: String,data: Array[Byte]): Unit={ check();zip.putNextEntry(new ZipEntry(name));zip.write(data);zip.closeEntry() }
        entry("archive/data.pkl",bytes.toByteArray)
        storages.zipWithIndex.foreach((data,index)=>entry(s"archive/data/$index",data))
        entry("archive/byteorder","little".getBytes(UTF_8));entry("archive/version","3\n".getBytes(UTF_8))
      finally zip.close()
      check()
      if overwrite then Files.move(stage,target,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
      else Files.createLink(target,stage)
    finally Files.deleteIfExists(stage)
