package scalatorch.io

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files,Path,StandardOpenOption}
import java.util.concurrent.CancellationException
import scala.collection.mutable

/** Reader for the pre-1.6 (non-ZIP) PyTorch `torch.save` format: four consecutive pickles (magic number, protocol version,
  * system info, the state dict whose tensors are persistent storage references) and a list of storage keys, followed by one
  * int64 element count plus raw little-endian data per storage. Only a flat dict of plain contiguous float tensors is accepted;
  * the pickle interpreter understands just the opcodes and globals such a file needs, so nothing is ever executed. */
object LegacyPyTorchStateDict:
  private val MagicNumber=BigInt("119547037146038801333356")
  private val ProtocolVersion=1001L

  def isLegacy(path: Path): Boolean=
    val head=new Array[Byte](4)
    val input=Files.newInputStream(path)
    val n=try input.readNBytes(head,0,4) finally input.close()
    n==4 && !(head(0)=='P' && head(1)=='K')

  private case object Mark
  private final case class Global(module: String,name: String)
  private final case class StorageRef(key: String,dtype: PyTorchTensorArchive.StorageDType,elements: Long)
  private final case class TensorSpec(storage: StorageRef,offset: Long,shape: Vector[Long],stride: Vector[Long])

  def read(path: Path,limits: PyTorchArchiveLimits=PyTorchArchiveLimits.ImageModelWeights,
      cancelled: () => Boolean=()=>false): Map[String,PyTorchTensorArchive.StoredTensor]=
    import PyTorchTensorArchive.*
    def check(): Unit=if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("PyTorch archive reading cancelled")
    require(Files.isRegularFile(path),s"PyTorch tensor archive is not a regular file: $path")
    require(Files.size(path)<=limits.maxArchiveBytes,"PyTorch tensor archive exceeds its configured size limit")
    val channel=FileChannel.open(path,StandardOpenOption.READ)
    try
      val buf=channel.map(FileChannel.MapMode.READ_ONLY,0,channel.size()).order(ByteOrder.LITTLE_ENDIAN)
      var ops=0
      def fail(message: String): Nothing=throw IllegalArgumentException(s"unsupported legacy PyTorch archive: $message")
      def need(n: Long): Unit=if n<0 || buf.remaining()<n then throw IllegalArgumentException("truncated legacy PyTorch archive")
      def u8(): Int={ need(1);buf.get()&0xff }
      def string(n: Int): String={ need(n);val b=new Array[Byte](n);buf.get(b);new String(b,java.nio.charset.StandardCharsets.UTF_8) }
      def line(): String=
        val sb=new StringBuilder
        var c=u8()
        while c!='\n' do { sb.append(c.toChar);c=u8();require(sb.length<256,"overlong pickle line") }
        sb.toString
      /** Interprets one pickle starting at the current position. */
      def pickle(): Any=
        val stack=mutable.ArrayBuffer.empty[Any];val memo=mutable.Map.empty[Int,Any]
        def pop(): Any={ require(stack.nonEmpty,"pickle stack underflow");stack.remove(stack.size-1) }
        def popMark(): Vector[Any]=
          val at=stack.lastIndexOf(Mark);require(at>=0,"missing pickle mark")
          val items=stack.drop(at+1).toVector;stack.remove(at,stack.size-at);items
        var result: Option[Any]=None
        while result.isEmpty do
          ops+=1;require(ops<=limits.maxPickleOps,"pickle has too many operations")
          if (ops&0x3fff)==0 then check()
          u8() match
            case 0x80 => u8() // PROTO
            case 125 => stack+=mutable.LinkedHashMap.empty[Any,Any]
            case 93 => stack+=Vector.empty[Any]
            case 41 => stack+=Vector.empty[Any]
            case 40 => stack+=Mark
            case 88 => need(4);stack+=string(buf.getInt())
            case 85 => stack+=string(u8())
            case 84 => need(4);stack+=string(buf.getInt())
            case 75 => stack+=u8().toLong
            case 77 => need(2);stack+=(buf.getShort().toLong&0xffff)
            case 74 => need(4);stack+=buf.getInt().toLong
            case 0x8a => // LONG1
              val n=u8();need(n);val b=new Array[Byte](n);buf.get(b)
              val value=BigInt(b.reverse)
              stack+=(if value.isValidLong then value.toLong else value)
            case 78 => stack+=null
            case 0x88 => stack+=true
            case 0x89 => stack+=false
            case 116 => stack+=popMark()
            case 0x85 => stack+=Vector(pop()) // TUPLE1
            case 0x86 => val b=pop();stack+=Vector(pop(),b)
            case 0x87 => val c=pop();val b=pop();stack+=Vector(pop(),b,c)
            case 113 => memo(u8())=stack.last
            case 114 => need(4);memo(buf.getInt())=stack.last
            case 104 => val i=u8();stack+=memo.getOrElse(i,fail("unknown memo index"))
            case 106 => need(4);val i=buf.getInt();stack+=memo.getOrElse(i,fail("unknown memo index"))
            case 99 =>
              val g=Global(line(),line())
              stack+=g
            case 82 =>
              val args=pop().asInstanceOf[Vector[Any]];val f=pop()
              f match
                case Global("collections","OrderedDict") if args.isEmpty => stack+=mutable.LinkedHashMap.empty[Any,Any]
                case Global("torch._utils","_rebuild_tensor") | Global("torch._utils","_rebuild_tensor_v2") =>
                  args match
                    case (s: StorageRef) +: (off: Long) +: (shape: Vector[?]) +: (stride: Vector[?]) +: _ =>
                      stack+=TensorSpec(s,off,shape.asInstanceOf[Vector[Long]],stride.asInstanceOf[Vector[Long]])
                    case _ => fail("unexpected tensor rebuild arguments")
                case other => fail(s"global $other")
            case 81 => // BINPERSID
              pop() match
                case Vector("storage",Global("torch",kind),key: String,_,elements: Long,_) =>
                  val dtype=kind match
                    case "FloatStorage" => StorageDType.Float32
                    case "HalfStorage" => StorageDType.Float16
                    case other => fail(s"storage type $other")
                  stack+=StorageRef(key,dtype,elements)
                case _ => fail("persistent id")
            case 115 =>
              val v=pop();val k=pop();stack.last.asInstanceOf[mutable.LinkedHashMap[Any,Any]](k)=v
            case 117 =>
              val items=popMark();val d=stack.last.asInstanceOf[mutable.LinkedHashMap[Any,Any]]
              items.grouped(2).foreach { case Vector(k,v) => d(k)=v; case _ => fail("odd SETITEMS") }
            case 101 =>
              val items=popMark();stack(stack.size-1)=stack.last.asInstanceOf[Vector[Any]]++items
            case 97 => val v=pop();stack(stack.size-1)=stack.last.asInstanceOf[Vector[Any]]:+v
            case 46 => result=Some(pop())
            case other => fail(s"pickle opcode 0x${other.toHexString}")
        result.get
      require(pickle()==MagicNumber,"not a legacy PyTorch archive (magic number)")
      require(pickle()==ProtocolVersion,"unsupported legacy PyTorch protocol version")
      pickle()
      val main=pickle() match
        case d: mutable.LinkedHashMap[?,?] => d.asInstanceOf[mutable.LinkedHashMap[Any,Any]]
        case _ => fail("root object is not a state dict")
      val keys=pickle().asInstanceOf[Vector[Any]]
      val storages=mutable.Map.empty[String,(Long,Long,StorageDType)]
      var total=0L
      val specs=main.toVector.map {
        case (name: String,t: TensorSpec) => name->t
        case (name,_) => fail(s"entry $name is not a plain tensor")
      }
      val dtypes=specs.map((_,t)=>t.storage.key->t.storage.dtype).toMap
      for key <- keys do
        check()
        val k=key.asInstanceOf[String];need(8);val elements=buf.getLong()
        val dtype=dtypes.getOrElse(k,fail(s"storage $k is not referenced"))
        require(elements>=0 && elements<=limits.maxEntryBytes/dtype.bytes,"legacy storage is too large")
        val begin=buf.position().toLong
        need(elements*dtype.bytes);buf.position((begin+elements*dtype.bytes).toInt)
        storages(k)=(begin,elements,dtype)
      require(!buf.hasRemaining,"trailing data in legacy PyTorch archive")
      require(specs.size<=limits.maxTensors,"legacy archive has too many tensors")
      specs.map { (name,t) =>
        check()
        val (begin,stored,dtype)=storages.getOrElse(t.storage.key,fail(s"missing storage ${t.storage.key}"))
        require(t.shape.forall(_>=0),s"negative dimension in $name")
        val count=t.shape.foldLeft(1L)(Math.multiplyExact)
        require(count<=limits.maxElements,s"tensor $name exceeds the element limit")
        total=Math.addExact(total,count);require(total<=limits.maxTotalTensorElements,"total tensor element limit exceeded")
        var expected=1L;val contiguous=t.shape.zip(t.stride).reverse.forall { (d,s) => val ok=d<=1 || s==expected;expected*=math.max(d,1);ok }
        require(contiguous,s"tensor $name is not contiguous")
        require(t.offset>=0 && t.offset+count<=stored,s"tensor $name exceeds its storage")
        val values=new Array[Float](count.toInt);val at=(begin+t.offset*dtype.bytes).toInt
        var i=0
        while i<values.length do
          values(i)=dtype match
            case StorageDType.Float16 => java.lang.Float.float16ToFloat(buf.getShort(at+i*2))
            case _ => buf.getFloat(at+i*4)
          i+=1
        name->StoredTensor(dtype,t.shape,values)
      }.toMap
    catch
      case e: java.nio.BufferUnderflowException => throw IllegalArgumentException("truncated legacy PyTorch archive",e)
      case e: ArithmeticException => throw IllegalArgumentException("legacy PyTorch archive metadata overflow",e)
    finally channel.close()
