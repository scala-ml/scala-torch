package scalatorch.io

import java.io.{ByteArrayOutputStream,EOFException,InputStream}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.{CodingErrorAction,StandardCharsets}
import java.nio.file.{Files,Path}
import java.util.zip.{CRC32,ZipEntry,ZipFile}
import java.util.concurrent.CancellationException
import scala.collection.mutable
import scala.util.Using

final case class PyTorchArchiveLimits(
    maxArchiveBytes: Long = 64L*1024*1024,
    maxEntries: Int = 64,
    maxEntryBytes: Long = 64L*1024*1024,
    maxTotalUncompressedBytes: Long = 64L*1024*1024,
    maxPickleBytes: Int = 1024*1024,
    maxPickleOps: Int = 100000,
    maxTensors: Int = 8,
    maxElements: Long = 4L*1024*1024,
    maxTotalTensorElements: Long = Long.MaxValue,
    allowMissingCrc: Boolean = false
):
  require(maxArchiveBytes>=0 && maxEntries>=1 && maxEntryBytes>=0 && maxTotalUncompressedBytes>=0 &&
    maxPickleBytes>=1 && maxPickleOps>=1 && maxTensors>=1 && maxElements>=1 && maxTotalTensorElements>=0,"invalid PyTorch archive limits")

object PyTorchArchiveLimits:
  /** Bounded but large enough for conventional convolutional model state dictionaries. */
  val ImageModelWeights: PyTorchArchiveLimits = PyTorchArchiveLimits(
    maxArchiveBytes=256L*1024*1024,maxEntries=4096,maxEntryBytes=128L*1024*1024,
    maxTotalUncompressedBytes=512L*1024*1024,maxPickleBytes=16*1024*1024,
    maxPickleOps=2000000,maxTensors=4096,maxElements=4L*1024*1024)

object PyTorchTensorArchive:
  enum StorageDType(val bytes: Int):
    case Float32 extends StorageDType(4)
    case Float16 extends StorageDType(2)
    case BFloat16 extends StorageDType(2)
    case Int32 extends StorageDType(4)
    case Int64 extends StorageDType(8)

  final case class StoredTensor(dtype: StorageDType,shape: Vector[Long],values: Array[Float])
  final case class StoredIntegerTensor(dtype: StorageDType,shape: Vector[Long],values: Array[Long])
  sealed trait Value
  final case class TensorValue(tensor: StoredTensor) extends Value
  final case class IntegerTensorValue(tensor: StoredIntegerTensor) extends Value
  final case class TensorMetadataValue(dtype: StorageDType,shape: Vector[Long]) extends Value
  final case class DictValue(entries: Map[String,Value]) extends Value
  final case class SequenceValue(values: Vector[Value]) extends Value
  final case class StringValue(value: String) extends Value
  final case class IntegerValue(value: Long) extends Value
  final case class FloatValue(value: Double) extends Value
  final case class BooleanValue(value: Boolean) extends Value
  case object NoneValue extends Value
  final case class Archive(value: Value)

  /** True when `strides` is a permutation of the dense row-major layout of `shape` (size-1 dimensions are ignored), i.e. the view
    * covers exactly `prod(shape)` consecutive storage elements without overlap. */
  private def isDenseLayout(shape: Vector[Long],strides: Vector[Long]): Boolean=
    val dims=shape.zip(strides).filter(_._1>1).sortBy(-_._2)
    var expected=1L
    val ordered=dims.reverse.forall { (d,s) => val ok=s==expected;expected=Math.multiplyExact(expected,d);ok }
    ordered && strides.forall(_>=0)

  /** Storage-order values of a dense (possibly permuted) view reordered into logical row-major order. */
  private def logicalOrder(values: Array[Float],shape: Vector[Long],strides: Vector[Long]): Array[Float]=
    val contiguous=shape.zip(strides).filter(_._1>1).sortBy(-_._2).map(_._2)==shape.zip(strides).filter(_._1>1).map(_._2)
    if contiguous then values
    else
      val n=values.length;val out=new Array[Float](n);val rank=shape.length
      val dims=shape.map(_.toInt).toArray;val st=strides.map(_.toInt).toArray
      val index=new Array[Int](rank);var offset=0;var i=0
      while i<n do
        out(i)=values(offset)
        var d=rank-1;var carry=true
        while carry && d>=0 do
          index(d)+=1;offset+=st(d)
          if index(d)<dims(d) then carry=false
          else { offset-=st(d)*dims(d);index(d)=0;d-=1 }
        i+=1
      out

  private sealed trait Node
  private final case class Global(module: String,name: String) extends Node
  private final case class Storage(dtype: StorageDType,key: String,declaredElements: Long,order: ByteOrder) extends Node
  private final case class Tensor(tensor: StoredTensor) extends Node
  private final case class IntegerTensor(tensor: StoredIntegerTensor) extends Node
  private final case class DeferredTensor(store: Storage,offset: Long,shape: Vector[Long],elements: Int,strides: Vector[Long]) extends Node
  private class DictNode extends Node:
    val values=mutable.LinkedHashMap.empty[String,Node]
  private final class OrderedDictNode extends DictNode
  private final class ListNode extends Node:
    val values=mutable.ArrayBuffer.empty[Node]
  private final case class TupleNode(values: Vector[Node]) extends Node
  private final case class Text(value: String) extends Node
  private final case class Number(value: Long) extends Node
  private final case class Real(value: Double) extends Node
  private final case class Bool(value: Boolean) extends Node
  private case object NilNode extends Node
  private case object Mark

  private def fail(message: String): Nothing=throw IllegalArgumentException(message)

  private def readBounded(input: InputStream,limit: Long,label: String,check: () => Unit): Array[Byte]=
    val out=new ByteArrayOutputStream(math.min(limit,65536L).toInt)
    val buffer=new Array[Byte](8192)
    var total=0L
    var count=input.read(buffer)
    while count>=0 do
      check()
      if count>0 then
        total=Math.addExact(total,count.toLong)
        if total>limit || total>Int.MaxValue then fail(s"PyTorch archive $label exceeds its configured size limit")
        out.write(buffer,0,count)
      count=input.read(buffer)
    out.toByteArray

  private def validName(name: String): Boolean=
    name.nonEmpty && !name.startsWith("/") && !name.contains('\\') &&
      name.split("/",-1).forall(part=>part.nonEmpty && part!="." && part!="..")

  private def decodeUtf8(bytes: Array[Byte]): String=
    try StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
      .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString
    catch case e: java.nio.charset.CharacterCodingException => throw IllegalArgumentException("invalid UTF-8 in PyTorch pickle",e)

  def read(path: Path,limits: PyTorchArchiveLimits=PyTorchArchiveLimits()): Archive=
    read(path,limits,()=>false)

  def read(path: Path,limits: PyTorchArchiveLimits,cancelled: () => Boolean): Archive=
    read(path,limits,cancelled,()=>())

  def read(path: Path,limits: PyTorchArchiveLimits,cancelled: () => Boolean,onFirstTensorData: () => Unit): Archive=
    readInternal(path,limits,cancelled,onFirstTensorData,selectRootStateDict=false,None)

  /** Selects a completed root state_dict without interpreting subsequent training metadata. */
  def readStateDict(path: Path,limits: PyTorchArchiveLimits,cancelled: () => Boolean): Archive=
    readInternal(path,limits,cancelled,()=>(),selectRootStateDict=true,None)

  /** Reads only selected tensor values while preserving metadata for all state-dict entries. */
  def readStateDictSelected(path: Path,limits: PyTorchArchiveLimits,cancelled: () => Boolean,
      select: String => Boolean): Archive =
    require(select!=null,"tensor selector must not be null")
    readInternal(path,limits,cancelled,()=>(),selectRootStateDict=true,Some(select))

  private def readInternal(path: Path,limits: PyTorchArchiveLimits,cancelled: () => Boolean,
      onFirstTensorData: () => Unit,selectRootStateDict: Boolean,select: Option[String => Boolean]): Archive=
    require(cancelled!=null && onFirstTensorData!=null,"archive callbacks must not be null")
    def check(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("PyTorch archive reading cancelled")
    check()
    require(Files.isRegularFile(path),s"PyTorch tensor archive is not a regular file: $path")
    require(Files.size(path)<=limits.maxArchiveBytes,"PyTorch tensor archive exceeds its configured size limit")
    val zip=try new ZipFile(path.toFile)
    catch case e: java.io.IOException => throw IllegalArgumentException("PyTorch tensor archive must be a current ZIP format",e)
    try
      val entries=mutable.LinkedHashMap.empty[String,ZipEntry]
      val enumeration=zip.entries()
      while enumeration.hasMoreElements do
        check()
        val entry=enumeration.nextElement()
        require(entries.size<limits.maxEntries,"PyTorch archive has too many entries")
        require(validName(entry.getName),s"unsafe PyTorch ZIP entry name: ${entry.getName}")
        require(!entry.isDirectory,s"unexpected directory in PyTorch ZIP archive: ${entry.getName}")
        require(!entries.contains(entry.getName),s"duplicate PyTorch ZIP entry: ${entry.getName}")
        if entry.getSize>=0 then require(entry.getSize<=limits.maxEntryBytes,s"PyTorch ZIP entry is too large: ${entry.getName}")
        entries(entry.getName)=entry
      val picklePaths=entries.keys.filter(_.endsWith("/data.pkl")).toVector
      require(picklePaths.size==1,"PyTorch ZIP archive must contain exactly one data.pkl")
      val picklePath=picklePaths.head
      val root=picklePath.stripSuffix("data.pkl")
      val recognized=entries.keys.forall { name =>
        name.startsWith(root) && {
          val tail=name.substring(root.length)
          tail=="data.pkl" || tail=="version" || tail=="byteorder" || tail==".data/serialization_id" ||
            tail==".format_version" || tail==".storage_alignment" || tail.matches("data/[0-9]+")
        }
      }
      require(recognized,"PyTorch ZIP archive contains unsupported entries")
      val indexed=entries.iterator.map((name,entry)=>name.substring(root.length)->entry).toMap
      val total=indexed.valuesIterator.foldLeft(0L) { (size,entry) =>
        check()
        require(entry.getSize>=0 && entry.getSize<=limits.maxEntryBytes,s"invalid PyTorch ZIP entry size: ${entry.getName}")
        Math.addExact(size,entry.getSize)
      }
      require(total<=limits.maxTotalUncompressedBytes,"PyTorch archive expands beyond its configured size limit")
      val validated=mutable.Set.empty[String]
      // PyTorch's CRC-disabled writer emits zero CRC for every archive member.
      // Only accept that ambiguous representation when explicitly requested.
      val allCrcZero=indexed.valuesIterator.forall(_.getCrc==0L)
      val missingCrc=limits.allowMissingCrc && allCrcZero
      var tensorDataReported=false
      def verifyCrc(entry: ZipEntry,crc: CRC32): Unit =
        require(missingCrc || (entry.getCrc>=0 && crc.getValue==entry.getCrc),
          s"PyTorch ZIP entry CRC mismatch: ${entry.getName}"+
            (if allCrcZero then " (archive has no recorded CRC; explicit compatibility option required)" else ""))
      def entrySize(name: String): Option[Long]=indexed.get(name).map(_.getSize)
      def readEntry(name: String,limit: Long): Array[Byte]=
        val entry=indexed.getOrElse(name,fail(s"missing PyTorch storage record: $name"))
        val data=Using.resource(zip.getInputStream(entry))(input=>readBounded(input,math.min(limit,entry.getSize),name,check))
        require(data.length.toLong==entry.getSize,s"PyTorch ZIP entry length mismatch: $name")
        val crc=new CRC32();crc.update(data);verifyCrc(entry,crc)
        validated+=name
        data
      def readTensor(store: Storage,offset: Long,values: Array[Float],integerValues: Array[Long]): Unit =
        val name=s"data/${store.key}"
        val entry=indexed.getOrElse(name,fail(s"missing PyTorch storage record: $name"))
        val begin=Math.multiplyExact(offset,store.dtype.bytes.toLong)
        val length=if values!=null then values.length else integerValues.length
        val end=Math.addExact(begin,Math.multiplyExact(length.toLong,store.dtype.bytes.toLong))
        val buffer=new Array[Byte](65536)
        val crc=new CRC32()
        var position=0L
        var output=0
        Using.resource(zip.getInputStream(entry)) { input =>
          while position<entry.getSize do
            check()
            val size=math.min(buffer.length.toLong,entry.getSize-position).toInt
            val count=input.readNBytes(buffer,0,size)
            require(count==size,s"PyTorch ZIP entry length mismatch: $name")
            crc.update(buffer,0,count)
            if !tensorDataReported then
              tensorDataReported=true
              onFirstTensorData()
              check()
            val from=math.max(begin,position)
            val to=math.min(end,position+count)
            if from<to then
              val decoded=ByteBuffer.wrap(buffer).order(store.order)
              decoded.position((from-position).toInt)
              val stop=(to-position).toInt
              while decoded.position()<stop do
                store.dtype match
                  case StorageDType.Float32 => values(output)=decoded.getFloat()
                  case StorageDType.Float16 => values(output)=java.lang.Float.float16ToFloat(decoded.getShort())
                  case StorageDType.BFloat16 => values(output)=java.lang.Float.intBitsToFloat((decoded.getShort().toInt&0xffff)<<16)
                  case StorageDType.Int32 => integerValues(output)=decoded.getInt().toLong
                  case StorageDType.Int64 => integerValues(output)=decoded.getLong()
                output+=1
            position+=count
          require(input.read()== -1,s"PyTorch ZIP entry exceeds its declared size: $name")
        }
        require(output==length,s"PyTorch tensor storage length mismatch: $name")
        verifyCrc(entry,crc)
        validated+=name
      val byteOrder=indexed.get("byteorder").map { _ =>
        val bytes=readEntry("byteorder",limits.maxEntryBytes)
        decodeUtf8(bytes).trim match
          case "little" => ByteOrder.LITTLE_ENDIAN
          case "big" => ByteOrder.BIG_ENDIAN
          case _ => fail("unsupported PyTorch storage byte order")
      }.getOrElse(ByteOrder.LITTLE_ENDIAN)
      val pickle=readEntry("data.pkl",limits.maxPickleBytes)
      val interpreter=new PickleReader(pickle,entrySize,readTensor,byteOrder,limits,check,selectRootStateDict,select.nonEmpty)
      check()
      val materialized=new java.util.IdentityHashMap[DeferredTensor,Value]()
      def freezeSelected(node: Node,name: String): Value =
        check()
        node match
          case tensor @ DeferredTensor(store,offset,shape,elements,strides) =>
            if !select.get(name) then TensorMetadataValue(store.dtype,shape)
            else
              val existing=materialized.get(tensor)
              if existing!=null then existing
              else
                val value=if store.dtype==StorageDType.Int32 || store.dtype==StorageDType.Int64 then
                  val values=new Array[Long](elements)
                  readTensor(store,offset,null,values)
                  IntegerTensorValue(StoredIntegerTensor(store.dtype,shape,values))
                else
                  val values=new Array[Float](elements)
                  readTensor(store,offset,values,null)
                  TensorValue(StoredTensor(store.dtype,shape,logicalOrder(values,shape,strides)))
                materialized.put(tensor,value)
                value
          case dict: DictNode => DictValue(dict.values.iterator.map((key,value)=>key->freezeSelected(value,key)).toMap)
          case list: ListNode => SequenceValue(list.values.iterator.map(value=>freezeSelected(value,name)).toVector)
          case TupleNode(values) => SequenceValue(values.map(value=>freezeSelected(value,name)))
          case other => freeze(other,check)
      val parsed=interpreter.read()
      val result=Archive(if select.nonEmpty then freezeSelected(parsed,"") else freeze(parsed,check))
      // Validate even unreferenced entries without retaining their contents.
      val buffer=new Array[Byte](8192)
      indexed.foreach { (name,entry) =>
        check()
        if !validated(name) then
          var count=0L
          val crc=new CRC32()
          Using.resource(zip.getInputStream(entry)) { input =>
            var n=input.read(buffer)
            while n>=0 do
              check()
              if n>0 then
                count=Math.addExact(count,n.toLong)
                require(count<=entry.getSize,s"PyTorch ZIP entry exceeds its declared size: $name")
                crc.update(buffer,0,n)
              n=input.read(buffer)
          }
          require(count==entry.getSize,s"PyTorch ZIP entry length mismatch: $name")
          verifyCrc(entry,crc)
      }
      check()
      result
    catch
      case e: IllegalArgumentException => throw e
      case e: ArithmeticException => throw IllegalArgumentException("PyTorch archive metadata overflow",e)
      case e: java.util.zip.ZipException => throw IllegalArgumentException("invalid PyTorch ZIP archive",e)
      case e: EOFException => throw IllegalArgumentException("truncated PyTorch tensor archive",e)
      case e: java.io.IOException => throw IllegalArgumentException("cannot read PyTorch tensor archive",e)
    finally zip.close()

  private def freeze(node: Node,check: () => Unit): Value =
    check()
    node match
      case Tensor(tensor) => TensorValue(tensor)
      case IntegerTensor(tensor) => IntegerTensorValue(tensor)
      case dict: DictNode => DictValue(dict.values.iterator.map((key,value)=>key->freeze(value,check)).toMap)
      case list: ListNode => SequenceValue(list.values.iterator.map(value=>freeze(value,check)).toVector)
      case TupleNode(values) => SequenceValue(values.map(value=>freeze(value,check)))
      case Text(value) => StringValue(value)
      case Number(value) => IntegerValue(value)
      case Real(value) => FloatValue(value)
      case Bool(value) => BooleanValue(value)
      case NilNode => NoneValue
      case other => fail(s"unsupported PyTorch pickle value: ${other.getClass.getSimpleName}")

  private final class PickleReader(data: Array[Byte],entrySize: String => Option[Long],
      readTensor: (Storage,Long,Array[Float],Array[Long]) => Unit,
      order: ByteOrder,limits: PyTorchArchiveLimits,check: () => Unit,selectRootStateDict: Boolean,deferTensors: Boolean):
    private val in=ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
    private val stack=mutable.ArrayBuffer.empty[Any]
    private val memo=mutable.Map.empty[Int,Any]
    private var nextMemo=0
    private var operations=0
    private var tensors=0
    private var totalTensorElements=0L
    private var frameEnd = -1L

    private def completedRootStateDict: Option[Node] =
      if !selectRootStateDict || stack.size<4 || stack(1)!=Mark then None
      else (stack.head,stack(stack.size-2),stack.last) match
        case (root: DictNode,Text("state_dict"),state: DictNode) if root.values.isEmpty =>
          val prior=stack.slice(2,stack.size-2).toVector
          val safe=prior.size%2==0 && prior.grouped(2).forall {
            case Vector(Text(key),_: Text | _: Number | _: Real | _: Bool | NilNode) => key!="state_dict"
            case _ => false
          }
          Option.when(safe)(state)
        case _ => None

    private def need(count: Int): Unit=if count<0 || in.remaining()<count then throw EOFException("truncated pickle opcode")
    private def byte(): Int={ need(1); in.get()&0xff }
    private def int32(): Int={ need(4); in.getInt() }
    private def int64(): Long={ need(8); in.getLong() }
    private def bytes(count: Int): Array[Byte]={ need(count); val out=new Array[Byte](count); in.get(out); out }
    private def line(): String=
      val start=in.position()
      while in.hasRemaining && in.get()!='\n'.toByte do ()
      if !in.hasRemaining then throw EOFException("unterminated pickle GLOBAL")
      val end=in.position()-1
      val out=new Array[Byte](end-start)
      val copy=in.duplicate(); copy.position(start).limit(end); copy.get(out)
      decodeUtf8(out)
    private def push(value: Any): Unit=stack+=value
    private def pop(): Any=if stack.isEmpty then fail("pickle stack underflow") else stack.remove(stack.length-1)
    private def peek: Any=if stack.isEmpty then fail("pickle stack underflow") else stack.last
    private def popMark(): Vector[Any]=
      val index=stack.lastIndexOf(Mark)
      if index<0 then fail(s"pickle MARK is missing at byte ${in.position()} with stack ${stack.toVector}")
      val values=stack.drop(index+1).toVector
      stack.dropRightInPlace(stack.length-index)
      values
    private def asLong(value: Any): Long=value match
      case Number(number) => number
      case _ => fail("pickle tensor metadata must be integers")
    private def asString(value: Any): String=value match
      case Text(text) => text
      case _ => fail("pickle mapping keys and storage metadata must be strings")
    private def memoPut(index: Int,value: Any): Unit=
      require(index>=0 && index<=limits.maxPickleOps,"invalid pickle memo index")
      memo(index)=value
      nextMemo=math.max(nextMemo,index+1)
    private def storage(value: Any): Storage=value match
      case store: Storage => store
      case id: TupleNode =>
        require(id.values.length==5,"unsupported PyTorch persistent storage identifier")
        require(asString(id.values(0))=="storage","unsupported PyTorch persistent object")
        val storageType=id.values(1) match
          case Global("torch","FloatStorage") => StorageDType.Float32
          case Global("torch","HalfStorage") => StorageDType.Float16
          case Global("torch","BFloat16Storage") => StorageDType.BFloat16
          case Global("torch","IntStorage") => StorageDType.Int32
          case Global("torch","LongStorage") => StorageDType.Int64
          case _ => fail("unsupported PyTorch storage type")
        val key=asString(id.values(2))
        require(key.matches("[0-9]+"),"invalid PyTorch storage key")
        val device=asString(id.values(3))
        require(device=="cpu" || device.matches("cuda:[0-9]+"),"unsupported PyTorch storage device tag")
        val elements=asLong(id.values(4))
        require(elements>=0 && elements<=limits.maxElements,"PyTorch storage element count exceeds its limit")
        val expected=Math.multiplyExact(elements,storageType.bytes.toLong)
        require(entrySize(s"data/$key").getOrElse(fail(s"missing PyTorch storage record: $key"))==expected,
          "PyTorch storage record length does not match its metadata")
        Storage(storageType,key,elements,order)
      case _ => fail("unsupported PyTorch persistent object")
    private def buildTensor(args: Vector[Any],version2: Boolean): Node=
      require(args.length==(if version2 then 6 else 4),"invalid PyTorch tensor rebuild arguments")
      if version2 then
        require(args(4).isInstanceOf[Bool],"PyTorch tensor requires_grad flag must be boolean")
        args(5) match
          case hooks: DictNode => require(hooks.values.isEmpty,"PyTorch tensor backward hooks are unsupported")
          case _ => fail("PyTorch tensor backward hooks must be an empty OrderedDict")
      val store=storage(args(0))
      val offset=asLong(args(1))
      val shape=asTupleLong(args(2))
      val strides=asTupleLong(args(3))
      require(shape.length<=8 && shape.length==strides.length,"invalid PyTorch tensor dimensions")
      require(shape.forall(_>=0),"negative PyTorch tensor dimension")
      val elements=shape.foldLeft(1L)((n,d)=>Math.multiplyExact(n,d))
      require(elements<=limits.maxElements && elements<=Int.MaxValue,"PyTorch tensor exceeds its element limit")
      require(elements<=limits.maxTotalTensorElements-totalTensorElements,"PyTorch tensors exceed their total element limit")
      totalTensorElements+=elements
      val expectedStrides=shape.indices.reverse.foldLeft(Vector.empty[Long])((acc,index)=>
        val stride=if acc.isEmpty then 1L else Math.multiplyExact(acc.head,shape(index+1))
        stride+:acc)
      val contiguous=strides==expectedStrides
      // a dense permutation of the contiguous layout (e.g. channels-last weights) is gathered into logical order; overlapping or strided views stay rejected
      require(contiguous || (isDenseLayout(shape,strides) && store.dtype!=StorageDType.Int32 && store.dtype!=StorageDType.Int64),"non-contiguous PyTorch tensor views are unsupported")
      require(offset>=0 && Math.addExact(offset,elements)<=store.declaredElements,"PyTorch tensor view exceeds its storage")
      tensors+=1
      require(tensors<=limits.maxTensors,"PyTorch archive contains too many tensors")
      if deferTensors then DeferredTensor(store,offset,shape,elements.toInt,strides)
      else if store.dtype==StorageDType.Int32 || store.dtype==StorageDType.Int64 then
        val values=new Array[Long](elements.toInt)
        readTensor(store,offset,null,values)
        IntegerTensor(StoredIntegerTensor(store.dtype,shape,values))
      else
        val values=new Array[Float](elements.toInt)
        readTensor(store,offset,values,null)
        Tensor(StoredTensor(store.dtype,shape,logicalOrder(values,shape,strides)))
    private def asTupleLong(value: Any): Vector[Long]=value match
      case TupleNode(items) => items.map(asLong)
      case _ => fail("PyTorch tensor shape/stride must be tuples")
    private def tuple(items: Vector[Any]): TupleNode=TupleNode(items.map {
      case node: Node => node
      case other => fail(s"unsupported pickle tuple value: ${other.getClass.getSimpleName}")
    })
    private def safeOrderedMetadata(value: Node,depth: Int=0): Boolean=
      if depth>32 then false
      else value match
        case dict: DictNode => dict.values.forall {
          case ("version",Number(version)) => version>=0 && version<=1024
          case (_,nested: DictNode) => safeOrderedMetadata(nested,depth+1)
          case _ => false
        }
        case _ => false
    private def reduce(): Unit=
      val args=pop() match
        case TupleNode(items) => items
        case _ => fail("pickle REDUCE arguments must be a tuple")
      val callable=pop() match
        case global: Global => global
        case _ => fail("pickle REDUCE target is not an allowlisted global")
      val result=callable match
        case Global("torch._utils","_rebuild_tensor_v2") => buildTensor(args,version2=true)
        case Global("torch._utils","_rebuild_tensor") => buildTensor(args,version2=false)
        case Global("collections","OrderedDict") if args.isEmpty => new OrderedDictNode
        case _ => fail(s"pickle global is not allowlisted: ${callable.module}.${callable.name}")
      push(result)
    private def setItems(items: Vector[Any]): Unit=
      require(items.length%2==0,"pickle dictionary has an unmatched item")
      val target=peek match
        case dict: DictNode => dict
        case _ => fail("pickle SETITEM target must be a dictionary")
      items.grouped(2).foreach { pair =>
        val key=asString(pair(0))
        require(!target.values.contains(key),s"duplicate PyTorch pickle dictionary key: $key")
        pair(1) match
          case node: Node => target.values(key)=node
          case _ => fail("unsupported value in PyTorch pickle dictionary")
      }

    def read(): Node=
      require(byte()==0x80,"PyTorch archive data.pkl must use a binary pickle protocol")
      val protocol=byte()
      require(protocol>=2 && protocol<=5,s"unsupported PyTorch pickle protocol: $protocol")
      var done=false
      while !done do
        operations+=1
        if (operations & 1023)==0 then check()
        require(operations<=limits.maxPickleOps,"PyTorch pickle has too many opcodes")
        val opcode=byte()
        opcode match
          case 0x95 => // FRAME
            val length=int64(); require(length>=0 && length<=in.remaining(),"invalid pickle frame length")
            require(frameEnd<0,"nested pickle FRAME is invalid")
            frameEnd=Math.addExact(in.position().toLong,length)
          case 'c' => push(Global(line(),line()))
          case 0x93 =>
            val name=asString(pop()); val module=asString(pop()); push(Global(module,name))
          case 'q' => memoPut(byte(),peek)
          case 'r' => memoPut(int32(),peek)
          case 0x94 => memoPut(nextMemo,peek)
          case 'h' => push(memo.getOrElse(byte(),fail("missing pickle memo reference")))
          case 'j' => push(memo.getOrElse(int32(),fail("missing pickle memo reference")))
          case '(' => push(Mark)
          case ')' => push(TupleNode(Vector.empty))
          case ']' => push(new ListNode)
          case '}' => push(new DictNode)
          case 'N' => push(NilNode)
          case 0x88 => push(Bool(true))
          case 0x89 => push(Bool(false))
          case 'K' => push(Number(byte().toLong))
          case 'M' => need(2); push(Number((byte()|(byte()<<8)).toLong))
          case 'J' => push(Number(int32().toLong))
          case 'G' => push(Real(java.lang.Double.longBitsToDouble(java.lang.Long.reverseBytes(int64()))))
          case 0x8a =>
            val size=byte(); require(size<=8,"pickle integer is too large")
            val raw=bytes(size)
            var value=if size>0 && (raw.last&0x80)!=0 then -1L else 0L
            var i=size-1
            while i>=0 do { value=(value<<8)|(raw(i)&0xffL); i-=1 }
            push(Number(value))
          case 'X' =>
            val size=int32(); require(size>=0 && size<=in.remaining(),"invalid pickle string length")
            push(Text(decodeUtf8(bytes(size))))
          case 0x8c =>
            val size=byte(); push(Text(decodeUtf8(bytes(size))))
          case 0x8d =>
            val size=int64(); require(size>=0 && size<=Int.MaxValue && size<=in.remaining(),"invalid pickle string length")
            push(Text(decodeUtf8(bytes(size.toInt))))
          case 0x8f =>
            val size=byte(); push(Text(decodeUtf8(bytes(size))))
          case 0x8e =>
            val size=int64(); require(size>=0 && size<=Int.MaxValue && size<=in.remaining(),"invalid pickle string length")
            push(Text(decodeUtf8(bytes(size.toInt))))
          case 't' => push(tuple(popMark()))
          case 0x85 => push(tuple(Vector(pop())))
          case 0x86 => val b=pop(); val a=pop(); push(tuple(Vector(a,b)))
          case 0x87 => val c=pop(); val b=pop(); val a=pop(); push(tuple(Vector(a,b,c)))
          case 'R' => reduce()
          case 'Q' => push(storage(pop()))
          case 's' =>
            val value=pop(); val key=pop()
            val target=peek match
              case dict: DictNode => dict
              case _ => fail("pickle SETITEM target must be a dictionary")
            setOne(target,key,value)
          case 'u' =>
            val items=popMark()
            setItems(items)
          case 'a' =>
            val value=pop()
            peek match
              case list: ListNode => value match
                case node: Node => list.values+=node
                case _ => fail("unsupported pickle list item")
              case _ => fail("pickle APPEND target must be a list")
          case 'e' =>
            val items=popMark()
            peek match
              case list: ListNode => items.foreach { case node: Node => list.values+=node; case _ => fail("unsupported pickle list item") }
              case _ => fail("pickle APPENDS target must be a list")
          case 'b' =>
            val state=pop() match
              case dict: DictNode => dict
              case _ => fail("PyTorch OrderedDict BUILD state must be a dictionary")
            peek match
              case _: OrderedDictNode =>
                require(state.values.keySet==Set("_metadata"),"unsupported PyTorch OrderedDict BUILD state")
                state.values.get("_metadata").foreach(value =>
                  require(safeOrderedMetadata(value),"unsupported PyTorch OrderedDict module metadata"))
              case _ => fail("pickle BUILD target is not an allowlisted OrderedDict")
          case '.' =>
            done=true
          case opcode => fail(f"unsupported PyTorch pickle opcode: 0x$opcode%02x")
        // CPython writes dictionaries in SETITEMS batches separated by MARK.
        // A batch followed immediately by MARK is not the end of the dictionary.
        if selectRootStateDict && opcode=='u' && in.hasRemaining && (in.get(in.position())&0xff)!='(' then
          completedRootStateDict match
            case Some(value) => return value
            case None => ()
        if frameEnd>=0 then
          require(in.position().toLong<=frameEnd,"pickle opcode exceeds its declared frame")
          if in.position().toLong==frameEnd then frameEnd = -1L
      require(!in.hasRemaining,"trailing data after PyTorch pickle STOP")
      require(stack.size==1,"PyTorch pickle did not produce exactly one value")
      stack.head match
        case node: Node => node
        case _ => fail("PyTorch pickle root value is unsupported")

    private def setOne(target: DictNode,key: Any,value: Any): Unit=
      val string=asString(key)
      require(!target.values.contains(string),s"duplicate PyTorch pickle dictionary key: $string")
      value match
        case node: Node => target.values(string)=node
        case _ => fail("unsupported value in PyTorch pickle dictionary")
