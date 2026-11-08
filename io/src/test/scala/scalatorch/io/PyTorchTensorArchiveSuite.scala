package scalatorch.io

import java.io.{ByteArrayOutputStream,DataOutputStream}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}
import java.util.zip.{ZipEntry,ZipOutputStream}

class PyTorchTensorArchiveSuite extends munit.FunSuite:
  test("CRC-disabled torch.save archives require explicit opt-in") {
    import PyTorchTensorArchive.*
    val path=Files.createTempFile("torch-no-crc-", ".pt")
    try
      scala.util.Using.resource(this.getClass.getResourceAsStream("/pytorch-no-crc.pt")) { input =>
        Files.copy(input,path,java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      }
      intercept[IllegalArgumentException](read(path))
      val accepted=read(path,PyTorchArchiveLimits(allowMissingCrc=true)).value.asInstanceOf[DictValue].entries
      assertEquals(accepted("weight").asInstanceOf[TensorValue].tensor.values.toVector,Vector(1f,2f))
      val selected=readStateDictSelected(path,PyTorchArchiveLimits(allowMissingCrc=true),()=>false,_=>false)
      assertEquals(selected.value.asInstanceOf[DictValue].entries("weight"),TensorMetadataValue(StorageDType.Float32,Vector(2L)))
    finally Files.deleteIfExists(path)
  }
  test("rejects corrupted selected and unselected ZIP storage contents") {
    import PyTorchTensorArchive.*
    val payload=Array[Byte](17,42,93,114,27,61,49,13)
    for selected <- Vector(false,true); corruptUsed <- Vector(false,true) do
      val path=Files.createTempFile("torch-bad-crc-", ".ckpt")
      try
        val zip=new ZipOutputStream(Files.newOutputStream(path))
        def put(name: String,bytes: Array[Byte]): Unit =
          val crc=new java.util.zip.CRC32();crc.update(bytes)
          val entry=new ZipEntry("archive/"+name)
          entry.setMethod(ZipEntry.STORED);entry.setSize(bytes.length);entry.setCompressedSize(bytes.length);entry.setCrc(crc.getValue)
          zip.putNextEntry(entry);zip.write(bytes);zip.closeEntry()
        try
          put("data.pkl",pickleTensor(Vector(0f,0f),Vector(2)))
          put("data/0",if corruptUsed then payload else floats(1f,2f))
          put("data/1",if corruptUsed then floats(1f,2f) else payload)
        finally zip.close()
        val bytes=Files.readAllBytes(path)
        val position=bytes.indices.find(i=>i+payload.length<=bytes.length && payload.indices.forall(j=>bytes(i+j)==payload(j))).get
        bytes(position)=(bytes(position)^1).toByte
        Files.write(path,bytes)
        intercept[IllegalArgumentException] {
          readStateDictSelected(path,PyTorchArchiveLimits(),()=>false,_=>selected)
        }
        intercept[IllegalArgumentException](read(path))
        intercept[IllegalArgumentException](read(path,PyTorchArchiveLimits(allowMissingCrc=true)))
      finally Files.deleteIfExists(path)
  }
  test("selected memoized tensor is decoded once and unselected alias retains metadata") {
    import PyTorchTensorArchive.*
    val path=Files.createTempFile("torch-selected-alias-", ".ckpt")
    def key(s: String): Array[Byte] =
      val data=s.getBytes(UTF_8)
      Array('X'.toByte)++ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data.length).array()++data
    val tensor=pickleTensor(Vector(3f),Vector(1)).drop(2).dropRight(1)
    val pickle=Array[Byte](0x80.toByte,2,'}'.toByte,'('.toByte)++key("a")++tensor++
      Array('q'.toByte,7.toByte)++key("b")++Array('h'.toByte,7.toByte)++
      key("c")++Array('h'.toByte,7.toByte,'u'.toByte,'.'.toByte)
    try
      saveArchive(path,pickle,Map("0"->floats(3f)))
      val values=readStateDictSelected(path,PyTorchArchiveLimits(),()=>false,_!="b").value.asInstanceOf[DictValue].entries
      assertEquals(values("b"),TensorMetadataValue(StorageDType.Float32,Vector(1L)))
      assert(values("a").asInstanceOf[AnyRef] eq values("c").asInstanceOf[AnyRef])
      assertEquals(values("a").asInstanceOf[TensorValue].tensor.values.toVector,Vector(3f))
    finally Files.deleteIfExists(path)
  }
  test("selected state dict keeps metadata but does not allocate unselected tensor arrays") {
    import PyTorchTensorArchive.*
    val bean=java.lang.management.ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assume(bean.isThreadAllocatedMemorySupported,"thread allocation accounting required")
    bean.setThreadAllocatedMemoryEnabled(true)
    val path=Files.createTempFile("torch-selected-", ".ckpt")
    def key(s: String): Array[Byte] =
      val data=s.getBytes(UTF_8)
      Array('X'.toByte)++ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data.length).array()++data
    val count=4*1024*1024
    val tensor=pickleTensor(Vector(0f),Vector(count),storageElements=count).drop(2).dropRight(1)
    val pickle=Array[Byte](0x80.toByte,2,'}'.toByte,'('.toByte)++key("first")++tensor++
      key("second")++tensor++Array('u'.toByte,'.'.toByte)
    try
      saveArchive(path,pickle,Map("0"->new Array[Byte](count*4)))
      val before=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
      val values=PyTorchTensorArchive.readStateDictSelected(path,PyTorchArchiveLimits(maxElements=count),()=>false,
        _=="first").value.asInstanceOf[DictValue].entries
      val allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-before
      assertEquals(values("first").asInstanceOf[TensorValue].tensor.values.length,count)
      assertEquals(values("second"),TensorMetadataValue(StorageDType.Float32,Vector(count.toLong)))
      assert(allocated<24L*1024*1024,s"allocated $allocated bytes for one selected 16 MiB tensor")
      intercept[IllegalArgumentException] {
        PyTorchTensorArchive.readStateDictSelected(path,PyTorchArchiveLimits(maxElements=count,maxTotalTensorElements=count),
          ()=>false,_=="first")
      }
      var checks=0
      intercept[java.util.concurrent.CancellationException] {
        PyTorchTensorArchive.readStateDictSelected(path,PyTorchArchiveLimits(maxElements=count),
          ()=>{checks+=1;checks>=50},_=>false)
      }
    finally Files.deleteIfExists(path)
  }
  test("selectively reads a root state_dict without interpreting trailing training objects") {
    import PyTorchTensorArchive.*
    val path=Files.createTempFile("torch-state-dict-only-", ".ckpt")
    def key(s: String): Array[Byte] =
      val data=s.getBytes(UTF_8)
      Array('X'.toByte)++ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data.length).array()++data
    val tensor=pickleTensor(Vector(2f),Vector(1)).drop(2).dropRight(1)
    val dangerous=Array('c'.toByte)++"os\nsystem\n".getBytes(UTF_8)++Array(')'.toByte,'R'.toByte)
    val pickle=Array[Byte](0x80.toByte,2,'}'.toByte,'('.toByte)++key("state_dict")++
      Array('}'.toByte,'('.toByte)++key("weight")++tensor++Array('u'.toByte)++
      Array('('.toByte)++key("weight2")++tensor++Array('u'.toByte)++
      key("callbacks")++dangerous++Array('u'.toByte,'.'.toByte)
    try
      saveArchive(path,pickle,Map("0"->floats(2f)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
      val state=PyTorchTensorArchive.readStateDict(path,PyTorchArchiveLimits(),()=>false).value.asInstanceOf[DictValue].entries
      assertEquals(state.keySet,Set("weight","weight2"))
      assertEquals(state("weight").asInstanceOf[TensorValue].tensor.values.toVector,Vector(2f))
    finally Files.deleteIfExists(path)
  }
  test("reads integer checkpoint tensors exactly") {
    import PyTorchTensorArchive.*
    for (storage,dtype,bytes,expected) <- Vector(
      ("IntStorage",StorageDType.Int32,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(-7).putInt(42).array(),Vector(-7L,42L)),
      ("LongStorage",StorageDType.Int64,ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN).putLong(9007199254740993L).putLong(-8L).array(),Vector(9007199254740993L,-8L))) do
      val path=Files.createTempFile("torch-integer-", ".ckpt")
      try
        saveArchive(path,pickleTensor(Vector(0f,0f),Vector(2),storageType=storage),Map("0"->bytes))
        val tensor=PyTorchTensorArchive.read(path).value.asInstanceOf[IntegerTensorValue].tensor
        assertEquals(tensor.dtype,dtype)
        assertEquals(tensor.values.toVector,expected)
      finally Files.deleteIfExists(path)
  }
  test("reports first tensor data once and can cancel immediately afterward") {
    import java.util.concurrent.CancellationException
    val path=Files.createTempFile("torch-first-data-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f),Vector(1)),Map("0"->floats(1f)))
      var notices=0
      val read=PyTorchTensorArchive.read(path,PyTorchArchiveLimits(),()=>false,()=>{notices+=1})
      assertEquals(notices,1)
      assertEquals(read.value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor.values.toVector,Vector(1f))
      var cancel=false
      intercept[CancellationException] {
        PyTorchTensorArchive.read(path,PyTorchArchiveLimits(),()=>cancel,()=>{cancel=true})
      }
      assert(cancel)
    finally Files.deleteIfExists(path)
  }
  test("cancellation interrupts referenced tensor decoding and permits a later read") {
    import java.util.concurrent.CancellationException
    val path=Files.createTempFile("torch-cancel-tensor-", ".pt")
    val count=1024*1024
    try
      saveArchive(path,pickleTensor(Vector(0f),Vector(count),storageElements=count),Map("0"->new Array[Byte](count*4)))
      var checks=0
      intercept[CancellationException] {
        PyTorchTensorArchive.read(path,PyTorchArchiveLimits(),()=>{checks+=1; checks>=40})
      }
      assertEquals(checks,40)
      val result=PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
      assertEquals(result.values.length,count)
    finally Files.deleteIfExists(path)
  }
  test("cancellation interrupts validation of an unreferenced storage") {
    import java.util.concurrent.CancellationException
    val path=Files.createTempFile("torch-cancel-unused-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f),Vector(1)),Map("0"->floats(1f),"1"->new Array[Byte](4*1024*1024)))
      var checks=0
      intercept[CancellationException] {
        PyTorchTensorArchive.read(path,PyTorchArchiveLimits(),()=>{checks+=1; checks>=50})
      }
      assertEquals(checks,50)
      assertEquals(PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor.values.toVector,Vector(1f))
    finally Files.deleteIfExists(path)
  }
  test("multiple tensors can read distinct views of a shared storage") {
    import PyTorchTensorArchive.*
    val path=Files.createTempFile("torch-shared-storage-", ".pt")
    def key(s: String): Array[Byte] =
      val data=s.getBytes(UTF_8)
      Array('X'.toByte)++ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data.length).array()++data
    val first=pickleTensor(Vector(0f),Vector(2),storageElements=3,offset=0).drop(2).dropRight(1)
    val second=pickleTensor(Vector(0f),Vector(2),storageElements=3,offset=1).drop(2).dropRight(1)
    val pickle=Array[Byte](0x80.toByte,2,'}'.toByte,'('.toByte)++key("a")++first++key("b")++second++Array('u'.toByte,'.'.toByte)
    try
      saveArchive(path,pickle,Map("0"->floats(1f,2f,3f)))
      val values=PyTorchTensorArchive.read(path).value.asInstanceOf[DictValue].entries
      assertEquals(values("a").asInstanceOf[TensorValue].tensor.values.toVector,Vector(1f,2f))
      assertEquals(values("b").asInstanceOf[TensorValue].tensor.values.toVector,Vector(2f,3f))
    finally Files.deleteIfExists(path)
  }
  test("streamed tensors preserve storage offsets across buffer boundaries") {
    val path=Files.createTempFile("torch-storage-offset-", ".pt")
    val count=65536/4+3
    val bytes=ByteBuffer.allocate((count+2)*4).order(ByteOrder.LITTLE_ENDIAN)
    for i <- 0 until count+2 do bytes.putFloat(i.toFloat)
    try
      saveArchive(path,pickleTensor(Vector(0f),Vector(count),storageElements=count+2,offset=1),Map("0"->bytes.array()))
      val tensor=PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
      for i <- Vector(0,1,16382,16383,16384,count-1) do assertEquals(tensor.values(i),(i+1).toFloat)
    finally Files.deleteIfExists(path)
  }
  test("large float storage is decoded without an additional full storage array") {
    val bean=java.lang.management.ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assume(bean.isThreadAllocatedMemorySupported,"thread allocation accounting required")
    bean.setThreadAllocatedMemoryEnabled(true)
    val path=Files.createTempFile("torch-streamed-storage-", ".pt")
    val count=4*1024*1024
    try
      saveArchive(path,pickleTensor(Vector(0f),Vector(count),storageElements=count),Map("0"->new Array[Byte](count*4)))
      val before=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
      val tensor=PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxElements=count.toLong)).value
        .asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
      val allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-before
      assertEquals(tensor.values.length,count)
      assert(allocated<24L*1024*1024,s"allocated $allocated bytes for 16 MiB of tensor data")
    finally Files.deleteIfExists(path)
  }
  test("large unused storage is validated without materializing it on the heap") {
    val bean=java.lang.management.ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assume(bean.isThreadAllocatedMemorySupported,"thread allocation accounting required")
    bean.setThreadAllocatedMemoryEnabled(true)
    val path=Files.createTempFile("torch-unused-storage-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f),Vector(1)),Map("0"->floats(1f),"1"->new Array[Byte](16*1024*1024)))
      val before=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
      val result=PyTorchTensorArchive.read(path)
      val allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-before
      assertEquals(result.value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor.values.toVector,Vector(1f))
      assert(allocated<4L*1024*1024,s"allocated $allocated bytes for unused storage")
      intercept[IllegalArgumentException] {
        PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxTotalUncompressedBytes=1024*1024))
      }
    finally Files.deleteIfExists(path)
  }
  test("limits cumulative decoded tensor elements before allocating each tensor") {
    import PyTorchTensorArchive.*
    val path=Files.createTempDirectory("torch-total-limit").resolve("weights.pt")
    def t=TensorValue(StoredTensor(StorageDType.Float32,Vector(2L),Array(1f,2f)))
    PyTorchTensorArchiveWriter.write(path,DictValue(Map("a"->t,"b"->t)))
    intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxTotalTensorElements=3)))
    assert(PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxTotalTensorElements=4)).value.isInstanceOf[DictValue])
  }
  test("round trips binary floating point metadata") {
    import PyTorchTensorArchive.*
    val path=Files.createTempDirectory("torch-float-metadata").resolve("metadata.pt")
    val value=DictValue(Map("rate"->FloatValue(0.0001),"negative"->FloatValue(-3.5)))
    PyTorchTensorArchiveWriter.write(path,value)
    assertEquals(PyTorchTensorArchive.read(path).value,value)
  }
  private def pickleTensor(values: Vector[Float],shape: Vector[Int],storageType: String="FloatStorage",protocol: Int=2,
      storageElements: Int = -1,offset: Int = 0,strideOverride: Option[Vector[Int]]=None): Array[Byte] =
    val bytes=new ByteArrayOutputStream
    val out=new DataOutputStream(bytes)
    def op(value: Int): Unit=out.writeByte(value)
    def raw(value: String): Unit=out.write(value.getBytes(UTF_8))
    def shortUnicode(value: String): Unit=
      val data=value.getBytes(UTF_8); require(data.length<256); op(0x8c); op(data.length); out.write(data)
    def global(module: String,name: String): Unit=
      if protocol>=4 then { shortUnicode(module); shortUnicode(name); op(0x93) }
      else { op('c'); raw(module); op('\n'); raw(name); op('\n') }
    def unicode(value: String): Unit=
      val data=value.getBytes(UTF_8); op('X'); out.writeInt(Integer.reverseBytes(data.length)); out.write(data)
    def binInt(value: Int): Unit={ op('J'); out.writeInt(Integer.reverseBytes(value)) }
    def tuple(values: Vector[Int]): Unit={ op('('); values.foreach(v=>{ op('J'); out.writeInt(Integer.reverseBytes(v)) }); op('t') }
    global("torch._utils","_rebuild_tensor_v2")
    op('(')
    op('(')
    unicode("storage")
    global("torch",storageType)
    unicode("0")
    unicode("cpu")
    binInt(if storageElements<0 then values.length else storageElements)
    op('t'); op('Q')
    binInt(offset)
    tuple(shape)
    tuple(strideOverride.getOrElse(shape.indices.map(i=>shape.drop(i+1).product).toVector))
    op(0x89)
    global("collections","OrderedDict"); op(')'); op('R')
    op('t'); op('R'); op('.')
    out.flush()
    val body=bytes.toByteArray
    if protocol>=4 then Array(0x80.toByte,protocol.toByte,0x95.toByte)++ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(body.length.toLong).array()++body
    else Array(0x80.toByte,protocol.toByte)++body

  private def saveArchive(path: Path,pickle: Array[Byte],storages: Map[String,Array[Byte]]): Unit=
    val output=new ZipOutputStream(Files.newOutputStream(path))
    try
      def entry(name: String,data: Array[Byte]): Unit=
        output.putNextEntry(new ZipEntry(name)); output.write(data); output.closeEntry()
      entry("archive/data.pkl",pickle)
      storages.foreach((key,data)=>entry(s"archive/data/$key",data))
      entry("archive/version","3\n".getBytes(UTF_8))
    finally output.close()

  private def floats(values: Float*): Array[Byte]=
    val buffer=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN)
    values.foreach(buffer.putFloat)
    buffer.array()

  test("reads a plain F32 tensor from a current PyTorch ZIP archive") {
    val path=Files.createTempFile("torch-tensor-archive-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1.25f,-2.5f),Vector(2)),Map("0"->floats(1.25f,-2.5f)))
      val archive=PyTorchTensorArchive.read(path)
      val tensor=archive.value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
      assertEquals(tensor.dtype,PyTorchTensorArchive.StorageDType.Float32)
      assertEquals(tensor.shape,Vector(2L))
      assertEquals(tensor.values.toVector,Vector(1.25f,-2.5f))
    finally Files.deleteIfExists(path)
  }

  test("reads protocol 4 and 5 frames without dynamic global loading") {
    Vector(4,5).foreach { protocol =>
      val path=Files.createTempFile("torch-protocol-", ".pt")
      try
        saveArchive(path,pickleTensor(Vector(0.5f,-0.25f),Vector(2),protocol=protocol),Map("0"->floats(0.5f,-0.25f)))
        val tensor=PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
        assertEquals(tensor.values.toVector,Vector(0.5f,-0.25f))
      finally Files.deleteIfExists(path)
    }
  }

  test("converts Float16 and BFloat16 storages to Float32 values") {
    val half=ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putShort(0x3c00.toShort).putShort(0xc000.toShort).array()
    val bf16=ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putShort(0x3f80.toShort).putShort(0xc000.toShort).array()
    Vector(("HalfStorage",PyTorchTensorArchive.StorageDType.Float16,half),
      ("BFloat16Storage",PyTorchTensorArchive.StorageDType.BFloat16,bf16)).foreach { (kind,dtype,raw) =>
      val path=Files.createTempFile("torch-half-", ".pt")
      try
        saveArchive(path,pickleTensor(Vector(1f,-2f),Vector(2),storageType=kind),Map("0"->raw))
        val tensor=PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
        assertEquals(tensor.dtype,dtype)
        assertEquals(tensor.values.toVector,Vector(1f,-2f))
      finally Files.deleteIfExists(path)
    }
  }

  test("reads tensors nested in a primitive string-keyed mapping") {
    val path=Files.createTempFile("torch-tensor-map-", ".pt")
    try
      val prefix=Array[Byte](0x80.toByte,2,'}'.toByte)
      val key="emb_params".getBytes(UTF_8)
      val keyBytes=Array('X'.toByte)++ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(key.length).array()++key
      val suffix=Array('s'.toByte,'.'.toByte)
      saveArchive(path,prefix++keyBytes++pickleTensor(Vector(3f,4f),Vector(1,2)).drop(2).dropRight(1)++suffix,
        Map("0"->floats(3f,4f)))
      val archive=PyTorchTensorArchive.read(path)
      val mapping=archive.value.asInstanceOf[PyTorchTensorArchive.DictValue].entries
      assert(mapping("emb_params").isInstanceOf[PyTorchTensorArchive.TensorValue])
    finally Files.deleteIfExists(path)
  }

  test("rejects a pickle reducer outside the tensor allowlist") {
    val path=Files.createTempFile("torch-pickle-global-", ".pt")
    try
      val payload=Array[Byte](0x80.toByte,2,'c'.toByte)++"os\nsystem\n".getBytes(UTF_8)++Array(')'.toByte,'R'.toByte,'.'.toByte)
      saveArchive(path,payload,Map.empty)
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
    finally Files.deleteIfExists(path)
  }

  test("rejects ZIP traversal paths, malformed frame bounds and configured input limits") {
    val path=Files.createTempFile("torch-archive-limits-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f),Vector(1)),Map("0"->floats(1f)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxArchiveBytes=1)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxEntryBytes=1)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxPickleBytes=1)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxPickleOps=1)))
      val frame=pickleTensor(Vector(1f),Vector(1),protocol=4)
      ByteBuffer.wrap(frame).order(ByteOrder.LITTLE_ENDIAN).putLong(3,1L)
      saveArchive(path,frame,Map("0"->floats(1f)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
      val zip=new ZipOutputStream(Files.newOutputStream(path))
      try
        zip.putNextEntry(new ZipEntry("archive/../escape"));zip.write(1);zip.closeEntry()
        zip.putNextEntry(new ZipEntry("archive/data.pkl"));zip.write(pickleTensor(Vector(1f),Vector(1)));zip.closeEntry()
      finally zip.close()
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
    finally Files.deleteIfExists(path)
  }

  test("rejects non-ZIP legacy PyTorch archives with the reader's format error") {
    val path=Files.createTempFile("torch-old-format-", ".pt")
    try
      Files.write(path,Array[Byte](0x80.toByte,2,'c'.toByte,'t'.toByte,'o'.toByte,'r'.toByte,'c'.toByte,'h'.toByte))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
    finally Files.deleteIfExists(path)
  }

  test("enforces tensor element limits before materializing storage values") {
    val path=Files.createTempFile("torch-tensor-limit-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f,2f),Vector(2)),Map("0"->floats(1f,2f)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path,PyTorchArchiveLimits(maxElements=1)))
    finally Files.deleteIfExists(path)
  }

  test("gathers a dense permuted (channels-last style) view into logical order") {
    val path=Files.createTempFile("torch-tensor-permuted-", ".pt")
    try
      saveArchive(path,pickleTensor(Vector(1f,2f,3f,4f,5f,6f),Vector(2,3),strideOverride=Some(Vector(1,2))),Map("0"->floats(1f,2f,3f,4f,5f,6f)))
      val tensor=PyTorchTensorArchive.read(path).value.asInstanceOf[PyTorchTensorArchive.TensorValue].tensor
      assertEquals(tensor.values.toSeq,Seq(1f,3f,5f,2f,4f,6f))
    finally Files.deleteIfExists(path)
  }

  test("rejects non-contiguous tensor views and views outside their storage") {
    val path=Files.createTempFile("torch-tensor-view-", ".pt")
    try
      val data=floats(1f,2f,3f,4f)
      saveArchive(path,pickleTensor(Vector(1f,3f),Vector(2),strideOverride=Some(Vector(2))),Map("0"->data))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
      saveArchive(path,pickleTensor(Vector(1f,2f),Vector(2),storageElements=2,offset=1),Map("0"->floats(1f,2f)))
      intercept[IllegalArgumentException](PyTorchTensorArchive.read(path))
    finally Files.deleteIfExists(path)
  }
