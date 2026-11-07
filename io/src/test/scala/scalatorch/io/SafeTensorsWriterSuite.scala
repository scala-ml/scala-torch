package scalatorch.io

import scalatorch.*
import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import scala.util.Using

class SafeTensorsWriterSuite extends munit.FunSuite:
  test("large float writes avoid another full-size heap allocation") {
    val bean=java.lang.management.ManagementFactory.getThreadMXBean.asInstanceOf[com.sun.management.ThreadMXBean]
    assume(bean.isThreadAllocatedMemorySupported,"thread allocation accounting required")
    bean.setThreadAllocatedMemoryEnabled(true)
    val path=Files.createTempFile("safetensors-stream", ".safetensors")
    val tensor=Tensor.fromFloats(Array.fill(4000000)(1.25f),Shape.from(Seq(4000000L)))
    try
      SafeTensorsWriter.write(path,Map("warm"->SafeTensorWrite.F16(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L))))),overwrite=true)
      val before=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())
      SafeTensorsWriter.write(path,Map("weight"->SafeTensorWrite.F16(tensor)),overwrite=true)
      val allocated=bean.getThreadAllocatedBytes(Thread.currentThread().threadId())-before
      assert(allocated<8L*1024*1024,s"writer allocated $allocated bytes for a 16 MB tensor")
      Using.resource(SafeTensors.open(path))(file=>assertEquals(file.readFloat("weight").toFloatArray.last,1.25f))
    finally Files.deleteIfExists(path)
  }
  test("float writer streams offset, reverse and broadcast views in logical order") {
    val path=Files.createTempFile("safetensors-views", ".safetensors")
    val source=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(2L,3L)))
    val reverse=source.slice(1,2,-1,-1)
    val offset=source.slice(0,1,2)
    val expanded=source.slice(0,0,1).broadcastTo(Shape.from(Seq(2L,3L)))
    try
      SafeTensorsWriter.write(path,Map("reverse"->SafeTensorWrite.F16(reverse),"offset"->SafeTensorWrite.F32(offset),
        "broadcast"->SafeTensorWrite.BF16(expanded)),overwrite=true)
      Using.resource(SafeTensors.open(path)) { file =>
        assertEquals(file.readFloat("reverse").toFloatArray.toVector,Vector(3f,2f,1f,6f,5f,4f))
        assertEquals(file.readFloat("offset").toFloatArray.toVector,Vector(4f,5f,6f))
        assertEquals(file.readFloat("broadcast").toFloatArray.toVector,Vector(1f,2f,3f,1f,2f,3f))
      }
    finally Files.deleteIfExists(path)
  }
  private val matrix=Tensor.fromFloats(Array(1f,2f,3f,4f,5f,6f),Shape.from(Seq(2,3)))
  private def values=Map[String,SafeTensorWrite](
    "float"->SafeTensorWrite.F32(matrix.transpose(0,1)),
    "half"->SafeTensorWrite.F16(Tensor.fromFloats(Array(1f,-2f,0.000000059604645f,65504f),Shape.from(Seq(4)))),
    "bfloat"->SafeTensorWrite.BF16(Tensor.fromFloats(Array(1f,-2f,0.5f),Shape.from(Seq(3)))),
    "double"->SafeTensorWrite.F64(Tensor.fromDoubles(Array(1.25,-2.5),Shape.from(Seq(2)))),
    "long"->SafeTensorWrite.I64(Tensor.fromLongs(Array(9007199254740993L,-7L),Shape.from(Seq(2)))),
    "bool"->SafeTensorWrite.Bool(Tensor.fromBooleans(Array(false,true),Shape.from(Seq(2)))),
    "empty"->SafeTensorWrite.F32(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,3)))),
    "scalar"->SafeTensorWrite.F32(Tensor.fromFloats(Array(-0f),Shape.from(Seq.empty))))
  test("writer preserves shapes dtypes metadata and noncontiguous logical order") {
    val dir=Files.createTempDirectory("safetensors-writer")
    try
      val path=dir.resolve("result.safetensors")
      SafeTensorsWriter.write(path,values,Map("format"->"pt","日本語"->"猫"))
      Using.resource(SafeTensors.open(path)) { archive =>
        assertEquals(archive.metadata,Map("format"->"pt","日本語"->"猫"))
        assertEquals(archive.readFloat("float").shape.dims,Vector(3L,2L))
        assertEquals(archive.readFloat("float").toFloatArray.toVector,Vector(1f,4f,2f,5f,3f,6f))
        assertEquals(archive.tensors("half").dtype,SafeDType.F16)
        assertEquals(archive.readFloat("half").toFloatArray.toVector,Vector(1f,-2f,math.pow(2,-24).toFloat,65504f))
        assertEquals(archive.tensors("bfloat").dtype,SafeDType.BF16)
        assertEquals(archive.readFloat("bfloat").toFloatArray.toVector,Vector(1f,-2f,.5f))
        assertEquals(archive.readDouble("double").toDoubleArray.toVector,Vector(1.25,-2.5))
        assertEquals(archive.readLong("long").toLongArray.toVector,Vector(9007199254740993L,-7L))
        assertEquals(archive.readBoolean("bool").toBooleanArray.toVector,Vector(false,true))
        assertEquals(archive.readFloat("empty").shape.dims,Vector(0L,3L))
        assertEquals(java.lang.Float.floatToRawIntBits(archive.readFloat("scalar").toFloatArray.head),0x80000000)
      }
      sys.props.get("scalatorch.writer.export").foreach(destination => Files.copy(path,Path.of(destination)))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("failed writes and collisions never change the existing destination") {
    val dir=Files.createTempDirectory("safetensors-writer-errors")
    try
      val path=dir.resolve("existing.safetensors");Files.writeString(path,"original")
      intercept[java.nio.file.FileAlreadyExistsException](SafeTensorsWriter.write(path,values))
      assertEquals(Files.readString(path),"original")
      intercept[CancellationException](SafeTensorsWriter.write(path,values,overwrite=true,cancelled=()=>true))
      assertEquals(Files.readString(path),"original")
      intercept[IllegalArgumentException](SafeTensorsWriter.write(dir.resolve("bad"),Map("__metadata__"->values("float"))))
      intercept[IllegalArgumentException](SafeTensorsWriter.write(dir.resolve("bad"),Map("\ud800"->values("float"))))
      intercept[IllegalArgumentException](SafeTensorsWriter.write(dir.resolve("bad"),values,limits=ReadLimits(maxHeaderBytes=64)))
      val large=SafeTensorWrite.F32(Tensor.fromFloats(Array.fill(40000)(1f),Shape.from(Seq(40000))))
      var polls=0
      intercept[CancellationException](SafeTensorsWriter.write(path,Map("large"->large),overwrite=true,cancelled=()=>{polls+=1;polls>=5}))
      assert(polls>=5)
      assertEquals(Files.readString(path),"original")
      val paths=Files.list(dir);try assertEquals(paths.count(),1L) finally paths.close()
      SafeTensorsWriter.write(path,Map("value"->values("float")),overwrite=true)
      Using.resource(SafeTensors.open(path))(archive=>assertEquals(archive.tensors.keySet,Set("value")))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("raw tensor copies preserve integer bytes and validate their length") {
    val dir=Files.createTempDirectory("safetensors-raw")
    try
      val path=dir.resolve("raw.safetensors")
      val bytes=Array[Byte](1,2,3,4,-1,-2,-3,-4)
      val raw=SafeTensorWrite.Raw(SafeDType.I32,Shape.from(Seq(2L)),bytes)
      SafeTensorsWriter.write(path,Map("integer"->raw))
      Using.resource(SafeTensors.open(path)) { archive =>
        assertEquals(archive.tensors("integer").dtype,SafeDType.I32)
        assertEquals(archive.readRaw("integer").toVector,bytes.toVector)
      }
      intercept[IllegalArgumentException](SafeTensorsWriter.write(dir.resolve("bad.safetensors"),
        Map("bad"->SafeTensorWrite.Raw(SafeDType.I32,Shape.from(Seq(2L)),Array[Byte](1)))))
      assert(!Files.exists(dir.resolve("bad.safetensors")))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("planned writes load tensors one at a time and preserve destination on failure") {
    val dir=Files.createTempDirectory("safetensors-planned")
    try
      val path=dir.resolve("result.safetensors")
      val loaded=scala.collection.mutable.ArrayBuffer.empty[String]
      def plan(name: String,value: Float)=SafeTensorPlan(SafeDType.F32,Shape.from(Seq(1L)),()=>{
        loaded+=name
        SafeTensorWrite.F32(Tensor.fromFloats(Array(value),Shape.from(Seq(1L))))
      })
      SafeTensorsWriter.writePlanned(path,Map("z"->plan("z",2f),"a"->plan("a",1f)),Map("format"->"pt"))
      assertEquals(loaded.toVector,Vector("a","z"))
      Using.resource(SafeTensors.open(path)) { archive =>
        assertEquals(archive.readFloat("a").toFloatArray.toVector,Vector(1f))
        assertEquals(archive.readFloat("z").toFloatArray.toVector,Vector(2f))
        assertEquals(archive.metadata,Map("format"->"pt"))
      }
      val original=Files.readAllBytes(path).toVector
      val bad=SafeTensorPlan(SafeDType.F32,Shape.from(Seq(1L)),()=>throw new IllegalStateException("loader failed"))
      intercept[IllegalStateException](SafeTensorsWriter.writePlanned(path,Map("a"->plan("a",1f),"z"->bad),overwrite=true))
      assertEquals(Files.readAllBytes(path).toVector,original)
      val wrong=SafeTensorPlan(SafeDType.F16,Shape.from(Seq(1L)),()=>SafeTensorWrite.F32(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L)))))
      intercept[IllegalArgumentException](SafeTensorsWriter.writePlanned(path,Map("wrong"->wrong),overwrite=true))
      assertEquals(Files.readAllBytes(path).toVector,original)
      var commitCalled=false
      intercept[CancellationException](SafeTensorsWriter.writePlanned(path,Map("a"->plan("a",1f)),overwrite=true,
        beforePublish=()=>{commitCalled=true;throw new CancellationException("cancel at publication")}))
      assert(commitCalled)
      assertEquals(Files.readAllBytes(path).toVector,original)
      val contents=Files.list(dir);try assertEquals(contents.count(),1L) finally contents.close()
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("half encodings preserve signed zero, ties and nonfinite values") {
    val dir=Files.createTempDirectory("safetensors-writer-half")
    try
      val input=Array(0f,-0f,1f,java.lang.Float.intBitsToFloat(0x3f808000),
        Float.PositiveInfinity,Float.NegativeInfinity,java.lang.Float.intBitsToFloat(0x7f800001))
      val tensor=Tensor.fromFloats(input,Shape.from(Seq(input.length)))
      val path=dir.resolve("half.safetensors")
      SafeTensorsWriter.write(path,Map("half"->SafeTensorWrite.F16(tensor),"bfloat"->SafeTensorWrite.BF16(tensor)))
      Using.resource(SafeTensors.open(path)) { archive =>
        for name <- Seq("half","bfloat") do
          val got=archive.readFloat(name).toFloatArray
          assertEquals(java.lang.Float.floatToRawIntBits(got(1)),0x80000000)
          assertEquals(got(3),if name=="bfloat" then 1f else 1.00390625f)
          assertEquals(got(4),Float.PositiveInfinity);assertEquals(got(5),Float.NegativeInfinity)
          assert(got(6).isNaN)
      }
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
