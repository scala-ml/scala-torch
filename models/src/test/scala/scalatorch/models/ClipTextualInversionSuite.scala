package scalatorch.models

import scalatorch.{Shape,Tensor}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path}

class ClipTextualInversionSuite extends munit.FunSuite:
  test("PyTorch editor stages changed coordinates while preserving dtype and metadata") {
    import scalatorch.io.PyTorchTensorArchive.*
    val dir=Files.createTempDirectory("native-pt-edit")
    try
      val source=dir.resolve("source.pt");val stage=dir.resolve("stage.pt")
      val value=DictValue(Map("string_to_param"->DictValue(Map("*"->TensorValue(StoredTensor(StorageDType.Float16,
        Vector(1,2),Array(1f,2f))))),"step"->IntegerValue(42),"name"->StringValue("source")))
      scalatorch.io.PyTorchTensorArchiveWriter.write(source,value)
      val original=Files.readAllBytes(source)
      val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(original))
      val baseline=Map("clip_l"->Vector(Vector(1.0,2.0)))
      val edited=Map("clip_l"->Vector(Vector(1.0004882812500002,2.0)))
      ClipTextualInversion.stagePyTorchEditDraft(source,stage,edited,baseline,hash)
      val result=scalatorch.io.PyTorchTensorArchive.read(stage).value.asInstanceOf[DictValue].entries
      assertEquals(result("step"),IntegerValue(42))
      val tensor=result("string_to_param").asInstanceOf[DictValue].entries("*").asInstanceOf[TensorValue].tensor
      assertEquals(tensor.dtype,StorageDType.Float16)
      assertEquals(tensor.values.toVector,Vector(1.0009765625f,2f))
      sys.props.get("scalatorch.pytorch.edit.test.output").foreach(output=>Files.copy(stage,Path.of(output)))
      assertEquals(Files.readAllBytes(source).toVector,original.toVector)
      intercept[IllegalArgumentException](ClipTextualInversion.stagePyTorchEditDraft(source,dir.resolve("bad.pt"),edited,baseline,"0"*64))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("editor drafts preserve double precision through the model layer") {
    val dir=Files.createTempDirectory("native-embedding-precision")
    try
      val source=dir.resolve("source.safetensors");val stage=dir.resolve("stage.safetensors")
      val header="{\"emb_params\":{\"dtype\":\"F64\",\"shape\":[1,2],\"data_offsets\":[0,16]}}".getBytes(UTF_8)
      val data=ByteBuffer.allocate(8+header.length+16).order(ByteOrder.LITTLE_ENDIAN)
        .putLong(header.length.toLong).put(header).putDouble(1.0).putDouble(2.0).array()
      Files.write(source,data)
      val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(data))
      val baseline=Map("clip_l"->Vector(Vector(1.0,2.0)))
      val value=1.0000000000000002
      val edited=Map("clip_l"->Vector(Vector(value,2.0)))
      ClipTextualInversion.stageSafetensorsEditDraft(source,stage,edited,baseline,hash)
      val actual=ByteBuffer.wrap(Files.readAllBytes(stage)).order(ByteOrder.LITTLE_ENDIAN)
      assertEquals(actual.getDouble(8+header.length),value)
      assertEquals(actual.getDouble(8+header.length+8),2.0)
      assertEquals(Files.readAllBytes(source).toVector,data.toVector)
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("embedding editor stages changed SDXL coordinates without rewriting other tensors") {
    val dir=Files.createTempDirectory("native-embedding-stage")
    try
      val source=dir.resolve("embedding.safetensors");val stage=dir.resolve("stage.safetensors")
      val left=Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(2,2)))
      val right=Tensor.fromFloats(Array(5f,6f,7f,8f,9f,10f),Shape.from(Seq(2,3)))
      val extra=Tensor.fromLongs(Array(99L),Shape.from(Seq(1)))
      scalatorch.io.SafeTensorsWriter.write(source,Map("clip_l"->scalatorch.io.SafeTensorWrite.F16(left),
        "clip_g"->scalatorch.io.SafeTensorWrite.BF16(right),"extra"->scalatorch.io.SafeTensorWrite.I64(extra)),
        Map("step"->"99","custom"->"keep"))
      val original=Files.readAllBytes(source)
      val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(original))
      val baseline=ClipTextualInversion(Vector(Array(1f,2f),Array(3f,4f)),Some(Vector(Array(5f,6f,7f),Array(8f,9f,10f))))
      val revised=baseline.copy(clipL=Vector(Array(1f,2f),Array(3f,4.5f)),
        clipG=Some(Vector(Array(5f,6f,7f),Array(8f,9.5f,10f))))
      ClipTextualInversion.stageSafetensorsEdit(source,stage,revised,baseline,hash)
      assertEquals(Files.readAllBytes(source).toVector,original.toVector)
      val archive=scalatorch.io.SafeTensors.open(stage)
      try
        assertEquals(archive.metadata,Map("step"->"99","custom"->"keep"))
        assertEquals(archive.readFloat("clip_l").toFloatArray.toVector,Vector(1f,2f,3f,4.5f))
        assertEquals(archive.readFloat("clip_g").toFloatArray.toVector,Vector(5f,6f,7f,8f,9.5f,10f))
        assertEquals(archive.readLong("extra").toLongArray.toVector,Vector(99L))
      finally archive.close()
      val wrong=dir.resolve("wrong.safetensors")
      intercept[IllegalArgumentException](ClipTextualInversion.stageSafetensorsEdit(source,wrong,revised.copy(clipG=None),baseline,hash))
      assert(!Files.exists(wrong))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("NUI SD1 and SDXL embeddings save and reload as safetensors") {
    val dir=Files.createTempDirectory("native-embedding-save")
    try
      val sd1=ClipTextualInversion(Vector(Array(1f,2f,3f),Array(4f,5f,6f)),None)
      val file=dir.resolve("sd1.safetensors")
      ClipTextualInversion.writeSafetensors(file,sd1,step=Some(42))
      val loaded=ClipTextualInversion.readSafetensors(file,false,3,0)
      assertEquals(loaded.clipL.map(_.toVector),sd1.clipL.map(_.toVector))
      val archive=scalatorch.io.SafeTensors.open(file)
      try assertEquals(archive.metadata,Map("step"->"42")) finally archive.close()
      intercept[java.nio.file.FileAlreadyExistsException](ClipTextualInversion.writeSafetensors(file,sd1))
      val xl=ClipTextualInversion(sd1.clipL,Some(Vector(Array(7f,8f),Array(9f,10f))))
      val xlFile=dir.resolve("sdxl.safetensors")
      ClipTextualInversion.writeSafetensors(xlFile,xl)
      val loadedXl=ClipTextualInversion.readSafetensors(xlFile,true,3,2)
      assertEquals(loadedXl.clipL.map(_.toVector),xl.clipL.map(_.toVector))
      assertEquals(loadedXl.clipG.get.map(_.toVector),xl.clipG.get.map(_.toVector))
      sys.props.get("scalatorch.embedding.export.dir").foreach { directory =>
        Files.copy(file,Path.of(directory).resolve("sd1.safetensors"))
        Files.copy(xlFile,Path.of(directory).resolve("sdxl.safetensors"))
      }
      intercept[IllegalArgumentException](ClipTextualInversion.writeSafetensors(dir.resolve("bad.safetensors"),
        ClipTextualInversion(Vector(Array(Float.NaN)),None)))
      intercept[IllegalArgumentException](ClipTextualInversion.writeSafetensors(dir.resolve("bad.safetensors"),
        ClipTextualInversion(sd1.clipL,Some(Vector(Array(1f))))))
    finally
      val paths=Files.walk(dir);try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  private def write(path: Path,entries: Vector[(String,Vector[Int],Array[Float])]): Unit =
    var offset=0L
    val headers=entries.map { (name,shape,values) =>
      require(shape.product==values.length)
      val begin=offset;offset+=values.length.toLong*4
      s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${shape.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }
    val header=headers.mkString("{",",","}").getBytes(UTF_8)
    val data=ByteBuffer.allocate(Math.toIntExact(offset)).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  test("safetensors reader accepts one- and multi-vector SD1 CLIP embeddings") {
    val path=Files.createTempFile("ti-sd1", ".safetensors")
    try
      write(path,Vector(("emb_params",Vector(3),Array(1f,2f,3f))))
      val one=ClipTextualInversion.readSafetensors(path,isSdxl=false,clipLWidth=3,clipGWidth=0)
      assertEquals(one.clipL.map(_.toVector),Vector(Vector(1f,2f,3f)))
      write(path,Vector(("string_to_param",Vector(2,3),Array(1f,2f,3f,4f,5f,6f))))
      val many=ClipTextualInversion.readSafetensors(path,isSdxl=false,clipLWidth=3,clipGWidth=0)
      assertEquals(many.clipL.map(_.toVector),Vector(Vector(1f,2f,3f),Vector(4f,5f,6f)))
      assertEquals(many.clipG,None)
    finally Files.deleteIfExists(path)
  }

  test("safetensors reader requires matching CLIP-L and CLIP-G vectors for SDXL") {
    val path=Files.createTempFile("ti-sdxl", ".safetensors")
    try
      write(path,Vector(("clip_l",Vector(2,3),Array(1f,2f,3f,4f,5f,6f)),
        ("clip_g",Vector(2,4),Array(7f,8f,9f,10f,11f,12f,13f,14f))))
      val result=ClipTextualInversion.readSafetensors(path,isSdxl=true,clipLWidth=3,clipGWidth=4)
      assertEquals(result.clipL.size,2)
      assertEquals(result.clipG.map(_.map(_.toVector)),Some(Vector(Vector(7f,8f,9f,10f),Vector(11f,12f,13f,14f))))
      write(path,Vector(("clip_l",Vector(2,3),Array.fill(6)(1f)),("clip_g",Vector(1,4),Array.fill(4)(2f))))
      intercept[IllegalArgumentException](ClipTextualInversion.readSafetensors(path,true,3,4))
    finally Files.deleteIfExists(path)
  }

  test("safetensors reader rejects width, nonfinite and excessive-vector mismatches") {
    val path=Files.createTempFile("ti-invalid", ".safetensors")
    try
      write(path,Vector(("embedding",Vector(2),Array(1f,2f))))
      intercept[IllegalArgumentException](ClipTextualInversion.readSafetensors(path,false,3,0))
      write(path,Vector(("embedding",Vector(3),Array(1f,Float.NaN,3f))))
      intercept[IllegalArgumentException](ClipTextualInversion.readSafetensors(path,false,3,0))
      write(path,Vector(("embedding",Vector(257,3),Array.fill(257*3)(1f))))
      intercept[IllegalArgumentException](ClipTextualInversion.readSafetensors(path,false,3,0))
    finally Files.deleteIfExists(path)
  }

  test("PyTorch reader accepts direct tensors and the two NUI SD1 wrappers") {
    val path=Files.createTempFile("ti-pt-sd1", ".pt")
    try
      PyTorchEmbeddingFixture.write(path,"tensor",("0",Vector(1f,2f,3f),Vector(3)))
      val direct=ClipTextualInversion.readPyTorch(path,false,3,0)
      assertEquals(direct.clipL.map(_.toVector),Vector(Vector(1f,2f,3f)))
      PyTorchEmbeddingFixture.write(path,"emb_params",("emb_params",Vector(1f,2f,3f,4f,5f,6f),Vector(2,3)))
      val wrapped=ClipTextualInversion.readPyTorch(path,false,3,0)
      assertEquals(wrapped.clipL.map(_.toVector),Vector(Vector(1f,2f,3f),Vector(4f,5f,6f)))
      PyTorchEmbeddingFixture.write(path,"string_to_param",("*",Vector(7f,8f,9f),Vector(3)))
      val legacy=ClipTextualInversion.readPyTorch(path,false,3,0)
      assertEquals(legacy.clipL.map(_.toVector),Vector(Vector(7f,8f,9f)))
    finally Files.deleteIfExists(path)
  }

  test("PyTorch reader requires exactly matched SDXL CLIP-L and CLIP-G tensors") {
    val path=Files.createTempFile("ti-pt-sdxl", ".pt")
    try
      PyTorchEmbeddingFixture.write(path,"sdxl",("clip_l",Vector(1f,2f,3f,4f,5f,6f),Vector(2,3)),
        ("clip_g",Vector(7f,8f,9f,10f,11f,12f,13f,14f),Vector(2,4)))
      val result=ClipTextualInversion.readPyTorch(path,true,3,4)
      assertEquals(result.clipL.map(_.toVector),Vector(Vector(1f,2f,3f),Vector(4f,5f,6f)))
      assertEquals(result.clipG.map(_.map(_.toVector)),Some(Vector(Vector(7f,8f,9f,10f),Vector(11f,12f,13f,14f))))
      intercept[IllegalArgumentException](ClipTextualInversion.readPyTorch(path,false,3,0))
      intercept[IllegalArgumentException](ClipTextualInversion.readPyTorch(path,true,2,4))
    finally Files.deleteIfExists(path)
  }
