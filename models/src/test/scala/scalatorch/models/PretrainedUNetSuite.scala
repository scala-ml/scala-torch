package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.Module
import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8

class PretrainedUNetSuite extends munit.FunSuite:
  test("complete pinned public config parses") {
    val stream=getClass.getResourceAsStream("/tiny-sd/config.json")
    val json=try new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8) finally stream.close()
    val c=UNetConfig.fromJson(json)
    assertEquals(c.contextWidth,32)
    assertEquals(c.groups,32)
  }
  test("published tiny U-Net loads every weight and runs frozen odd-sized inference") {
    val dir=sys.props.get("scalatorch.unet.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.unet.dir to pinned tiny-sd-pipe/unet")
    pinned(dir.get)
    val m=PretrainedUNet.load(dir.get)
    assertEquals(Module.parameters(m).size,304)
    assert(Module.parameters(m).forall(!_.trainable))
    val x=Variable(Tensor.fromFloats(Array.tabulate(4*3*5)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,3,5))),false)
    val c=Variable(Tensor.fromFloats(Array.tabulate(3*32)(i => math.cos(i*.13).toFloat),Shape.from(Seq(1,3,32))),false)
    val y=m(x,Tensor.fromFloats(Array(7.5f),Shape.from(Seq(1))),c)
    assertEquals(y.data.shape,x.data.shape)
    assert(!y.requiresGrad)
    assert(y.data.toFloatArray.forall(_.isFinite))
    assert(y.data.toFloatArray.exists(v => math.abs(v)>1e-5))
    intercept[IllegalArgumentException](PretrainedUNet.load(dir.get,maxParameters=1))
  }

  private val hashes=Map(
    "config.json"->"2f80e0a35261d11d52c976b8bf15ec45ab414f46c5506e7a1f0c48851382ef06",
    "diffusion_pytorch_model.safetensors"->"d122d9ced096e51609cd202cb97ff6f2614e0be5394f3b38f84c8c58df04e07c",
    "shapes.tsv"->"56d586abab1789e1e880d0f9b7677be5fb3ee062aee01d1d3382e58d7efebef9",
    "output-3x5-f32.bin"->"945c4f0cb4751cba2f1e681103c4ed7326bc17351e16c9888eeb11869231d318",
    "output-4x4-f32.bin"->"41cb5f022aa87374607ec338a9906369145252f807ef4dd6cfc49b93c56e8270",
    "output-4x6-f32.bin"->"bec95207b1bc79df2337864b8b6e3b2bc779449cf6706407c2bf563d193a94a2")
  private def verifyDigest(stream: java.io.InputStream,expected: String): Unit =
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val bytes=new Array[Byte](65536)
    var count=stream.read(bytes)
    while count != -1 do
      require(count>0,"hash input made no progress")
      digest.update(bytes,0,count)
      count=stream.read(bytes)
    val actual=java.util.HexFormat.of().formatHex(digest.digest())
    require(actual==expected,s"pinned U-Net reference SHA256 mismatch: expected $expected, got $actual")
  private def pinned(dir: Path): Unit =
    for name <- Vector("config.json","diffusion_pytorch_model.safetensors") do
      val stream=Files.newInputStream(dir.resolve(name))
      try verifyDigest(stream,hashes(name)) finally stream.close()
  private def resource(name: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(s"/tiny-sd/$name")
    val bytes=try stream.readAllBytes() finally stream.close()
    hashes.get(name).foreach(expected => verifyDigest(new java.io.ByteArrayInputStream(bytes),expected))
    bytes
  private val shapes=new String(resource("shapes.tsv"),UTF_8).linesIterator.map { line =>
    val parts=line.split("\t")
    parts(0)->parts(1).split(",").map(_.toInt).toVector
  }.toVector
  private def fixture(entries: Vector[(String,Vector[Int])]=shapes,dtype: String="F32")(f: Path => Unit): Unit =
    val dir=Files.createTempDirectory("scala-torch-unet")
    try
      Files.write(dir.resolve("config.json"),resource("config.json"))
      val elementBytes=if dtype=="BOOL" then 1 else if dtype=="F32" then 4 else 2
      var offset=0
      val header=entries.map { (name,dims) =>
        val begin=offset;offset+=dims.product*elementBytes
        s"\"$name\":{\"dtype\":\"$dtype\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
      }.mkString("{",",","}").getBytes(UTF_8)
      val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
      entries.foreach { (name,dims) =>
        for i <- 0 until dims.product do
          val v=if name=="conv_out.bias" then Vector(.25f,.5f,-.75f,1f)(i) else 0f
          dtype match
            case "F32" => data.putFloat(v)
            case "F16" => data.putShort(java.lang.Float.floatToFloat16(v))
            case "BF16" => data.putShort((java.lang.Float.floatToRawIntBits(v) >>> 16).toShort)
            case "BOOL" => data.put(if v==0 then 0.toByte else 1.toByte)
            case _ => throw IllegalArgumentException("unknown fixture dtype")
      }
      Files.write(dir.resolve("diffusion_pytorch_model.safetensors"),
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).array()++header++data.array())
      f(dir)
    finally
      Files.deleteIfExists(dir.resolve("config.json"))
      Files.deleteIfExists(dir.resolve("diffusion_pytorch_model.safetensors"))
      Files.deleteIfExists(dir)

  test("archive loads in training mode and maps output bias without public weights") {
    fixture() { dir =>
      val m=PretrainedUNet.load(dir,trainable=true)
      assertEquals(Module.parameters(m).size,304)
      assert(Module.parameters(m).forall(_.trainable))
      val x=Variable(Tensor.fromFloats(Array.fill(4*2*2)(.3f),Shape.from(Seq(1,4,2,2))),false)
      val c=Variable(Tensor.fromFloats(Array.fill(32)(.2f),Shape.from(Seq(1,1,32))),false)
      val y=scalatorch.autograd.Autograd.noGrad(m(x,Tensor.fromFloats(Array(0f),Shape.from(Seq(1))),c))
      assertEquals(y.data.toFloatArray.toVector,Vector(.25f,.5f,-.75f,1f).flatMap(v => Vector.fill(4)(v)))
    }
  }
  test("reject missing, extra, misshaped and unsupported-dtype checkpoint tensors") {
    fixture(shapes.filterNot(_._1=="conv_out.bias")) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedUNet.load(dir)).getMessage.contains("missing U-Net weight"))
    }
    fixture(shapes :+ ("unexpected"->Vector(1))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedUNet.load(dir)).getMessage.contains("unused U-Net weights"))
    }
    fixture(shapes.map((name,dims) => name->(if name=="conv_in.weight" then dims.reverse else dims))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedUNet.load(dir)).getMessage.contains("shape mismatch"))
    }
    fixture(dtype="BOOL") { dir =>
      assert(intercept[IllegalArgumentException](PretrainedUNet.load(dir)).getMessage.contains("unsupported by this typed read"))
    }
    fixture() { dir => intercept[IllegalArgumentException](PretrainedUNet.load(dir,maxParameters=1)) }
  }
  test("public U-Net matches independent Candle 0.9.1 on three spatial/time cases") {
    val dir=sys.props.get("scalatorch.unet.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.unet.dir to pinned tiny-sd-pipe/unet")
    pinned(dir.get)
    val m=PretrainedUNet.load(dir.get)
    for (h,w,t) <- Vector((3,5,7.5f),(4,4,0f),(4,6,19f)) do
      val x=Variable(Tensor.fromFloats(Array.tabulate(4*h*w)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,h,w))),false)
      val c=Variable(Tensor.fromFloats(Array.tabulate(3*32)(i => math.cos(i*.13).toFloat),Shape.from(Seq(1,3,32))),false)
      val actual=m(x,Tensor.fromFloats(Array(t),Shape.from(Seq(1))),c).data.toFloatArray
      val bytes=ByteBuffer.wrap(resource(s"output-${h}x${w}-f32.bin")).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(bytes.remaining()/4)(bytes.getFloat())
      assertEquals(actual.length,expected.length)
      val error=actual.zip(expected).map((a,b) => math.abs(a.toDouble-b)).max
      println(s"U-Net ${h}x${w} time=$t maxAbsError=$error")
      assert(error<2e-5,s"U-Net reference discrepancy: $error")
  }

  test("half and bfloat16 archives materialize frozen Float32 parameters") {
    for dtype <- Vector("F16","BF16") do
      fixture(dtype=dtype) { dir =>
        val m=PretrainedUNet.load(dir)
        assertEquals(Module.parameters(m).size,304)
        assert(Module.parameters(m).forall(!_.trainable))
        assertEquals(m.convOut.bias.get.data.toFloatArray.toVector,Vector(.25f,.5f,-.75f,1f))
      }
  }

  test("reference gate rejects modified input or output bytes") {
    val bytes=resource("config.json")
    bytes(0)=(bytes(0)^1).toByte
    intercept[IllegalArgumentException](verifyDigest(new java.io.ByteArrayInputStream(bytes),hashes("config.json")))
  }
