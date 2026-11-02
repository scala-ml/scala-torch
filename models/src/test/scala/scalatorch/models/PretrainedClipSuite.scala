package scalatorch.models

import scalatorch.*
import scalatorch.nn.Module
import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import scalatorch.autograd.Variable

class PretrainedClipSuite extends munit.FunSuite:
  private val hashes=Map(
    "gelu-skip1-f32.bin"->"37eebdd92d23db4c60f03c94d5761a5edd7515ad8d78c985a01b932411a340c3",
    "tanh-b1-f32.bin"->"a440bff4dff75d019f2e2b7f4e22a886d2faffb7fc0a2ee2d06afc9650648d70",
    "config.json"->"aae70cf2c528c154e5cbd2eba1249e759b91bbce15a9be74698f4a6492ec4879",
    "shapes.tsv"->"73700db8309d42953c1c8b08b07db5ec4463d0885502d2701c54598750c759f6",
    "gelu-b2-f32.bin"->"49d0501a5a8e3cb92ece701466df31ddad891de5b0b96fdeee8d0e3d265a6ef0",
    "gelu-b1-f32.bin"->"4e8e48472fd91f41052539720af67b6f65ed3a692cb21a6e5b9c83d9252fdf3f",
    "quick-b1-f32.bin"->"929f0c363453b7c37652f14344716dd2ed254c0eb58ff464fbd748b20a4c26b0",
    "model.safetensors"->"57f8ef72a6c2d5abd52829b83ed38d217cce0d515021db79147baf60dd94175f")
  private def verify(stream: java.io.InputStream,expected: String): Unit =
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val bytes=new Array[Byte](65536)
    var count=stream.read(bytes)
    while count != -1 do
      require(count>0,"hash stream made no progress")
      digest.update(bytes,0,count);count=stream.read(bytes)
    require(java.util.HexFormat.of().formatHex(digest.digest())==expected,"pinned CLIP SHA256 mismatch")
  private def pinned(dir: Path): Unit =
    for name <- Vector("config.json","model.safetensors") do
      val stream=Files.newInputStream(dir.resolve(name))
      try verify(stream,hashes(name)) finally stream.close()
  private def resource(name: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(s"/tiny-clip/$name")
    val bytes=try stream.readAllBytes() finally stream.close()
    hashes.get(name).foreach(hash => verify(new java.io.ByteArrayInputStream(bytes),hash))
    bytes
  private val configJson=new String(resource("config.json"),java.nio.charset.StandardCharsets.UTF_8)
  test("strict CLIP config reads public architecture and rejects unsupported settings") {
    val c=ClipTextConfig.fromJson(configJson)
    assertEquals(c.width,32);assertEquals(c.layers,5);assertEquals(c.activation,ClipActivation.Gelu)
    for json <- Vector(configJson.replace("\"attention_dropout\": 0.0","\"attention_dropout\": 0.1"),
      configJson.replace("\"hidden_size\": 32","\"hidden_size\": 33"),
      configJson.replace("\"num_hidden_layers\": 5","\"num_hidden_layers\": 0"),
      configJson.replace("\"hidden_act\": \"gelu\"","\"hidden_act\": \"relu\""),
      configJson.replace("\"layer_norm_eps\": 1e-05","\"layer_norm_eps\": 0"),
      configJson.replace("\"model_type\": \"clip_text_model\"","\"model_type\": \"clip\""),
      configJson.dropRight(2)+",\"unknown_semantics\":true}",
      configJson.dropRight(2)+",\"hidden_size\":32}",configJson+"{}") do
      intercept[IllegalArgumentException](ClipTextConfig.fromJson(json))
  }
  test("public CLIP loads all84 parameters and returns frozen conditioning") {
    val dir=sys.props.get("scalatorch.clip.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.clip.dir to pinned tiny-sd-pipe/text_encoder")
    pinned(dir.get)
    val model=PretrainedClip.load(dir.get)
    assertEquals(Module.parameters(model).size,84)
    assert(Module.parameters(model).forall(!_.trainable))
    val ids=Tensor.fromLongs(Array.tabulate(77)(i => (i%23).toLong),Shape.from(Seq(1,77)))
    val output=model(ids)
    assertEquals(output.data.shape,Shape.from(Seq(1,77,32)))
    assert(!output.requiresGrad)
    assert(output.data.toFloatArray.forall(_.isFinite))
    intercept[IllegalArgumentException](PretrainedClip.load(dir.get,maxParameters=1))
  }

  test("public conditioning matches Candle for batches, earlier layers and activation variants") {
    val dir=sys.props.get("scalatorch.clip.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.clip.dir to pinned tiny-sd-pipe/text_encoder")
    pinned(dir.get)
    val original=PretrainedClip.load(dir.get)
    for (name,batch,skip,activation) <- Vector(("gelu-b1",1,0,ClipActivation.Gelu),
      ("gelu-b2",2,0,ClipActivation.Gelu),("gelu-skip1",1,1,ClipActivation.Gelu),
      ("quick-b1",1,0,ClipActivation.QuickGelu),("tanh-b1",1,0,ClipActivation.GeluTanh)) do
      val model=original.copy(layers=original.layers.map(_.copy(activation=activation)))
      val ids=Tensor.fromLongs(Array.tabulate(batch*77)(i => ((i*17+3)%997).toLong),Shape.from(Seq(batch,77)))
      val actual=model(ids,skipLastLayers=skip).data.toFloatArray
      val bytes=ByteBuffer.wrap(resource(s"$name-f32.bin")).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(bytes.remaining()/4)(bytes.getFloat())
      assertEquals(actual.length,expected.length)
      val error=actual.zip(expected).map((a,b) => math.abs(a.toDouble-b)).max
      println(s"CLIP $name maxAbsError=$error")
      assert(error<2e-5,s"CLIP reference discrepancy: $error")
  }
  test("published CLIP conditioning drives the published U-Net with text-sensitive output") {
    val clipDir=sys.props.get("scalatorch.clip.dir").map(Path.of(_))
    val unetDir=sys.props.get("scalatorch.unet.dir").map(Path.of(_))
    assume(clipDir.nonEmpty && unetDir.nonEmpty,"set pinned CLIP and U-Net directories")
    pinned(clipDir.get)
    val clip=PretrainedClip.load(clipDir.get);val unet=PretrainedUNet.load(unetDir.get)
    val ids=Tensor.fromLongs(Array.tabulate(77)(i => ((i*17+3)%997).toLong),Shape.from(Seq(1,77)))
    val context=clip(ids)
    val image=Variable(Tensor.fromFloats(Array.tabulate(4*3*5)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,3,5))),false)
    val times=Tensor.fromFloats(Array(7.5f),Shape.from(Seq(1)))
    val actual=unet(image,times,context)
    assert(!actual.requiresGrad)
    assert(actual.data.toFloatArray.forall(_.isFinite))
    val otherIds=Tensor.fromLongs(Array.fill(77)(2L),Shape.from(Seq(1,77)))
    val other=unet(image,times,clip(otherIds))
    assert(actual.data.toFloatArray.zip(other.data.toFloatArray).exists((a,b) => math.abs(a-b)>1e-5))
  }
  test("pinned reference guard rejects altered fixture bytes") {
    val bytes=resource("gelu-b1-f32.bin");bytes(0)=(bytes(0)^1).toByte
    intercept[IllegalArgumentException](verify(new java.io.ByteArrayInputStream(bytes),hashes("gelu-b1-f32.bin")))
  }

  private val shapes=new String(resource("shapes.tsv"),UTF_8).linesIterator.map { line =>
    val parts=line.split("\t")
    parts(0)->parts(1).split(",").map(_.toInt).toVector
  }.toVector
  private def fixture(entries: Vector[(String,Vector[Int])]=shapes,dtype: String="F32",sharded: Boolean=false)(f: Path => Unit): Unit =
    val dir=Files.createTempDirectory("scala-torch-clip")
    try
      Files.write(dir.resolve("config.json"),resource("config.json"))
      def write(name: String,subset: Vector[(String,Vector[Int])]): Unit =
        val elementBytes=if dtype=="BOOL" then 1 else if dtype=="F32" then 4 else 2
        var offset=0
        val header=subset.map { (key,dims) =>
          val begin=offset;offset+=dims.product*elementBytes
          s"\"$key\":{\"dtype\":\"$dtype\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
        }.mkString("{",",","}").getBytes(UTF_8)
        val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
        subset.foreach { (key,dims) =>
          for i <- 0 until dims.product do
            val v=if key=="text_model.final_layer_norm.bias" then 1f else 0f
            dtype match
              case "F32" => data.putFloat(v)
              case "F16" => data.putShort(java.lang.Float.floatToFloat16(v))
              case "BF16" => data.putShort((java.lang.Float.floatToRawIntBits(v) >>> 16).toShort)
              case "BOOL" => data.put(if v==0 then 0.toByte else 1.toByte)
              case _ => throw IllegalArgumentException("unknown fixture dtype")
        }
        Files.write(dir.resolve(name),ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).array()++header++data.array())
      if sharded then
        val chunks=entries.grouped((entries.size+1)/2).toVector.zipWithIndex
        chunks.foreach((subset,i) => write(s"part-$i.safetensors",subset))
        val map=chunks.flatMap((subset,i) => subset.map((key,_) => s"\"$key\":\"part-$i.safetensors\""))
        Files.writeString(dir.resolve("model.safetensors.index.json"),map.mkString("{\"weight_map\":{",",","}}"))
      else write("model.safetensors",entries)
      f(dir)
    finally
      val files=Files.list(dir)
      try files.forEach(p => Files.delete(p)) finally files.close()
      Files.delete(dir)

  test("single and sharded archives preserve dtype conversion, final norm bias and training opt-in") {
    for dtype <- Vector("F32","F16","BF16");sharded <- Vector(false,true) do
      fixture(dtype=dtype,sharded=sharded) { dir =>
        val model=PretrainedClip.load(dir,trainable=true)
        assertEquals(Module.parameters(model).size,84)
        assert(Module.parameters(model).forall(_.trainable))
        val ids=Tensor.fromLongs(Array(1L,2L,3L),Shape.from(Seq(1,3)))
        val out=model(ids)
        assert(out.requiresGrad)
        assert(out.data.toFloatArray.forall(_==1f))
        val frozen=PretrainedClip.load(dir)
        assert(!frozen(ids).requiresGrad)
      }
  }
  test("CLIP archive rejects missing, extra, misshaped, unsupported dtype and over-budget weights") {
    fixture(shapes.filterNot(_._1=="text_model.final_layer_norm.bias")) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedClip.load(dir)).getMessage.contains("missing CLIP weight"))
    }
    fixture(shapes :+ ("unexpected"->Vector(1))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedClip.load(dir)).getMessage.contains("unused CLIP weights"))
    }
    fixture(shapes.map((name,dims) => name->(if name=="text_model.embeddings.token_embedding.weight" then dims.reverse else dims))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedClip.load(dir)).getMessage.contains("shape mismatch"))
    }
    fixture(dtype="BOOL") { dir =>
      assert(intercept[IllegalArgumentException](PretrainedClip.load(dir)).getMessage.contains("unsupported by this typed read"))
    }
    fixture() { dir =>
      intercept[IllegalArgumentException](PretrainedClip.load(dir,maxParameters=1))
      intercept[IllegalArgumentException](PretrainedClip.load(dir,maxScoreElements= -1))
    }
  }
