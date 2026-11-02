package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.Module
import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8

class PretrainedVaeSuite extends munit.FunSuite:
  private val hashes=Map(
    "encode-b1-6x10-f32.bin"->"770c541a441425522c470bac7a94969921be6d57335a4a6ca8db30b43ce9d1b8",
    "config.json"->"e5d2dc3a363761f93349b5751db4526015420d4b76c8cbde4d0201d4bfcb04c1",
    "shapes.tsv"->"aafe71e4bd7d788cd538dbdec8dbb9ed96cf4c0beb30ef69a8aa59bdc1a88583",
    "decode-b1-4x4-f32.bin"->"bacea083280f2ea735141a76737e5f8aedefa397c29434fcb4e7afc1207807ab",
    "decode-b2-3x5-f32.bin"->"0293e10eb1d7588eab9761b060959f1c3c7bf95b4e0b38e4672ec17c9fee70ba",
    "encode-b2-8x8-f32.bin"->"4658a57ffa1a35164b4c24e16758dfa3de3ebdfd4e64ba317e190445d0cf92cf",
    "encode-b1-7x11-f32.bin"->"d64d0831ef77088b9c7b4290b2f5bf1729239e48fdf92807571b5a7f6823da2c",
    "decode-b1-3x5-f32.bin"->"c953abb2f631631be06f87bd6b10b31a2f701e5129be30ba95796ec135b6924e",
    "diffusion_pytorch_model.safetensors"->"7999b516094ac4239121d9f62a37e6be99620e7016220011bb99cf8b1a391aad")
  private def verify(stream: java.io.InputStream,expected: String): Unit =
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val bytes=new Array[Byte](65536)
    var count=stream.read(bytes)
    while count != -1 do
      require(count>0,"hash stream made no progress")
      digest.update(bytes,0,count);count=stream.read(bytes)
    require(java.util.HexFormat.of().formatHex(digest.digest())==expected,"pinned VAE SHA256 mismatch")
  private def pinned(dir: Path): Unit =
    for name <- Vector("config.json","diffusion_pytorch_model.safetensors") do
      val stream=Files.newInputStream(dir.resolve(name))
      try verify(stream,hashes(name)) finally stream.close()
  private def resource(name: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(s"/tiny-vae/$name")
    val bytes=try stream.readAllBytes() finally stream.close()
    hashes.get(name).foreach(hash => verify(new java.io.ByteArrayInputStream(bytes),hash))
    bytes
  private val json=new String(resource("config.json"),java.nio.charset.StandardCharsets.UTF_8)
  test("VAE config preserves pipeline scale and rejects unsupported or malformed semantics") {
    val c=VaeConfig.fromJson(json)
    assertEquals(c.channels,Vector(32,64));assertEquals(c.latentChannels,4)
    assertEqualsDouble(c.scalingFactor,.18215,0)
    assert(c.quantConv && c.postQuantConv && c.midAttention)
    for bad <- Vector(json.replace("\"act_fn\": \"silu\"","\"act_fn\": \"relu\""),
      json.replace("\"layers_per_block\": 1","\"layers_per_block\": 0"),
      json.replace("\"norm_num_groups\": 32","\"norm_num_groups\": 3"),
      json.replace("\"scaling_factor\": 0.18215","\"scaling_factor\": 0"),
      json.replace("DownEncoderBlock2D","CrossAttnDownBlock2D"),
      json.dropRight(2)+",\"latent_channels\":4}",json.dropRight(2)+",\"shift_factor\":1}",
      json.dropRight(2)+",\"unknown_option\":false}",json+"{}") do
      intercept[IllegalArgumentException](VaeConfig.fromJson(bad))
  }
  test("public VAE loads all124 parameters and performs frozen raw encode/decode") {
    val dir=sys.props.get("scalatorch.vae.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.vae.dir to pinned tiny-sd-pipe/vae")
    pinned(dir.get)
    val model=PretrainedVae.load(dir.get)
    assertEquals(Module.parameters(model).size,124)
    assert(Module.parameters(model).forall(!_.trainable))
    val image=Variable(Tensor.fromFloats(Array.tabulate(3*6*10)(i => math.sin(i*.13).toFloat),Shape.from(Seq(1,3,6,10))),false)
    val moments=model.encodeMoments(image)
    assertEquals(moments.data.shape,Shape.from(Seq(1,8,3,5)))
    val decoded=model.decode(moments.narrow(1,0,4))
    assertEquals(decoded.data.shape,image.data.shape)
    assert(!moments.requiresGrad && !decoded.requiresGrad)
    assert(moments.data.toFloatArray.forall(_.isFinite) && decoded.data.toFloatArray.forall(_.isFinite))
    intercept[IllegalArgumentException](PretrainedVae.load(dir.get,maxParameters=1))
  }

  test("public encoder moments and decoder images match six independent Candle cases") {
    val dir=sys.props.get("scalatorch.vae.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.vae.dir to pinned tiny-sd-pipe/vae")
    pinned(dir.get)
    val model=PretrainedVae.load(dir.get)
    def compare(name: String,actual: Tensor[Float]): Unit =
      val bytes=ByteBuffer.wrap(resource(s"$name-f32.bin")).order(ByteOrder.LITTLE_ENDIAN)
      val expected=Array.fill(bytes.remaining()/4)(bytes.getFloat())
      val values=actual.toFloatArray
      assertEquals(values.length,expected.length)
      val error=values.zip(expected).map((a,b) => math.abs(a.toDouble-b)).max
      println(s"VAE $name maxAbsError=$error")
      assert(error<2e-5,s"VAE reference discrepancy: $error")
    for (b,h,w) <- Vector((1,6,10),(1,7,11),(2,8,8)) do
      val x=Variable(Tensor.fromFloats(Array.tabulate(b*3*h*w)(i => math.sin(i*.13).toFloat),Shape.from(Seq(b,3,h,w))),false)
      val output=model.encodeMoments(x).data
      assertEquals(output.shape,Shape.from(Seq(b,8,h/2,w/2)))
      compare(s"encode-b$b-${h}x$w",output)
    for (b,h,w) <- Vector((1,3,5),(1,4,4),(2,3,5)) do
      val x=Variable(Tensor.fromFloats(Array.tabulate(b*4*h*w)(i => math.cos(i*.17).toFloat),Shape.from(Seq(b,4,h,w))),false)
      val output=model.decode(x).data
      assertEquals(output.shape,Shape.from(Seq(b,3,h*2,w*2)))
      compare(s"decode-b$b-${h}x$w",output)
      val differentScale=model.copy(config=model.config.copy(scalingFactor=2))
      assertEquals(differentScale.decode(x).data.toFloatArray.toVector,output.toFloatArray.toVector)
  }
  test("pinned reference guard rejects altered data") {
    val bytes=resource("decode-b1-3x5-f32.bin");bytes(0)=(bytes(0)^1).toByte
    intercept[IllegalArgumentException](verify(new java.io.ByteArrayInputStream(bytes),hashes("decode-b1-3x5-f32.bin")))
  }
  test("public VAE batching equals independent image rows") {
    val dir=sys.props.get("scalatorch.vae.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.vae.dir to pinned tiny-sd-pipe/vae")
    pinned(dir.get)
    val model=PretrainedVae.load(dir.get)
    val data=Tensor.fromFloats(Array.tabulate(2*3*8*8)(i => math.sin(i*.13).toFloat),Shape.from(Seq(2,3,8,8)))
    val batched=model.encodeMoments(Variable(data,false)).data.toFloatArray
    val rows=(0 until 2).flatMap(i => model.encodeMoments(Variable(data.slice(0,i,i+1),false)).data.toFloatArray).toArray
    val error=batched.zip(rows).map((a,b) => math.abs(a.toDouble-b)).max
    println(s"Scala VAE encoder batch-vs-rows maxabs=$error first=${batched.take(4).mkString(",")}")
    assert(error<1e-6)
    val latent=Tensor.fromFloats(Array.tabulate(2*4*3*5)(i => math.cos(i*.17).toFloat),Shape.from(Seq(2,4,3,5)))
    val decoded=model.decode(Variable(latent,false)).data.toFloatArray
    val decodedRows=(0 until 2).flatMap(i => model.decode(Variable(latent.slice(0,i,i+1),false)).data.toFloatArray).toArray
    val decodeError=decoded.zip(decodedRows).map((a,b) => math.abs(a.toDouble-b)).max
    assert(decodeError<1e-6,s"decoder batching discrepancy: $decodeError")
  }

  private val shapes=new String(resource("shapes.tsv"),UTF_8).linesIterator.map { line =>
    val parts=line.split("\t")
    parts(0)->parts(1).split(",").map(_.toInt).toVector
  }.toVector
  private def fixture(entries: Vector[(String,Vector[Int])]=shapes,dtype: String="F32",config: String=json)(f: Path => Unit): Unit =
    val dir=Files.createTempDirectory("scala-torch-vae")
    try
      Files.writeString(dir.resolve("config.json"),config)
      val elementBytes=if dtype=="BOOL" then 1 else if dtype=="F32" then 4 else 2
      var offset=0
      val header=entries.map { (name,dims) =>
        val begin=offset;offset+=dims.product*elementBytes
        s"\"$name\":{\"dtype\":\"$dtype\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
      }.mkString("{",",","}").getBytes(UTF_8)
      val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
      entries.foreach { (name,dims) =>
        for i <- 0 until dims.product do
          val v=if name=="decoder.conv_out.bias" then Vector(.25f,.5f,-.75f)(i) else 0f
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


  private def checkConstant(model: VaeModel): Unit =
    val latent=Variable(Tensor.fromFloats(Array.fill(4*2*3)(.3f),Shape.from(Seq(1,4,2,3))),false)
    val output=scalatorch.autograd.Autograd.noGrad(model.decode(latent))
    assertEquals(output.data.shape,Shape.from(Seq(1,3,4,6)))
    assertEquals(output.data.toFloatArray.toVector,Vector(.25f,.5f,-.75f).flatMap(v => Vector.fill(24)(v)))
    val image=Variable(Tensor.fromFloats(Array.fill(3*4*6)(.2f),Shape.from(Seq(1,3,4,6))),false)
    val moments=scalatorch.autograd.Autograd.noGrad(model.encodeMoments(image))
    assertEquals(moments.data.toFloatArray.toVector,Vector.fill(8*2*3)(0f))

  test("synthetic complete archive maps weights and preserves trainability") {
    fixture() { dir =>
      val model=PretrainedVae.load(dir,trainable=true)
      assertEquals(Module.parameters(model).size,124)
      assert(Module.parameters(model).forall(_.trainable))
      checkConstant(model)
    }
  }
  test("missing, extra, misshaped, unsupported dtype and over-budget VAE weights reject") {
    fixture(shapes.filterNot(_._1=="decoder.conv_out.bias")) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedVae.load(dir)).getMessage.contains("missing VAE weight"))
    }
    fixture(shapes :+ ("unexpected"->Vector(1))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedVae.load(dir)).getMessage.contains("unused VAE weights"))
    }
    fixture(shapes.map((name,dims) => name->(if name=="encoder.conv_in.weight" then dims.reverse else dims))) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedVae.load(dir)).getMessage.contains("shape mismatch"))
    }
    fixture(dtype="BOOL") { dir =>
      assert(intercept[IllegalArgumentException](PretrainedVae.load(dir)).getMessage.contains("unsupported by this typed read"))
    }
    fixture() { dir => intercept[IllegalArgumentException](PretrainedVae.load(dir,maxParameters=1)) }
  }
  test("half and bfloat16 VAE archives convert to frozen Float32") {
    for dtype <- Vector("F16","BF16") do fixture(dtype=dtype) { dir =>
      val model=PretrainedVae.load(dir)
      assert(Module.parameters(model).forall(!_.trainable))
      checkConstant(model)
    }
  }
  test("legacy attention names and singleton convolution projections load explicitly") {
    val legacy=shapes.map { (name,dims) =>
      val renamed=name.replace(".to_q.",".query.").replace(".to_k.",".key.")
        .replace(".to_v.",".value.").replace(".to_out.0.",".proj_attn.")
      renamed->(if name.contains(".attentions.") && dims.size==2 then dims++Vector(1,1) else dims)
    }
    fixture(legacy) { dir =>
      val model=PretrainedVae.load(dir)
      assertEquals(Module.parameters(model).size,124)
      checkConstant(model)
    }
  }
  test("optional quant convolutions and mid attention follow config and reject leftover weights") {
    val config=json.trim.dropRight(1)+",\"use_quant_conv\":false,\"use_post_quant_conv\":false,\"mid_block_add_attention\":false}"
    val entries=shapes.filterNot { (name,_) =>
      name.startsWith("quant_conv.") || name.startsWith("post_quant_conv.") || name.contains(".attentions.")
    }
    fixture(entries,config=config) { dir =>
      val model=PretrainedVae.load(dir)
      assertEquals(Module.parameters(model).size,entries.size)
      assert(model.encoder.quantConv.isEmpty && model.decoder.postQuantConv.isEmpty)
      assert(model.encoder.midBlock.attention.isEmpty && model.decoder.midBlock.attention.isEmpty)
      checkConstant(model)
    }
    fixture(config=config) { dir =>
      assert(intercept[IllegalArgumentException](PretrainedVae.load(dir)).getMessage.contains("unused VAE weights"))
    }
  }

  test("posterior API and diffusion scaling preserve gradients and explicit raw semantics") {
    fixture() { dir =>
      val model=PretrainedVae.load(dir)
      val image=Variable(Tensor.fromFloats(Array.fill(3*4*6)(.1f),Shape.from(Seq(1,3,4,6))),false)
      val posterior=model.encode(image)
      assertEquals(posterior.mean.data.shape,Shape.from(Seq(1,4,2,3)))
      assertEquals(posterior.kl().data.toFloatArray.toVector,Vector(0f))
      val latent=Variable(Tensor.fromFloats(Array.fill(24)(1f),posterior.mean.data.shape))
      val scaled=model.toDiffusionLatents(latent)
      assertEquals(scaled.data.toFloatArray.toVector,Vector.fill(24)(model.config.scalingFactor.toFloat))
      scaled.sum().backward()
      assertEquals(latent.grad.get.toFloatArray.toVector,Vector.fill(24)(model.config.scalingFactor.toFloat))
      val imageOut=model.decodeDiffusionLatents(scaled)
      assertEquals(imageOut.data.toFloatArray.toVector,Vector(.25f,.5f,-.75f).flatMap(v => Vector.fill(24)(v)))
      val wrong=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1))))
      intercept[IllegalArgumentException](model.toDiffusionLatents(wrong))
      intercept[IllegalArgumentException](model.decodeDiffusionLatents(wrong))
    }
  }
  test("public diffusion decode divides by configured scale exactly once") {
    val dir=sys.props.get("scalatorch.vae.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.vae.dir to pinned tiny-sd-pipe/vae")
    pinned(dir.get)
    val original=PretrainedVae.load(dir.get)
    val model=original.copy(config=original.config.copy(scalingFactor=2))
    val latent=Variable(Tensor.fromFloats(Array.tabulate(4*3*5)(i => math.cos(i*.17).toFloat),Shape.from(Seq(1,4,3,5))),false)
    val actual=model.decodeDiffusionLatents(model.toDiffusionLatents(latent))
    assertEquals(actual.data.toFloatArray.toVector,model.decode(latent).data.toFloatArray.toVector)
    assert(!actual.requiresGrad)
  }

  test("published CLIP U-Net Euler VAE loop yields reproducible conditioning-sensitive images") {
    val dirs=Vector("clip","unet","vae").map(key => sys.props.get(s"scalatorch.$key.dir").map(Path.of(_)))
    assume(dirs.forall(_.nonEmpty),"set pinned CLIP U-Net and VAE directories")
    val Vector(clipDir,unetDir,vaeDir)=dirs.map(_.get): @unchecked
    pinned(vaeDir)
    for (dir,name,hash) <- Vector(
      (clipDir,"model.safetensors","57f8ef72a6c2d5abd52829b83ed38d217cce0d515021db79147baf60dd94175f"),
      (clipDir,"config.json","aae70cf2c528c154e5cbd2eba1249e759b91bbce15a9be74698f4a6492ec4879"),
      (unetDir,"diffusion_pytorch_model.safetensors","d122d9ced096e51609cd202cb97ff6f2614e0be5394f3b38f84c8c58df04e07c"),
      (unetDir,"config.json","2f80e0a35261d11d52c976b8bf15ec45ab414f46c5506e7a1f0c48851382ef06")) do
      val stream=Files.newInputStream(dir.resolve(name))
      try verify(stream,hash) finally stream.close()
    val clip=PretrainedClip.load(clipDir);val unet=PretrainedUNet.load(unetDir);val vae=PretrainedVae.load(vaeDir)
    val schedule=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true).euler(4)
    val scheduler=EulerScheduler(schedule)
    val initial=Variable(Tensor.fromFloats(Array.tabulate(60)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,3,5))),false)
    def run(token: Long): Vector[Float] =
      val context=clip(Tensor.fromLongs(Array(0L,token,2L),Shape.from(Seq(1,3))))
      var latent=initial*schedule.initialNoiseSigma.toFloat
      for i <- schedule.timesteps.indices do
        val time=Tensor.fromFloats(Array(schedule.timesteps(i)),Shape.from(Seq(1)))
        val predicted=unet(scheduler.scaleModelInput(i,latent),time,context)
        latent=scheduler.step(i,predicted,latent).previous
        assert(!latent.requiresGrad)
      val image=vae.decodeDiffusionLatents(latent)
      assertEquals(image.data.shape,Shape.from(Seq(1,3,6,10)))
      assert(!image.requiresGrad && image.data.toFloatArray.forall(_.isFinite))
      image.data.toFloatArray.toVector
    val first=run(7L)
    assertEquals(first,run(7L))
    val other=run(31L)
    assert(first.zip(other).exists((a,b) => math.abs(a-b)>1e-5))
  }
