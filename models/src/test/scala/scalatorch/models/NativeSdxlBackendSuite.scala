package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}
import scalatorch.models.*
import scalatorch.nn.LinearAdapterScope
import NativeJson.*

class NativeSdxlBackendSuite extends munit.FunSuite:
  private def writeAdapter(path: Path,name: String,input: Int,output: Int): Unit =
    val entries=Vector((s"$name.lora_A.weight",Vector(1,input),Array.fill(input)(1f/input)),
      (s"$name.lora_B.weight",Vector(output,1),Array.tabulate(output)(i=>if i%2==0 then 1000f else -1000f)))
    var offset=0
    val header=entries.map { (key,dims,values) =>
      val begin=offset;offset+=values.length*4
      s"\"$key\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  test("NUI SDXL original size and crop metadata is strict and becomes six time IDs") {
    val fields=Map("sdfx_sdxl_original_height"->Str("80"),"sdfx_sdxl_original_width"->Str("96"),
      "sdfx_sdxl_crop_top"->Str("2"),"sdfx_sdxl_crop_left"->Str("4"))
    val size=NativeGenerationMetadata.sdxlSize(Some(Obj(fields)),true).get
    assertEquals(size.timeIds(64,72).toFloatArray.toVector,Vector(80f,96f,2f,4f,64f,72f))
    assertEquals(NativeGenerationMetadata.sdxlSize(None,true),None)
    intercept[NativeBackendError](NativeGenerationMetadata.sdxlSize(Some(Obj(fields)),false))
    intercept[IllegalArgumentException](NativeGenerationMetadata.sdxlSize(Some(Obj(fields-"sdfx_sdxl_original_width")),true))
    intercept[NumberFormatException](NativeGenerationMetadata.sdxlSize(Some(Obj(fields.updated("sdfx_sdxl_original_height",Str("oops")))),true))
    intercept[IllegalArgumentException](NativeGenerationMetadata.sdxlSize(Some(Obj(fields.updated("sdfx_sdxl_crop_top",Str("-1")))),true))
  }

  test("SDXL worker loads dual encoders, preserves them across VAE replacement and failed load") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL required")
    val original=Path.of(directory.get)
    val root=Files.createTempDirectory("native-sdxl")
    try
      for part <- Seq("text_encoder","text_encoder_2","unet","vae","tokenizer","tokenizer_2") do
        Files.createDirectory(root.resolve(part))
        val names=if part.startsWith("tokenizer") then Seq("vocab.json","merges.txt","tokenizer_config.json")
          else Seq("config.json",if part.startsWith("text") then "model.safetensors" else "diffusion_pytorch_model.safetensors")
        names.foreach(n => Files.copy(original.resolve(s"$part/$n"),root.resolve(s"$part/$n")))
      Files.copy(original.resolve("model_index.json"),root.resolve("model_index.json"))
      val scheduler=root.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
      val backend=new NativeDiffusionBackend(root,scheduler)
      def run(kind: String,params: Map[String,NativeJson]=Map.empty)=backend.execute(NativeRequest("r",kind,params,None),context)
      val metadata=run("load_model")
      assertEquals(metadata("family"),Str("sdxl"))
      val params=Map("output_dir"->Str(root.toString),"params"->Obj(Map("prompt"->Str("apple"),"negative_prompt"->Str("blur"),
        "seed"->Number(42),"width"->Number(8),"height"->Number(8),"steps"->Number(2),"sampler_name"->Str("DPM++ 2M"))))
      val fields=params("params").asInstanceOf[Obj].fields
      def imagePath(result: Map[String,NativeJson]): String=
        result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      def pixels(result: Map[String,NativeJson]): Vector[Int] =
        val path=imagePath(result)
        val image=javax.imageio.ImageIO.read(Path.of(path).toFile)
        assertEquals(image.getWidth,8)
        image.getRGB(0,0,8,8,null,0,8).toVector
      val expected=pixels(run("txt2img",params))
      val encoder2Loaded=PretrainedClip.loadWithLinearTargets(root.resolve("text_encoder_2"),aliasPrefix="text_encoder_2")
      val encoder2Targets=encoder2Loaded.linearTargets
      val target=encoder2Targets("text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj")
      val adapter=root.resolve("sdxl-encoder2.safetensors")
      writeAdapter(adapter,"text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj",
        target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      val tokenizer2=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2"))
      val tokenIds=tokenizer2.encode("apple")
      val tokenTensor=scalatorch.Tensor.fromLongs(tokenIds,scalatorch.Shape.from(Seq(1L,tokenIds.length.toLong)))
      val plainEncoder2=encoder2Loaded.model(tokenTensor).data.toFloatArray.toVector
      val encoder2Adapters=DiffusionLoRA.load(adapter,encoder2Targets)
      val adaptedEncoder2=LinearAdapterScope.withAdapters(encoder2Adapters) {
        encoder2Loaded.model(tokenTensor).data.toFloatArray.toVector
      }
      assert(adaptedEncoder2!=plainEncoder2,"second SDXL text encoder aliases must target live linear layers")
      val withAdapter=params.updated("params",Obj(fields.updated("loras",Arr(Vector(Obj(Map(
        "name"->Str("encoder2"),"weight"->Number(1),"unet"->Null)))))))
        .updated("lora_paths",Obj(Map("encoder2"->Str(adapter.toString))))
      val zeroAdapter=withAdapter.updated("params",Obj(fields.updated("loras",Arr(Vector(Obj(Map(
        "name"->Str("encoder2"),"weight"->Number(0),"unet"->Null)))))))
      assertEquals(pixels(run("txt2img",zeroAdapter)),expected)
      val adapted=pixels(run("txt2img",withAdapter))
      assertEquals(pixels(run("txt2img",withAdapter)),adapted,"SDXL adapter replay must be deterministic")
      val source=imagePath(run("txt2img",params))
      val imgParams=params.updated("params",Obj(fields.updated("img2img",Obj(Map("init_image"->Str(source),
        "denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0))))))
      val imgExpected=pixels(run("img2img",imgParams))
      assertEquals(pixels(run("img2img",imgParams)),imgExpected)
      val badPrompt=params.updated("params",Obj(fields.updated("prompt",Str("<lora:test:1> apple"))))
      assertEquals(intercept[NativeBackendError](run("txt2img",badPrompt)).code,"unsupported")
      def fileCount(): Long =
        val files=Files.list(root)
        try files.count() finally files.close()
      val before=fileCount()
      var cancelled=false
      val cancellation=new NativeWorkContext(() => cancelled,() => !cancelled,(_,_) => cancelled=true)
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("cancel","txt2img",params,None),cancellation))
      assertEquals(fileCount(),before)
      assertEquals(pixels(run("txt2img",params)),expected)
      run("set_vae",Map("path"->Null))
      assertEquals(pixels(run("txt2img",params)),expected)
      val retained=backend.systemStatus("model")
      val encoderConfig=root.resolve("text_encoder_2/config.json")
      val saved=Files.readString(encoderConfig)
      Files.writeString(encoderConfig,"{}")
      intercept[IllegalArgumentException](run("load_model"))
      assertEquals(backend.systemStatus("model"),retained)
      assertEquals(pixels(run("txt2img",params)),expected)
      Files.writeString(encoderConfig,saved)
      val denied=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("denied","load_model",Map.empty,None),denied))
      assertEquals(backend.systemStatus("model"),retained)
      val padConfig=root.resolve("tokenizer_2/tokenizer_config.json")
      Files.writeString(padConfig,"""{"pad_token":"!","model_max_length":77}""")
      val changed=run("load_model")
      assert(changed("hash")!=metadata("hash"))
      pixels(run("txt2img",params))
      Files.writeString(padConfig,"""{"pad_token":"missing-token","model_max_length":77}""")
      intercept[IllegalArgumentException](run("load_model"))
      assertEquals(backend.systemStatus("model").asInstanceOf[Obj].fields("hash"),changed("hash"))
      Files.writeString(padConfig,"""{"pad_token":{"content":"!"},"model_max_length":77}""")
      val index=root.resolve("model_index.json")
      Files.writeString(index,"""{"_class_name":"StableDiffusionXLInpaintPipeline"}""")
      intercept[IllegalArgumentException](run("load_model"))
      assertEquals(backend.systemStatus("model").asInstanceOf[Obj].fields("hash"),changed("hash"))
      Files.writeString(index,Files.readString(original.resolve("model_index.json")))
      if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
        val gpu=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(maxBytes=16777216,offloadWeights=true),
          unetExecution=UNetExecution.Cuda(maxBytes=16777216,offloadWeights=true))
        gpu.execute(NativeRequest("load","load_model",Map.empty,None),context)
        pixels(gpu.execute(NativeRequest("txt","txt2img",params,None),context))
        pixels(gpu.execute(NativeRequest("img","img2img",imgParams,None),context))
        val modelBefore=gpu.systemStatus("model")
        intercept[Exception](gpu.execute(NativeRequest("bad","set_vae",Map("path"->Str(root.resolve("absent").toString)),None),context))
        assertEquals(gpu.systemStatus("model"),modelBefore)
    finally
      val paths=Files.walk(root)
      try
        import scala.jdk.CollectionConverters.*
        paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally paths.close()
  }
