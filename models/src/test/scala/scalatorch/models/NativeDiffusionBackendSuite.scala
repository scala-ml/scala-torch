package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.nio.{ByteBuffer,ByteOrder}

class NativeDiffusionBackendSuite extends munit.FunSuite:
  private def writeAdapter(path: Path,name: String,input: Int,output: Int): Unit =
    val down=Array.fill(input)(1f/input)
    val up=Array.fill(output)(8f)
    val entries=Vector((s"$name.lora_down.weight",Vector(1,input),down),
      (s"$name.lora_up.weight",Vector(output,1),up))
    var offset=0
    val header=entries.map { (key,dims,values) =>
      val begin=offset;offset+=values.length*4
      s"\"$key\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())

  test("public worker loads indexed CLIP UNet and VAE archives and hashes their source files") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val source=Path.of(configured.get).getParent
    val temp=Files.createTempDirectory("native-indexed")
    import NativeJson.*
    try
      for (part,filename) <- Vector("text_encoder"->"model.safetensors","unet"->"diffusion_pytorch_model.safetensors",
          "vae"->"diffusion_pytorch_model.safetensors") do
        val target=Files.createDirectory(temp.resolve(part))
        Files.copy(source.resolve(part).resolve("config.json"),target.resolve("config.json"))
        val shard="weights-00001-of-00001.safetensors"
        Files.copy(source.resolve(part).resolve(filename),target.resolve(shard))
        val names=scala.util.Using.resource(scalatorch.io.SafeTensors.open(target.resolve(shard)))(_.tensors.keys.toVector)
        Files.writeString(target.resolve(filename+".index.json"),NativeJson.render(Obj(Map(
          "weight_map"->Obj(names.map(_->Str(shard)).toMap)))))
      val tokenizer=Files.createDirectory(temp.resolve("tokenizer"))
      for name <- Vector("vocab.json","merges.txt") do Files.copy(source.resolve("tokenizer").resolve(name),tokenizer.resolve(name))
      val config=temp.resolve("scheduler.json")
      Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(temp,config)
      val context=new NativeWorkContext(() => false,() => true,(_,_) => ())
      val result=backend.execute(NativeRequest("load","load_model",Map("path"->Str(temp.toString)),None),context)
      assert(result("hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      val image=backend.execute(NativeRequest("image","txt2img",Map("output_dir"->Str(temp.toString),
        "params"->Obj(Map("width"->Number(10),"height"->Number(6),"steps"->Number(1)))),None),context)
      assertEquals(image("images").asInstanceOf[Arr].values.size,1)
    finally
      val paths=Files.walk(temp)
      try
        import scala.jdk.CollectionConverters.*
        paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally paths.close()
  }
  test("public native backend loads generates PNG and unloads without Python") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val config=Files.createTempFile("native-euler", ".json")
    val output=Files.createTempDirectory("native-output")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    try
      val backend=new NativeDiffusionBackend(root,config)
      val events=scala.collection.mutable.ArrayBuffer.empty[String]
      val context=new NativeWorkContext(() => false,() => true,(name,_) => { events+=name;() })
      def run(kind: String,params: Map[String,NativeJson])=
        backend.execute(NativeRequest("r",kind,params,None),context)
      import NativeJson.*
      val catalog=run("list_models",Map("dirs"->Arr(Vector(Str(root.getParent.toString)))))
      assertEquals(catalog("models").asInstanceOf[Arr].values.size,1)
      assertEquals(run("list_models",Map("dirs"->Arr(Vector(Str(output.toString)))))("models"),Arr(Vector.empty))
      val samplers=run("list_samplers",Map.empty)("samplers").asInstanceOf[Arr].values
      assertEquals(samplers.map(_.asInstanceOf[Obj].fields("name")),Vector(Str("Euler"),Str("DPM++ 2M")))
      val metadata=run("load_model",Map("path"->Str(root.toString),"dtype"->Str("fp32"),"vram_mode"->Str("cpu")))
      assert(metadata("hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      assertEquals(backend.systemStatus("model").asInstanceOf[Obj].fields("hash"),metadata("hash"))
      val externalVae=output.resolve("external VAE.fp16.safetensors")
      Files.copy(root.resolve("vae/diffusion_pytorch_model.safetensors"),externalVae)
      val swapped=run("set_vae",Map("path"->Str(externalVae.toString)))
      assert(swapped("vae_hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      val retained=backend.systemStatus("model")
      intercept[Exception](run("set_vae",Map("path"->Str(output.resolve("missing.safetensors").toString))))
      assertEquals(backend.systemStatus("model"),retained)
      val deniedVae=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("vae-denied","set_vae",Map("path"->Null),None),deniedVae))
      assertEquals(backend.systemStatus("model"),retained)
      val restored=run("set_vae",Map("path"->Null))
      assertEquals(restored("vae_name"),Null)
      intercept[NativeBackendError](run("load_model",Map("dtype"->Str("fp16"))))
      val result=run("txt2img",NativeGenerationDefaults.envelope++Map("output_dir"->Str(output.toString),"vae_recovery_dir"->Str(output.toString),"params"->Obj(NativeGenerationDefaults.generation++Map(
        "checkpoint"->Str(root.toString),"vae"->Str("Automatic"),
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Automatic")))))
      assertEquals(result("scheduler"),Str("Karras"))
      val images=result("images").asInstanceOf[Arr].values
      val path=Path.of(images.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val image=javax.imageio.ImageIO.read(path.toFile)
      assertEquals(image.getWidth,10);assertEquals(image.getHeight,6)
      assertEquals(events.toVector,Vector("progress","progress"))
      def pixelsFor(prompt: String,negative: String="",clipSkip: Int=1): Vector[Int]=
        val generated=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
          "prompt"->Str(prompt),"negative_prompt"->Str(negative),"width"->Number(10),"height"->Number(6),
          "steps"->Number(2),"seed"->Number(42),"clip_skip"->Number(clipSkip)))))
        val file=generated("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val decoded=javax.imageio.ImageIO.read(Path.of(file).toFile)
        decoded.getRGB(0,0,10,6,null,0,10).toVector
      val plainPrompt=pixelsFor("a red cat")
      val target=scalatorch.models.PretrainedUNet.loadWithLinearTargets(root.resolve("unet"),maxParameters=1000000000L)
        .linearTargets("unet.time_embedding.linear_1")
      val adapterPath=output.resolve("tiny.safetensors")
      writeAdapter(adapterPath,"unet.time_embedding.linear_1",target.data.shape.dims(1).toInt,target.data.shape.dims(0).toInt)
      def loraPixels(weight: Double): Vector[Int]=
        val response=run("txt2img",Map("output_dir"->Str(output.toString),"lora_paths"->Obj(Map("tiny"->Str(adapterPath.toString))),
          "params"->Obj(Map("prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
            "seed"->Number(42),"loras"->Arr(Vector(Obj(Map("name"->Str("tiny"),"weight"->Number(weight),"unet"->Number(weight)))))))))
        val imagePath=response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val decoded=javax.imageio.ImageIO.read(Path.of(imagePath).toFile)
        decoded.getRGB(0,0,10,6,null,0,10).toVector
      assertEquals(loraPixels(0),plainPrompt,"zero-scale LoRA must preserve the unadapted image")
      val adaptedPrompt=loraPixels(1)
      assert(adaptedPrompt!=plainPrompt,"nonzero native LoRA must affect generated pixels")
      assertEquals(loraPixels(1),adaptedPrompt,"native LoRA replay must be deterministic")
      assertEquals(pixelsFor("a red cat"),plainPrompt,"adapter factors must not mutate the base checkpoint")
      val weightedPrompt=pixelsFor("a (red:1.5) cat")
      assert(weightedPrompt!=plainPrompt,"single-CLIP worker must pass prompt emphasis into the U-Net")
      assertEquals(pixelsFor("a (red:1.5) cat"),weightedPrompt,"weighted prompt replay must be deterministic")
      val chunkedPrompt=pixelsFor("apple, "*40,"blur BREAK fog")
      assertEquals(pixelsFor("apple, "*40,"blur BREAK fog"),chunkedPrompt,
        "single-CLIP worker must handle unequal positive/negative chunk counts deterministically")
      // Exercise the public request boundary, not only metadata parsing: overrides
      // must reach both samplers and must not mutate the checkpoint configuration.
      for sampler <- Vector("Euler","DPM++ 2M") do
        def pixels(extra: Map[String,NativeJson]): Vector[Int]=
          val generated=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
            "prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
            "seed"->Number(42),"sampler_name"->Str(sampler),"extra"->Obj(extra)))))
          val file=generated("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
          val decoded=javax.imageio.ImageIO.read(Path.of(file).toFile)
          decoded.getRGB(0,0,10,6,null,0,10).toVector
        val baseline=pixels(Map.empty)
        assertEquals(pixels(Map("Model keyword settings"->Str("{}"),"prediction_type"->Str("epsilon"))),baseline)
        assert(pixels(Map("prediction_type"->Str("v_prediction")))!=baseline)
        assertEquals(pixels(Map.empty),baseline)
      val uniform=run("txt2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"clip_skip"->Number(2),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Uniform")))))
      assertEquals(uniform("scheduler"),Str("Uniform"))
      val uniformPath=Path.of(uniform("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assertEquals(javax.imageio.ImageIO.read(uniformPath.toFile).getWidth,10)
      val iterativeParams=Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(1),
        "seed"->Number(100),"subseed"->Number(7),"n_iter"->Number(2))))
      val iterative=run("txt2img",iterativeParams)("images").asInstanceOf[Arr].values
      assertEquals(iterative.map(_.asInstanceOf[Obj].fields("seed")),Vector(Number(100),Number(101)))
      assertEquals(iterative.map(_.asInstanceOf[Obj].fields("subseed")),Vector(Number(7),Number(8)))
      val batchEvents=scala.collection.mutable.ArrayBuffer.empty[Map[String,NativeJson]]
      val batchContext=new NativeWorkContext(() => false,() => true,(_,fields) => batchEvents+=fields)
      val batchedParams=iterativeParams.updated("params",Obj(Map("prompt"->Str("cat"),"width"->Number(10),
        "height"->Number(6),"steps"->Number(1),"seed"->Number(200),"subseed"->Number(9),
        "batch_size"->Number(2),"n_iter"->Number(2))))
      val batched=backend.execute(NativeRequest("batch-size","txt2img",batchedParams,None),batchContext)("images").asInstanceOf[Arr].values
      assertEquals(batched.map(_.asInstanceOf[Obj].fields("seed")),Vector(200,201,202,203).map(v=>Number(BigDecimal(v))))
      assertEquals(batched.map(_.asInstanceOf[Obj].fields("subseed")),Vector(9,10,11,12).map(v=>Number(BigDecimal(v))))
      val batchProgress=batchEvents.filter(_.get("batch_size").contains(Number(2)))
      assert(batchProgress.nonEmpty,"progress must report the requested logical batch size")
      assert(batchProgress.forall(_.get("images").contains(Number(4))))
      val initCanvas=new java.awt.image.BufferedImage(100,60,java.awt.image.BufferedImage.TYPE_INT_RGB)
      val inpaintMask=new java.awt.image.BufferedImage(100,60,java.awt.image.BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 60; x <- 0 until 100 do
        initCanvas.setRGB(x,y,0xffff0000)
        inpaintMask.setRGB(x,y,if x>=40 && x<60 && y>=20 && y<40 then 0xffffffff else 0xff000000)
      val initPath=output.resolve("fullres-init.png");val maskPath=output.resolve("fullres-mask.png")
      javax.imageio.ImageIO.write(initCanvas,"png",initPath.toFile)
      javax.imageio.ImageIO.write(inpaintMask,"png",maskPath.toFile)
      val fullRes=run("img2img",Map("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("replace the center"),"width"->Number(64),"height"->Number(64),"steps"->Number(1),"seed"->Number(303),
        "img2img"->Obj(Map("init_image"->Str(initPath.toString),"mask"->Str(maskPath.toString),
          "denoising_strength"->Number(1),"resize_mode"->Number(0),"mask_blur"->Number(0),
          "inpaint_full_res"->Bool(true),"inpaint_full_res_padding"->Number(8)))))))
      val fullResEntry=fullRes("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals((fullResEntry("width"),fullResEntry("height")),(Number(100),Number(60)))
      val fullResImage=javax.imageio.ImageIO.read(Path.of(fullResEntry("path").asInstanceOf[Str].value).toFile)
      assertEquals((fullResImage.getWidth,fullResImage.getHeight),(100,60))
      assertEquals((fullResImage.getRGB(0,0)&0xffffff),0xff0000)
      assert((fullResImage.getRGB(50,30)&0xffffff)!=0xff0000)
      val custom=output.resolve("custom-mappings.txt")
      Files.writeString(custom,s"${metadata("hash").asInstanceOf[Str].value},a red|a blue,${root.getFileName}\n")
      val keywordParams=Map("prompt"->Str("cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"hooks"->Obj(Map("model_keyword"->Obj(Map("_mapping_dir"->Str(output.toString),"model_choice"->Str("iterate"))))))
      val keywordRequest=Map("output_dir"->Str(output.toString),"params"->Obj(keywordParams))
      def imageFields(result: Map[String,NativeJson])=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      def pixelsOf(fields: Map[String,NativeJson])=
        val img=javax.imageio.ImageIO.read(Path.of(fields("path").asInstanceOf[Str].value).toFile)
        img.getRGB(0,0,10,6,null,0,10).toVector
      val firstKeyword=imageFields(run("txt2img",keywordRequest))
      assertEquals(firstKeyword("effective_prompt"),Str("a red cat"))
      assertEquals(pixelsOf(firstKeyword),image.getRGB(0,0,10,6,null,0,10).toVector)
      val rejectKeyword=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("keyword-reject","txt2img",keywordRequest,None),rejectKeyword))
      intercept[IllegalArgumentException](run("txt2img",keywordRequest.updated("params",Obj(keywordParams.updated("width",Number(0))))))
      val secondKeyword=imageFields(run("txt2img",keywordRequest))
      assertEquals(secondKeyword("effective_prompt"),Str("a blue cat"))
      assertEquals(secondKeyword("model_keyword_source_prompt"),Str("a blue cat"))
      val directBlue=imageFields(run("txt2img",keywordRequest.updated("params",Obj(keywordParams.removed("hooks").updated("prompt",Str("a blue cat"))))))
      assertEquals(pixelsOf(secondKeyword),pixelsOf(directBlue))
      def filesNow()=
        val paths=Files.list(output)
        try
          import scala.jdk.CollectionConverters.*
          paths.iterator().asScala.toVector.map(_.getFileName.toString).toSet
        finally paths.close()
      val before=filesNow()
      var cancelBatch=false
      val cancelContext=new NativeWorkContext(() => cancelBatch,() => !cancelBatch,(_,fields) => {
        if fields.get("image").contains(Number(1)) then cancelBatch=true
      })
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("batch","txt2img",iterativeParams,None),cancelContext))
      assertEquals(filesNow(),before)
      val rejectPublication=new NativeWorkContext(() => false,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("reject","txt2img",iterativeParams,None),rejectPublication))
      assertEquals(filesNow(),before)
      val oversized=iterativeParams.updated("params",Obj(Map("seed"->Number(BigDecimal(Long.MaxValue)),"n_iter"->Number(2))))
      intercept[IllegalArgumentException](run("txt2img",oversized))
      val cancelled=new NativeWorkContext(() => true,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("x","unload",Map.empty,None),cancelled))
      val invalid=intercept[NativeBackendError](run("txt2img",Map("unknown"->Bool(true))))
      assertEquals(invalid.code,"unsupported")
      run("unload",Map.empty)
      assertEquals(backend.systemStatus("model"),Obj(Map.empty))
      intercept[NativeBackendError](run("txt2img",Map.empty))
    finally
      val files=Files.list(output)
      try files.forEach(p => { Files.delete(p);() }) finally files.close()
      Files.delete(output);Files.delete(config)
  }
