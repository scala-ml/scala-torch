package scalatorch.models.worker

import java.nio.file.{Files,Path}

class NativeDiffusionBackendSuite extends munit.FunSuite:
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
