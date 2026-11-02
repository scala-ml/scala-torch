package scalatorch.models.worker

import java.nio.file.{Files,Path}

class NativeDiffusionBackendSuite extends munit.FunSuite:
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
      val metadata=run("load_model",Map("path"->Str(root.toString),"dtype"->Str("fp32"),"vram_mode"->Str("cpu")))
      assert(metadata("hash").asInstanceOf[Str].value.matches("[0-9a-f]{64}"))
      intercept[NativeBackendError](run("load_model",Map("dtype"->Str("fp16"))))
      val result=run("txt2img",NativeGenerationDefaults.envelope++Map("output_dir"->Str(output.toString),"params"->Obj(NativeGenerationDefaults.generation++Map(
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(2),
        "seed"->Number(42),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Automatic")))))
      assertEquals(result("scheduler"),Str("Karras"))
      val images=result("images").asInstanceOf[Arr].values
      val path=Path.of(images.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val image=javax.imageio.ImageIO.read(path.toFile)
      assertEquals(image.getWidth,10);assertEquals(image.getHeight,6)
      assertEquals(events.toVector,Vector("progress","progress"))
      val cancelled=new NativeWorkContext(() => true,() => false,(_,_) => ())
      intercept[java.util.concurrent.CancellationException](backend.execute(NativeRequest("x","unload",Map.empty,None),cancelled))
      val invalid=intercept[NativeBackendError](run("txt2img",Map("unknown"->Bool(true))))
      assertEquals(invalid.code,"unsupported")
      run("unload",Map.empty)
      intercept[NativeBackendError](run("txt2img",Map.empty))
    finally
      val files=Files.list(output)
      try files.forEach(p => { Files.delete(p);() }) finally files.close()
      Files.delete(output);Files.delete(config)
  }
