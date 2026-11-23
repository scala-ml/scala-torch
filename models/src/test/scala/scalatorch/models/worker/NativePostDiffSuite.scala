package scalatorch.models.worker

import NativeJson.*
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import java.util.concurrent.{LinkedBlockingQueue,TimeUnit}

class NativePostDiffSuite extends munit.FunSuite:
  test("paper AYS experiment selects ten-step SDE and rejects incompatible requests") {
    val root=Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")),"tiny SD fixture required")
    val output=Files.createTempDirectory("paper-ays-worker-test")
    try
      val scheduler=output.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val base=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(10),"seed"->Number(42),"sampler_name"->Str("DPM++ 2M SDE"),"scheduler"->Str("Karras"),"cfg_scale"->Number(1))
      val experimental=base.updated("extra",Obj(Map("paper_ays_experiment"->Str("1"))))
      def run(g: Map[String,NativeJson])=backend.execute(NativeRequest("generate","txt2img",
        Map("params"->Obj(g),"output_dir"->Str(output.toString)),None),context)
      def pixels(result: Map[String,NativeJson]) =
        val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val image=ImageIO.read(Path.of(path).toFile)
        (0 until 16).flatMap(y=>(0 until 16).map(x=>image.getRGB(x,y))).toVector
      val result=run(experimental)
      assertEquals(result("scheduler"),Str("paper AYS (experimental)"))
      assert(result("warnings").asInstanceOf[Arr].values.exists(_.asInstanceOf[Str].value.contains("PNG replay metadata")))
      assertNotEquals(pixels(result),pixels(run(base)))
      intercept[IllegalArgumentException](run(experimental.updated("steps",Number(9))))
      intercept[IllegalArgumentException](run(experimental.updated("sampler_name",Str("Euler"))))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }

  test("experimental mixed resolution options are explicit and strictly validated") {
    assertEquals(NativePostDiff.parse(None),None)
    assertEquals(NativePostDiff.parse(Some(Null)),None)
    val parsed=NativePostDiff.parse(Some(Obj(Map("scale"->Number(BigDecimal("0.75")),"portion"->Number(BigDecimal("0.2")))))).get
    assertEqualsDouble(parsed.scale,0.75,1e-10)
    assertEqualsDouble(parsed.portion,0.2,1e-10)
    assertEquals(NativePostDiff.parse(Some(Str("{\"scale\":0.75,\"portion\":0.2}"))),Some(parsed))
    for bad <- Seq(Obj(Map.empty),Bool(true),Obj(Map("scale"->Str("0.75"),"portion"->Number(BigDecimal("0.2")))),
      Obj(Map("scale"->Number(0),"portion"->Number(BigDecimal("0.2")))),
      Obj(Map("scale"->Number(BigDecimal("0.75")),"portion"->Number(1))),
      Obj(Map("scale"->Number(BigDecimal("0.75")),"portion"->Number(BigDecimal("0.2")),"unknown"->Bool(true)))) do
      intercept[IllegalArgumentException](NativePostDiff.parse(Some(bad)))
  }
  test("native worker forwards experimental settings and rejects incompatible samplers") {
    val root=Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")),"tiny SD fixture required")
    val output=Files.createTempDirectory("postdiff-native-test")
    try
      val scheduler=output.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val base=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(4),"seed"->Number(42),"sampler_name"->Str("Euler a"),"cfg_scale"->Number(1))
      val experiment=Obj(Map("postdiff_experiment"->Str("{\"scale\":0.5,\"portion\":0.5}")))
      def run(g: Map[String,NativeJson]) = backend.execute(NativeRequest("generate","txt2img",
        Map("params"->Obj(g),"output_dir"->Str(output.toString)),None),context)
      def image(g: Map[String,NativeJson]) =
        val path=run(g)("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        ImageIO.read(Path.of(path).toFile)
      val plain=image(base)
      val changed=image(base.updated("extra",experiment))
      assertEquals((changed.getWidth,changed.getHeight),(16,16))
      val pixels=(0 until 16).flatMap(y=>(0 until 16).map(x=>(plain.getRGB(x,y),changed.getRGB(x,y))))
      assert(pixels.exists((a,b)=>a!=b),"experimental path must affect generated pixels")
      // Public postdiff and cfg_cache hooks also run on DPM++ 2M SDE, while unsupported samplers (Heun) reject cleanly
      val hooked=base.updated("sampler_name",Str("DPM++ 2M SDE")).updated("scheduler",Str("Karras")).updated("cfg_scale",Number(7))
        .updated("hooks",Obj(Map("postdiff"->Obj(Map("scale"->Number(BigDecimal("0.5")),"portion"->Number(BigDecimal("0.5")))),
          "cfg_cache"->Obj(Map("start"->Number(BigDecimal("0.25")),"interval"->Number(2))))))
      val hookedImg=image(hooked)
      assertEquals((hookedImg.getWidth,hookedImg.getHeight),(16,16))
      intercept[IllegalArgumentException](run(base.updated("sampler_name",Str("Heun")).updated("extra",experiment)))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
  test("JSONL worker accepts experimental request and remains usable after rejection") {
    val root=Path.of(sys.env.getOrElse("SCALATORCH_POSTDIFF_TINY", ".superpowers/models/tiny-sd-pipe")).toAbsolutePath
    assume(Files.exists(root.resolve("unet/config.json")),"tiny SD fixture required")
    val output=Files.createTempDirectory("postdiff-jsonl-test")
    val queue=new LinkedBlockingQueue[String]()
    val sink=new java.io.OutputStream:
      override def write(b: Int): Unit=throw new UnsupportedOperationException()
      override def write(b: Array[Byte],off: Int,len: Int): Unit =
        queue.add(new String(b,off,len,java.nio.charset.StandardCharsets.UTF_8).trim)
    val scheduler=output.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val worker=new NativeWorker(sink,new NativeDiffusionBackend(root,scheduler))
    def receive(id: String): Map[String,NativeJson] =
      val deadline=System.nanoTime()+TimeUnit.SECONDS.toNanos(30)
      var found=Option.empty[Map[String,NativeJson]]
      while found.isEmpty && System.nanoTime()<deadline do
        val line=queue.poll(1,TimeUnit.SECONDS)
        if line!=null then
          val fields=NativeJson.parse(line).asInstanceOf[Obj].fields
          if fields.get("id").contains(Str(id)) && Set(Str("result"),Str("error")).contains(fields("type")) then found=Some(fields)
      found.getOrElse(fail(s"missing final reply for $id"))
    def send(id: String,kind: String,params: Map[String,NativeJson]): Unit =
      worker.handle(NativeJson.render(Obj(Map("id"->Str(id),"type"->Str(kind),"params"->Obj(params)))))
      ()
    try
      send("load","load_model",Map.empty)
      assertEquals(receive("load")("type"),Str("result"))
      val base=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(4),"seed"->Number(42),"sampler_name"->Str("Euler a"),"cfg_scale"->Number(1),
        "extra"->Obj(Map("postdiff_experiment"->Str("{\"scale\":0.5,\"portion\":0.5}"))))
      def request(params: Map[String,NativeJson])=Map[String,NativeJson]("params"->Obj(params),"output_dir"->Str(output.toString))
      send("bad","txt2img",request(base.updated("sampler_name",Str("Heun"))))
      val bad=receive("bad")
      assertEquals(bad("type"),Str("error"))
      assertEquals(bad("code"),Str("invalid_params"))
      send("good","txt2img",request(base))
      val good=receive("good")
      assertEquals(good("type"),Str("result"))
      val result=good("result").asInstanceOf[Obj].fields
      val path=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
      val image=ImageIO.read(Path.of(path).toFile)
      assertEquals((image.getWidth,image.getHeight),(16,16))
    finally
      worker.close()
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
