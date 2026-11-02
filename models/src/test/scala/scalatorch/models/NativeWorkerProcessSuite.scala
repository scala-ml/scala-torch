package scalatorch.models

import scalatorch.models.worker.*
import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets.UTF_8
import java.util.concurrent.{LinkedBlockingQueue,TimeUnit}

class NativeWorkerProcessSuite extends munit.FunSuite:
  test("public worker subprocess speaks JSON Lines and generates a real PNG") {
    runWorker(false)
  }
  test("public worker subprocess selects CUDA VAE decode and generates a real PNG") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    runWorker(true)
  }
  test("public worker subprocess runs CUDA U-Net and VAE with progress and PNG output") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    runWorker(true,true)
  }
  test("public worker offloads CUDA weights under a two MiB allocation budget") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    runWorker(true,true,true)
  }
  private def runWorker(cuda: Boolean,unetCuda: Boolean=false,offload: Boolean=false): Unit = {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val root=Path.of(configured.get).getParent
    val temp=Files.createTempDirectory("native-worker-process")
    val config=temp.resolve("scheduler.json")
    val errors=temp.resolve("stderr.txt")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val javaExe=Path.of(sys.props("java.home"),"bin",if sys.props("os.name").startsWith("Windows") then "java.exe" else "java")
    val builder=new ProcessBuilder(javaExe.toString,"--enable-native-access=ALL-UNNAMED","-Dfile.encoding=UTF-8","-cp",sys.props("scalatorch.worker.classpath"),
      "scalatorch.models.worker.NativeWorkerMain",root.toString,config.toString)
    builder.redirectError(errors.toFile)
    // The worker must not depend on inherited Python or test JVM configuration.
    builder.environment().remove("PYTHONPATH")
    builder.environment().remove("PYTHONHOME")
    builder.environment().remove("JAVA_TOOL_OPTIONS")
    builder.environment().put("SDFX_SCALATORCH_VAE_DEVICE",if cuda then "cuda" else "cpu")
    builder.environment().remove("SDFX_SCALATORCH_VAE_MAX_BYTES")
    builder.environment().put("SDFX_SCALATORCH_UNET_DEVICE",if unetCuda then "cuda" else "cpu")
    builder.environment().remove("SDFX_SCALATORCH_UNET_MAX_BYTES")
    for component <- Seq("UNET","VAE") do
      builder.environment().put(s"SDFX_SCALATORCH_${component}_OFFLOAD_WEIGHTS",offload.toString)
      if offload then builder.environment().put(s"SDFX_SCALATORCH_${component}_MAX_BYTES","2097152")
    val process=builder.start()
    val input=new java.io.BufferedWriter(new java.io.OutputStreamWriter(process.getOutputStream,UTF_8))
    val lines=new LinkedBlockingQueue[String]()
    val reader=new Thread(() => {
      val stdout=new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream,UTF_8))
      try
        var line=stdout.readLine()
        while line!=null do
          lines.put(line)
          line=stdout.readLine()
      finally stdout.close()
    },"native-worker-test-reader")
    reader.setDaemon(true)
    reader.start()
    import NativeJson.*
    def receive(): Map[String,NativeJson]=
      val line=lines.poll(20,TimeUnit.SECONDS)
      assert(line!=null,s"worker response timeout; alive=${process.isAlive}; stderr=${new String(Files.readAllBytes(errors),UTF_8)}")
      NativeJson.parse(line).asInstanceOf[Obj].fields
    def send(id: String,kind: String,params: Map[String,NativeJson]=Map.empty): Unit=
      input.write(NativeJson.render(Obj(Map("id"->Str(id),"type"->Str(kind),"params"->Obj(params)))))
      input.newLine();input.flush()
    try
      val hello=receive()
      assertEquals(hello("type"),Str("hello"))
      assertEquals(hello("python"),Str(""))
      assertEquals(hello("device"),Str(if unetCuda then "cuda:0" else "cpu"))
      assertEquals(hello("cuda"),Bool(cuda || unetCuda))
      send("load","load_model",Map("path"->Str(root.toString),"dtype"->Str("fp32"),"vram_mode"->Str("cpu")))
      assertEquals(receive()("type"),Str("result"))
      send("status","system_status")
      val status=receive()("result").asInstanceOf[Obj].fields
      assertEquals(status("model").asInstanceOf[Obj].fields("name"),Str(root.getFileName.toString))
      assertEquals(status("runtime").asInstanceOf[Obj].fields("vae_device"),Str(if cuda then "cuda:0" else "cpu"))
      assertEquals(status("runtime").asInstanceOf[Obj].fields("unet_device"),Str(if unetCuda then "cuda:0" else "cpu"))
      assertEquals(status("runtime").asInstanceOf[Obj].fields("unet_offload_weights"),Bool(offload))
      assertEquals(status("runtime").asInstanceOf[Obj].fields("vae_offload_weights"),Bool(offload))
      send("generate","txt2img",Map("output_dir"->Str(temp.toString),"params"->Obj(Map(
        "prompt"->Str("a red cat"),"width"->Number(10),"height"->Number(6),"steps"->Number(3),
        "seed"->Number(BigDecimal(Long.MaxValue)),"sampler_name"->Str("DPM++ 2M"),"scheduler"->Str("Karras")))))
      for step <- 1 to 2 do
        val event=receive()
        assertEquals(event("event"),Str("progress"))
        assertEquals(event("step"),Number(step))
      val preview=receive()
      assertEquals(preview("event"),Str("preview"))
      assertEquals(preview("step"),Number(2))
      val previewBytes=java.util.Base64.getDecoder.decode(preview("jpeg_b64").asInstanceOf[Str].value)
      val previewImage=javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(previewBytes))
      assertEquals(previewImage.getWidth,256)
      assertEquals(previewImage.getHeight,154)
      val lastProgress=receive()
      assertEquals(lastProgress("event"),Str("progress"))
      assertEquals(lastProgress("step"),Number(3))
      val response=receive()
      assertEquals(response("id"),Str("generate"))
      val result=response("result").asInstanceOf[Obj].fields
      val image=result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals(image("seed"),Number(BigDecimal(Long.MaxValue)))
      val png=javax.imageio.ImageIO.read(Path.of(image("path").asInstanceOf[Str].value).toFile)
      assertEquals(png.getWidth,10);assertEquals(png.getHeight,6)
      send("unload","unload")
      assertEquals(receive()("type"),Str("result"))
      send("empty","txt2img")
      assertEquals(receive()("code"),Str("no_model"))
      send("shutdown","shutdown")
      assertEquals(receive()("id"),Str("shutdown"))
      assert(process.waitFor(5,TimeUnit.SECONDS),"worker did not exit after shutdown")
      assertEquals(process.exitValue(),0)
    finally
      if process.isAlive then process.destroyForcibly().waitFor(5,TimeUnit.SECONDS)
      input.close()
      reader.join(1000)
      val files=Files.list(temp)
      try files.forEach(p => { Files.delete(p);() }) finally files.close()
      Files.delete(temp)
  }
