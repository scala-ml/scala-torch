package scalatorch.models.worker

import java.nio.file.{Files,Path}
import NativeJson.*

class NativeSwitchingBackendSuite extends munit.FunSuite:
  test("a configured ControlNet directory itself is listed as a model") {
    val base=Files.createTempDirectory("controlnet-direct")
    Files.writeString(base.resolve("config.json"),"{}")
    Files.createFile(base.resolve("diffusion_pytorch_model.safetensors"))
    val entries=NativeFileCatalog.scan("controlnet",Vector(base.toString),()=>())
    assertEquals(entries.size,1)
    assertEquals(entries.head.fields("path"),Str(base.toString))
    assertEquals(entries.head.fields("name"),Str(base.getFileName.toString))
  }
  test("catalog discovers checkpoints and diffusers models without listing their component files") {
    val base=Files.createTempDirectory("native-model-list")
    val sub=Files.createDirectory(base.resolve("sub"))
    val checkpoint=Files.createFile(sub.resolve("model.safetensors"))
    val model=Files.createDirectory(base.resolve("pipeline"))
    for component <- Vector("unet","vae","text_encoder","tokenizer") do Files.createDirectory(model.resolve(component))
    Files.createFile(model.resolve("unet/config.json"))
    Files.createFile(model.resolve("unet/weights.safetensors"))
    Files.createFile(base.resolve("notes.txt"))
    val entries=NativeModelCatalog.scan(Vector(base.toString,sub.toString),()=>())
    assertEquals(entries.size,2)
    assertEquals(entries.map(_.fields("path")).toSet,Set(Str(checkpoint.toRealPath().toString),Str(model.toRealPath().toString)))
    assertEquals(entries.map(_.fields("kind")).toSet,Set(Str("safetensors"),Str("diffusers")))
  }
  test("default scheduler follows selected model and explicit scheduler remains fixed") {
    val dir=Files.createTempDirectory("switch-schedulers")
    val a=Files.createDirectory(dir.resolve("a"))
    val b=Files.createDirectory(dir.resolve("b"))
    val file=Files.createFile(dir.resolve("model.safetensors"))
    val default=a.resolve("scheduler/scheduler_config.json")
    val custom=dir.resolve("custom.json")
    assertEquals(NativeWorkerMain.schedulerFor(a,default,b),b.resolve("scheduler/scheduler_config.json"))
    assertEquals(NativeWorkerMain.schedulerFor(a,default,file),Path.of("-"))
    assertEquals(NativeWorkerMain.schedulerFor(file,Path.of("-"),b),b.resolve("scheduler/scheduler_config.json"))
    assertEquals(NativeWorkerMain.schedulerFor(a,custom,b),custom)
  }
  test("switch publishes only a successfully loaded backend") {
    val dir=Files.createTempDirectory("native-switch")
    val a=Files.createFile(dir.resolve("a.safetensors")).toRealPath()
    val b=Files.createFile(dir.resolve("b.safetensors")).toRealPath()
    val broken=Files.createFile(dir.resolve("broken.safetensors")).toRealPath()
    class TestBackend(path: Path) extends NativeWorkerBackend:
      val supported=Set("load_model","txt2img")
      override def systemStatus=Map("path"->Str(path.toString))
      def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson] =
        if path==broken then throw IllegalArgumentException("broken model")
        context.beginPublication()
        Map("path"->Str(path.toString))
    val backend=new NativeSwitchingBackend(a,p => new TestBackend(p))
    val allowed=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val denied=new NativeWorkContext(()=>false,()=>false,(_,_)=>())
    def load(path: Path,c: NativeWorkContext)=backend.execute(NativeRequest("load","load_model",Map("path"->Str(path.toString)),None),c)
    load(a,allowed)
    intercept[IllegalArgumentException](load(broken,allowed))
    assertEquals(backend.systemStatus("path"),Str(a.toString))
    intercept[java.util.concurrent.CancellationException](load(b,denied))
    assertEquals(backend.systemStatus("path"),Str(a.toString))
    load(b,allowed)
    assertEquals(backend.systemStatus("path"),Str(b.toString))
    assertEquals(backend.execute(NativeRequest("image","txt2img",Map.empty,None),allowed)("path"),Str(b.toString))
  }

  test("two real diffusers models can be loaded and used in one worker") {
    val configured=sys.env.get("SCALATORCH_SWITCH_MODELS")
    assume(configured.nonEmpty,"two tiny model paths required")
    val roots=configured.get.split(java.io.File.pathSeparator).map(Path.of(_).toRealPath())
    assertEquals(roots.length,2)
    val backend=new NativeSwitchingBackend(roots(0),p => new NativeDiffusionBackend(p,p.resolve("euler-demo.json")))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val output=Files.createTempDirectory("native-switch-images")
    for root <- roots do
      val model=backend.execute(NativeRequest("load","load_model",Map("path"->Str(root.toString)),None),ctx)
      assertEquals(model("path"),Str(root.toString))
      val generated=backend.execute(NativeRequest("image","txt2img",Map("output_dir"->Str(output.toString),
        "params"->Obj(Map("width"->Number(16),"height"->Number(16),"steps"->Number(1)))),None),ctx)
      assertEquals(generated("images").asInstanceOf[Arr].values.size,1)
  }
