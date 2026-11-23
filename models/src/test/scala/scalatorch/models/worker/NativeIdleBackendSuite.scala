package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.models.worker.NativeJson.*

class NativeIdleBackendSuite extends munit.FunSuite:
  private val ctx=new NativeWorkContext(() => false,() => true,(_,_) => ())

  test("schedulerFor handles idle initial '-' root cleanly") {
    val file=Files.createTempFile("idle-model",".safetensors")
    val dir=Files.createTempDirectory("idle-diffusers")
    try
      assertEquals(NativeWorkerMain.schedulerFor(Path.of("-"),Path.of("-"),Path.of("-")),Path.of("-"))
      assertEquals(NativeWorkerMain.schedulerFor(Path.of("-"),Path.of("-"),file),Path.of("-"))
      assertEquals(NativeWorkerMain.schedulerFor(Path.of("-"),Path.of("-"),dir),dir.resolve("scheduler/scheduler_config.json"))
    finally
      Files.deleteIfExists(file)
      Files.deleteIfExists(dir)
  }

  test("switching backend starting with '-' starts idle without touching disk models") {
    val backend=new NativeSwitchingBackend(Path.of("-"),path =>
      new NativeDiffusionBackend(path,NativeWorkerMain.schedulerFor(Path.of("-"),Path.of("-"),path)))
    assert(backend.features.contains("deepcache"))
    assert(backend.supported.contains("txt2img"))
    val samplers=backend.execute(NativeRequest("1","list_samplers",Map.empty,None),ctx)
    assert(samplers.contains("samplers"))
    assertEquals(backend.execute(NativeRequest("2","unload",Map.empty,None),ctx),Map.empty[String,NativeJson])
    val err=intercept[NativeBackendError] {
      backend.execute(NativeRequest("3","txt2img",Map.empty,None),ctx)
    }
    assertEquals(err.code,"no_model")
    intercept[IllegalArgumentException] {
      backend.execute(NativeRequest("4","load_model",Map.empty,None),ctx)
    }
  }
