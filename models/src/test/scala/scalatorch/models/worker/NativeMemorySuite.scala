package scalatorch.models.worker

import java.nio.file.Files
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

class NativeMemorySuite extends munit.FunSuite:
  test("memory request reports unavailable device data without inventing GPU usage") {
    val root=Files.createTempDirectory("native-memory")
    val config=root.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,config)
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    assert(backend.supported.contains("memory"))
    val result=backend.execute(NativeRequest("m","memory",Map.empty,None),ctx)
    assertEquals(result("device_used_mb"),Null)
    assertEquals(result("total_mb"),Number(0))
    if sys.props("os.name").toLowerCase(java.util.Locale.ROOT).startsWith("windows") then
      assert(result("ram_used_mb").asInstanceOf[Number].value>0)
      val status=backend.systemStatus("memory").asInstanceOf[Obj].fields
      assert(status("process_rss_mb").asInstanceOf[Number].value>0)
      assert(status("process_private_mb").asInstanceOf[Number].value>0)
    intercept[IllegalArgumentException](backend.execute(NativeRequest("m","memory",Map("bad"->Bool(true)),None),ctx))
  }
  test("CUDA memory is reported as whole-device usage") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA required")
    val root=Files.createTempDirectory("native-memory-cuda")
    val config=root.resolve("scheduler.json")
    Files.writeString(config,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val backend=new NativeDiffusionBackend(root,config,unetExecution=UNetExecution.fromEnvironment(Map("SDFX_SCALATORCH_UNET_DEVICE"->"cuda")))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    val result=backend.execute(NativeRequest("m","memory",Map.empty,None),ctx)
    assert(result("total_mb").asInstanceOf[Number].value>0)
    assert(result("device_used_mb").asInstanceOf[Number].value>=0)
    assertEquals(result("allocated_mb"),Number(0))
    val gpu=backend.systemStatus("memory").asInstanceOf[Obj].fields("gpu").asInstanceOf[Obj].fields
    assertEquals(gpu("used_mb"),result("device_used_mb"))
    assertEquals(gpu("total_mb"),result("total_mb"))
    val info=backend.execute(NativeRequest("i","system_info",Map.empty,None),ctx)
    val device=info("gpu").asInstanceOf[Obj].fields
    assert(device("device").asInstanceOf[Str].value.nonEmpty)
    assertEquals(device("total_mb"),result("total_mb"))
  }
