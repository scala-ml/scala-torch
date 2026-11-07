package scalatorch.models.worker

import java.nio.file.Files
import NativeJson.*

class NativeVramCalibrationFailureSuite extends munit.FunSuite:
  test("non-OOM calibration failures stop that size group and mark the profile") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA test is opt-in")
    val output=Files.createTempDirectory("native-vram-error-")
    val identity=Obj(Map("device"->Str("test")))
    val ctx=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    var calls=0
    val result=NativeVramEstimator.benchmark(Map("max_size"->Number(320),"max_batch"->Number(1),
      "session_id"->Str("error-test"),"output_dir"->Str(output.toString)),identity,
      NativeVramEstimator.fingerprint(identity),0,ctx,_=>()) { (_,_,_,_) =>
      calls+=1
      if calls==1 then throw new IllegalStateException("decode failed")
    }.asInstanceOf[Obj].fields
    assertEquals(result("status"),Str("stopped_after_error"))
    assertEquals(calls,7)
    val rows=result("rows").asInstanceOf[Arr].values
    assertEquals(rows.head.asInstanceOf[Obj].fields("status"),Str("error"))
    assertEquals(rows.size,7)
    assert(!rows.exists { case Obj(row) => row("mode")==Str("txt2img") && row("target")==Number(320);case _ => false })
  }
  test("calibration records an out of memory failure and skips larger cases in that group") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA test is opt-in")
    val output=Files.createTempDirectory("native-vram-oom-")
    val identity=Obj(Map("device"->Str("test")))
    val fingerprint=NativeVramEstimator.fingerprint(identity)
    val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
    var calls=0
    val result=NativeVramEstimator.benchmark(Map("max_size"->Number(320),"max_batch"->Number(1),
      "session_id"->Str("oom-test"),"output_dir"->Str(output.toString)),identity,fingerprint,0,context,_=>()) {
      (_,_,_,_) => calls+=1;throw new OutOfMemoryError("simulated allocation failure")
    }.asInstanceOf[Obj].fields
    assertEquals(result("status"),Str("stopped_after_error"))
    val rows=result("rows").asInstanceOf[Arr].values
    assertEquals(calls,4)
    assertEquals(rows.size,4)
    assert(rows.forall(_.asInstanceOf[Obj].fields("status")==Str("oom")))
    for mode <- Vector("txt2img","txt2img_hr","txt2img_hr_latent","img2img") do
      val estimate=NativeVramEstimator.estimate(Obj(result),mode,256,256,1,1000,4096)
      assertEquals(estimate("available"),Bool(false))
      assertEquals(estimate("reason"),Str("measured_oom"))
    assertEquals(NativeJson.parse(Files.readString(java.nio.file.Path.of(result("profile_path").asInstanceOf[Str].value))),Obj(result))
  }
