package scalatorch.models.worker

import NativeJson.*

class NativeVramEstimateOomSuite extends munit.FunSuite:
  test("a matching calibration OOM is not reported as a safe estimate") {
    val identity=Obj(Map("name"->Str("test")))
    def row(side: Int,batch: Int,status: String,delta: Int=0): NativeJson =
      Obj(Map("mode"->Str("txt2img"),"pixels"->Number(side*side),"batch_size"->Number(batch),
        "status"->Str(status),"reserved_delta_mb"->Number(delta)))
    val profile=Obj(Map("version"->Number(1),"status"->Str("stopped_after_error"),"identity"->identity,
      "fingerprint"->Str(NativeVramEstimator.fingerprint(identity)),"rows"->Arr(Vector(
        row(256,1,"complete",100),row(384,1,"complete",200),row(512,1,"oom")))))
    def estimate(width: Int,height: Int,batch: Int)=NativeVramEstimator.estimate(profile,"txt2img",width,height,batch,1000,4096)
    assertEquals(estimate(512,512,1)("available"),Bool(false))
    assertEquals(estimate(512,512,1)("reason"),Str("measured_oom"))
    assertEquals(estimate(384,384,1)("available"),Bool(true))
    assertNotEquals(estimate(256,1024,1).get("reason"),Some(Str("measured_oom")))
    assertNotEquals(estimate(512,512,2).get("reason"),Some(Str("measured_oom")))
  }
