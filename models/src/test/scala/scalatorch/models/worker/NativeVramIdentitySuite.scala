package scalatorch.models.worker

class NativeVramIdentitySuite extends munit.FunSuite:
  import NativeJson.*

  test("VRAM calibration is only compatible with the same offload and VAE tiling settings") {
    val model=Map[String,NativeJson]("name"->Str("sd15"),"hash"->Str("abc"))
    def state(vaeOffload: Boolean,unetOffload: Boolean,tiling: Boolean,record: Option[NativeJson]=None) =
      NativeVramEstimator.profile(model,"GPU",4L*1024*1024*1024,"cuda","cuda",record,true,
        vaeOffloadWeights=vaeOffload,unetOffloadWeights=unetOffload,vaeTiling=tiling)
    val baseline=state(true,true,false)
    val runtime=baseline("current_identity").asInstanceOf[Obj].fields("runtime").asInstanceOf[Obj].fields
    assertEquals(runtime("batch_execution"),Str("sequential"))
    val identity=baseline("current_identity")
    val fingerprint=baseline("current_fingerprint")
    val record=Obj(Map("version"->Number(1),"status"->Str("complete"),"identity"->identity,
      "fingerprint"->fingerprint,"rows"->Arr(Vector.empty)))
    assertEquals(state(true,true,false,Some(record))("compatible"),Bool(true))
    for (settings<-Vector((false,true,false),(true,false,false),(true,true,true))) do
      assertEquals(state(settings._1,settings._2,settings._3,Some(record))("compatible"),Bool(false))
    val changed=NativeVramEstimator.profile(model,"GPU",4L*1024*1024*1024,"cuda","cuda",Some(record),true,
      vaeOffloadWeights=true,unetOffloadWeights=true,maxVaeAttentionScoreElements=268435456L)
    assertEquals(changed("compatible"),Bool(false))
  }
