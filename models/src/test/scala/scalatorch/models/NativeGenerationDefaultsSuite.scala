package scalatorch.models.worker

class NativeGenerationDefaultsSuite extends munit.FunSuite:
  import NativeJson.*
  test("NUI inactive options accept defaults but reject active or unknown nested options") {
    val defaults=NativeGenerationDefaults.generation
    assertEquals(NativeGenerationDefaults.strip(defaults.updated("prompt",Str("cat")),defaults),Map("prompt"->Str("cat")))
    assertEquals(NativeGenerationDefaults.strip(Map("batch_size"->Number(2)),defaults),Map("batch_size"->Number(2)))
    for (key,value) <- Vector("hires"->Obj(Map("typo"->Bool(false))),
        "loras"->Arr(Vector(Str("lora"))),"eta"->Number(0)) do
      intercept[NativeBackendError](NativeGenerationDefaults.strip(Map(key->value),defaults))
    val active=Obj(Map("enabled"->Bool(true)))
    assertEquals(NativeGenerationDefaults.strip(Map("hires"->active),defaults),Map("hires"->active),
      "enabled Hires must reach strict backend parsing")
    assertEquals(NativeGenerationDefaults.strip(Map("hires"->Obj(Map("enabled"->Bool(false)))),defaults),Map.empty)
    assertEquals(NativeGenerationDefaults.strip(NativeGenerationDefaults.envelope,NativeGenerationDefaults.envelope),Map.empty)
    assertEquals(NativeGenerationDefaults.strip(Map("manual_loras"->Arr(Vector.empty)),NativeGenerationDefaults.envelope),Map.empty)
    intercept[NativeBackendError](NativeGenerationDefaults.strip(Map("manual_loras"->Arr(Vector(Str("x")))),NativeGenerationDefaults.envelope))
    for key <- Vector("hires_loras","img2img_hires_loras") do
      intercept[NativeBackendError](NativeGenerationDefaults.strip(Map(key->Arr(Vector(Str("adapter")))),NativeGenerationDefaults.envelope))
  }
  test("desktop disabled ControlNet slots are inactive, active slots pass through, malformed slots fail") {
    val disabled=Obj(Map("enabled"->Bool(false),"model"->Str("None"),"weight"->Number(1)))
    assertEquals(NativeGenerationDefaults.strip(Map("controlnet"->Arr(Vector(disabled))),NativeGenerationDefaults.generation),Map.empty)
    val active=Arr(Vector(disabled,Obj(Map("enabled"->Bool(true)))))
    assertEquals(NativeGenerationDefaults.strip(Map("controlnet"->active),NativeGenerationDefaults.generation),Map("controlnet"->active))
    for slot <- Vector(Obj(Map.empty),Obj(Map("enabled"->Str("false"))),Null) do
      intercept[NativeBackendError](NativeGenerationDefaults.strip(Map("controlnet"->Arr(Vector(disabled,slot))),NativeGenerationDefaults.generation))
  }
