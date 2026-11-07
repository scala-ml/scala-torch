package scalatorch.models.worker

class NativeLoraPromptSuite extends munit.FunSuite:
  test("parse and remove LoRA directives with separate text and UNet weights") {
    val result=NativeLoraPrompt.parse("cat <lora:one:0.5:0.25> <lyco:two:te=0.3:unet=0.8>")
    assertEquals(result._1,"cat  ")
    assertEquals(result._2,Vector(("one",0.5f,0.25f,None,None,None,None),("two",0.3f,0.8f,None,None,None,None)))
    assertEquals(NativeLoraPrompt.parse("<lora:x:1:0.5:-1> <lyco:y:dyn=2>")._2,
      Vector(("x",1f,.5f,Some(-1),None,None,None),("y",1f,1f,Some(2),None,None,None)))
    intercept[IllegalArgumentException](NativeLoraPrompt.parse("<lora:x:dyn=1.5>"))
  }
  test("reject unsupported network types and LoRA schedules") {
    intercept[NativeBackendError](NativeLoraPrompt.parse("<hypernet:x:1>"))
    assertEquals(NativeLoraPrompt.parse("<lora:x:1:start=0.2:stop=0.8>")._2.head._5,Some(.2))
    val window=NativeLoraPrompt.parse("<lora:x:1:step=0.25-0.75>")._2.head
    assertEquals((window._5,window._6),(Some(.25),Some(.75)))
    val overrideWindow=NativeLoraPrompt.parse("<lora:x:1:step=1-4:start=2>")._2.head
    assertEquals((overrideWindow._5,overrideWindow._6),(Some(2.0),Some(4.0)))
    assertEquals(NativeLoraPrompt.parse("<lora:x:step=bad>")._2.head._5,None)
    assertEquals(NativeLoraPrompt.parse("<lora:x:1:lbw=ALL>")._2.head._7,Some("ALL"))
    intercept[IllegalArgumentException](NativeLoraPrompt.parse("<lora:x:NaN>"))
  }
