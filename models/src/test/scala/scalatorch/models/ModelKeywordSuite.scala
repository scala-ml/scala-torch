package scalatorch.models

class ModelKeywordSuite extends munit.FunSuite:
  import ModelKeyword.*
  test("static NUI planner fixtures cover choice placement order seeds and empty mappings") {
    import scalatorch.models.worker.NativeJson
    import NativeJson.*
    val source=scala.io.Source.fromInputStream(this.getClass.getResourceAsStream("/model-keyword/reference.jsonl"),"UTF-8")
    try
      val lines=source.getLines().toVector
      assertEquals(lines.size,144)
      for line <- lines do
        val fields=NativeJson.parse(line).asInstanceOf[Obj].fields
        def optional(value: NativeJson): Option[String]=value match
          case Null => None
          case Str(text) => Some(text)
          case _ => fail("invalid fixture")
        val o=fields("options").asInstanceOf[Obj].fields
        def text(key: String)=o(key).asInstanceOf[Str].value
        val options=Options(text("model_choice"),text("lora_choice"),text("placement"),text("order"),
          optional(o("embedding")),optional(o("lora_model")),text("multiplier").toDouble)
        val result=plan("preview",Vector("one","two"),optional(fields("model")),optional(fields("lora")),options,State(3,2),fields("seed").asInstanceOf[Str].value.toLong)
        assertEquals(result.preview,fields("preview").asInstanceOf[Str].value)
        assertEquals(result.prompts,fields("prompts").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value))
        assertEquals(result.nextState,State(fields("next_model").asInstanceOf[Number].value.toBigInt,fields("next_lora").asInstanceOf[Number].value.toBigInt))
    finally source.close()
  }
  test("integer seed selection matches CPython static oracle including signed limits") {
    val bounds=Vector(2,3,7,2,16,31,1000,3)
    val references=Vector(0L->Vector(1,1,0,1,15,12,940,1),42L->Vector(0,0,5,1,7,7,142,2),
      -42L->Vector(0,0,5,1,7,7,142,2),Long.MaxValue->Vector(1,2,5,0,15,14,111,1),
      Long.MinValue->Vector(1,0,1,0,11,24,341,2))
    for (seed,expected) <- references do
      val rng=new KeywordRandom(seed)
      assertEquals(bounds.map(rng.below),expected)
  }
  test("random planning consumes preview choices before per-image choices") {
    val result=plan("preview",Vector("one","two","three"),Some("red|blue|green"),Some("x|y"),
      Options(modelChoice="random",loraChoice="< random >",loraModel=Some("style.safetensors"),order="model keyword first"),State(),42)
    assertEquals(result.preview,"<lora:style:0.7> green x preview")
    assertEquals(result.prompts,Vector("<lora:style:0.7> red y one","<lora:style:0.7> red x two","<lora:style:0.7> red x three"))
    assertEquals(result.nextState,State())
  }
  test("choice sequence remains compatible across multiple MT state refreshes") {
    val rng=new KeywordRandom(42)
    val buffer=java.nio.ByteBuffer.allocate(2000*4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    for _ <- 0 until 2000 do buffer.putInt(rng.below(1000))
    val digest=java.security.MessageDigest.getInstance("SHA-256").digest(buffer.array())
    assertEquals(java.util.HexFormat.of().formatHex(digest),"6168e151b402a4b314b512d22e72eb66c7623dac3c0b7e18b483274514eb1c05")
  }
  test("keyword choices preserve separators, explicit choices and fallback") {
    val rng=new KeywordRandom(42)
    assertEquals(choose(None,"iterate",0,rng),None)
    assertEquals(choose(Some("a| b |c"),"keyword1, keyword2",0,rng),Some("a, b, c"))
    assertEquals(choose(Some("a|b"),"keyword2",0,rng),Some("b"))
    assertEquals(choose(Some("a|b"),"stale",0,rng),Some("a"))
    assertEquals(choose(Some("a|b"),"b",0,rng),Some("b"))
    assertEquals(choose(Some("a|"),"keyword2",0,rng),Some(""))
    assertEquals(choose(Some("one"),"keyword2",0,rng),Some("one"))
  }
  test("placement order and LoRA tag reproduce NUI prompt text") {
    val options=Options(placement="keyword, prompt",order="model keyword first",embedding=Some("embedding"),loraModel=Some("nested/style.safetensors"))
    assertEquals(render("portrait",Some("model"),Some("lora trigger"),options),
      "<lora:style:0.7>, model, lora trigger, embedding, portrait")
    for (placement,expected) <- Vector("keyword prompt"->"key portrait","prompt keyword"->"portrait key",
        "keyword, prompt"->"key, portrait","prompt, keyword"->"portrait, key") do
      assertEquals(render("portrait",Some("key"),None,Options(placement=placement)),expected)
    assertEquals(render("portrait",None,None,Options()),"portrait")
    for (weight,text) <- Vector(0.00001->"1e-05",0.0001->"0.0001",1.0->"1.0",-0.0->"-0.0") do
      assertEquals(render("p",None,None,Options(loraModel=Some("x.safetensors"),multiplier=weight)),s"<lora:x:$text> p")
  }
  test("iteration advances independently, preview does not advance counters") {
    val options=Options(modelChoice="iterate",loraChoice="< iterate >",loraModel=Some("style.safetensors"),
      placement="keyword, prompt",order="model keyword first",embedding=Some("ti"),multiplier=0.5)
    val result=plan("preview",Vector("one","two","three"),Some("red|blue"),Some("x|y|z"),options,State(0,1),4)
    assertEquals(result.preview,"<lora:style:0.5>, red, y, ti, preview")
    assertEquals(result.prompts,Vector("<lora:style:0.5>, red, y, ti, one","<lora:style:0.5>, blue, z, ti, two","<lora:style:0.5>, red, x, ti, three"))
    assertEquals(result.nextState,State(3,4))
    val next=plan("",Vector("four"),Some("red|blue"),Some("x|y|z"),options,result.nextState,4)
    assert(next.prompts.head.contains("blue, y, ti, four"))
  }
  test("invalid options fail before planning and no selection stays unchanged") {
    intercept[IllegalArgumentException](plan("",Vector("a"),None,None,Options(placement="typo"),State(),0))
    intercept[IllegalArgumentException](plan("",Vector("a"),None,None,Options(multiplier=Double.NaN),State(),0))
    val result=plan("portrait",Vector("portrait"),None,None,Options(),State(),0)
    assertEquals(result.prompts,Vector("portrait"))
    assertEquals(result.nextState,State())
  }
