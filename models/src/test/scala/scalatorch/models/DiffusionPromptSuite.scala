package scalatorch.models

class DiffusionPromptSuite extends munit.FunSuite:
  test("attention, repetition, BREAK, comma relocation and padding match NUI") {
    val stream=getClass.getResourceAsStream("/nui-prompt.json")
    val json=try new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8) finally stream.close()
    for sample <- TokenizerJson.parse(json).arr do
      val fields=sample.obj
      val spans=DiffusionPrompt.parse(fields("text").str)
      val expected=fields("spans").arr.map { s =>
        PromptSpan(s.arr(0).str,s.arr(1).str.toDouble)
      }
      assertEquals(spans.map(_.text),expected.map(_.text),fields("text").str)
      spans.zip(expected).foreach((a,b) => assertEqualsDouble(a.weight,b.weight,1e-12))
      val chunks=DiffusionPrompt.chunks(spans,s => s.codePoints().toArray.map(_.toLong),1,2,0,Some(44L))
      val refs=fields("chunks").arr
      assertEquals(chunks.size,refs.size)
      chunks.zip(refs).foreach { (chunk,ref) =>
        assertEquals(chunk.tokens,ref.obj("tokens").arr.map(_.int.toLong))
        assertEquals(chunk.weights.size,chunk.tokens.size)
        chunk.weights.zip(ref.obj("weights").arr).foreach((a,b) => assertEqualsDouble(a,b.str.toDouble,1e-12))
      }
  }

  test("untrusted repeat and nesting expansion is bounded") {
    intercept[IllegalArgumentException](DiffusionPrompt.parse("(a:x999999999999999999999999)"))
    intercept[IllegalArgumentException](DiffusionPrompt.parse("("*4097+"a"))
    intercept[IllegalArgumentException](DiffusionPrompt.parse("(a:"+"9"*400+")"))
  }

  test("chunk limits, exact boundaries, trailing BREAK and first EOS padding") {
    def encode(s: String)=s.codePoints().toArray.map(_.toLong)
    def chunk(s: String)=DiffusionPrompt.chunks(DiffusionPrompt.parse(s),encode,1,2,0,contentLength=3)
    assertEquals(chunk("abc").map(_.tokens),Vector(Vector[Long](1,97,98,99,2)))
    assertEquals(chunk("abc BREAK").size,1)
    assertEquals(chunk("BREAK").map(_.tokens),Vector(Vector[Long](1,2,0,0,0)))
    assertEquals(chunk("abcd").last.tokens,Vector[Long](1,100,2,0,0))
    assertEquals(DiffusionPrompt.chunks(Vector(PromptSpan("x",2)),_ => Array[Long](8,2,9),1,2,0,
      contentLength=3).head.tokens,Vector[Long](1,8,2,0,0))
    intercept[IllegalArgumentException](chunk("a"*769))
    intercept[IllegalArgumentException](DiffusionPrompt.chunks(Vector(PromptSpan("a",Double.NaN)),encode,1,2,0))
  }
