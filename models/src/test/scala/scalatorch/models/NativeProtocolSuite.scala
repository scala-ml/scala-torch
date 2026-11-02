package scalatorch.models

import scalatorch.models.worker.*

class NativeProtocolSuite extends munit.FunSuite:
  test("concurrent progress and results remain separate complete envelopes") {
    val out=new java.io.ByteArrayOutputStream()
    val emitter=new NativeEmitter(out)
    emitter.hello(Vector("native_test"))
    val threads=(0 until 4).map { worker =>
      new Thread(() => (0 until 20).foreach { i =>
        emitter.event(s"r$worker","progress",Map("step"->NativeJson.Number(BigDecimal(i))))
      })
    }
    threads.foreach(_.start());threads.foreach(_.join(5000))
    assert(threads.forall(!_.isAlive))
    val lines=out.toString(java.nio.charset.StandardCharsets.UTF_8).linesIterator.toVector
    assertEquals(lines.size,81)
    lines.foreach(line => assert(NativeJson.parse(line).isInstanceOf[NativeJson.Obj]))
    intercept[IllegalArgumentException](emitter.event("r","progress",Map("id"->NativeJson.Str("other"))))
  }
  test("request envelopes preserve owners exact seeds and fractional parameters") {
    val r=NativeRequest.parse("""{"id":"r1","type":"txt2img","owner_id":"job1","params":{"seed":9223372036854775807,"cfg_scale":7.5,"prompt":"猫"}}""")
    assertEquals(r.owner,Some("job1"))
    assertEquals(r.params("seed"),NativeJson.Number(BigDecimal(Long.MaxValue)))
    assertEquals(r.params("cfg_scale"),NativeJson.Number(BigDecimal("7.5")))
    assertEquals(NativeJson.parse(NativeJson.render(NativeJson.Obj(r.params))),NativeJson.Obj(r.params))
  }
  test("malformed envelopes and duplicate nested values reject") {
    for text <- Vector("[]","{}", """{"id":"a","type":"ping","owner_id":null}""",
      """{"id":"a","type":"ping","params":[]}""",
      """{"id":"a","type":"ping","params":{"x":1,"x":2}}""",
      """{"id":"a","type":"ping","image_password":"secret"}""", "{} {}") do
      intercept[IllegalArgumentException](NativeRequest.parse(text))
  }
  test("emitter escapes newlines and writes complete UTF-8 JSON Lines") {
    val out=new java.io.ByteArrayOutputStream()
    val emitter=new NativeEmitter(out)
    emitter.error(Some("r1"),"invalid_params","猫\nsecond line")
    emitter.result("r2",Map("ok"->NativeJson.Bool(true)))
    val lines=out.toString(java.nio.charset.StandardCharsets.UTF_8).linesIterator.toVector
    assertEquals(lines.size,2)
    val first=NativeJson.parse(lines.head).asInstanceOf[NativeJson.Obj].fields
    assertEquals(first("message"),NativeJson.Str("猫\nsecond line"))
    assertEquals(first("id"),NativeJson.Str("r1"))
  }
