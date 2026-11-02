package scalatorch.models

import scalatorch.models.worker.*
import java.util.concurrent.{CountDownLatch,LinkedBlockingQueue,TimeUnit}

class NativeWorkerSuite extends munit.FunSuite:
  test("stream dispatch recovers malformed input and handles protection and shutdown") {
    val out=new java.io.ByteArrayOutputStream()
    val backend=new NativeWorkerBackend:
      def supported=Set.empty[String]
      def execute(r: NativeRequest,c: NativeWorkContext)=Map.empty[String,NativeJson]
    val worker=new NativeWorker(out,backend)
    worker.serve(new java.io.StringReader("""bad
{"id":"p","type":"interruption_protection","params":{"enabled":true}}
{"id":"i","type":"interrupt"}
{"id":"u","type":"missing"}
{"id":"s","type":"shutdown"}
{"id":"ignored","type":"ping"}
"""))
    val lines=out.toString(java.nio.charset.StandardCharsets.UTF_8).linesIterator.toVector
    assertEquals(lines.size,6)
    assert(lines(1).contains("invalid_params"))
    assert(lines(2).contains("\"revision\":1"))
    assert(lines(3).contains("\"blocked\":true"))
    assert(lines(4).contains("unsupported"))
    assert(lines(5).contains("\"id\":\"s\""))
  }
  test("reader responds during generation and cancellation allows recovery") {
    val lines=new LinkedBlockingQueue[String]()
    val output=new java.io.OutputStream:
      def write(b: Int): Unit=throw UnsupportedOperationException()
      override def write(b: Array[Byte],off: Int,len: Int): Unit=
        lines.add(new String(b,off,len,java.nio.charset.StandardCharsets.UTF_8).trim)
    val entered=new CountDownLatch(1)
    val backend=new NativeWorkerBackend:
      def supported=Set("generate")
      def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson]=
        if request.id=="a" then
          entered.countDown()
          while !context.cancelled() do Thread.sleep(1)
          context.checkCancelled()
        Map("done"->NativeJson.Bool(true))
    val worker=new NativeWorker(output,backend)
    def next()=Option(lines.poll(5,TimeUnit.SECONDS)).getOrElse(fail("missing response"))
    try
      assert(next().contains("hello"))
      worker.handle("""{"id":"a","type":"generate","owner_id":"job-a"}""")
      assert(entered.await(5,TimeUnit.SECONDS))
      worker.handle("""{"id":"p","type":"ping"}""")
      assert(next().contains("\"id\":\"p\""))
      worker.handle("""{"id":"i","type":"interrupt","params":{"target_id":"a","owner_id":"job-a"}}""")
      val responses=Vector(next(),next())
      assert(responses.exists(_.contains("interrupted")))
      assert(responses.exists(_.contains("\"affected\":true")))
      worker.handle("""{"id":"b","type":"generate"}""")
      assert(next().contains("\"done\":true"))
    finally worker.close()
  }
