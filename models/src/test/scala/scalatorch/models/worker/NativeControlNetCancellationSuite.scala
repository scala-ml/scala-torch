package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.util.concurrent.{CountDownLatch,LinkedBlockingQueue,TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.duration.*
import scala.util.Using
import scalatorch.models.PretrainedControlNet

/** Opt-in real checkpoint: cancellation crosses worker protocol into the restricted ZIP reader. */
class NativeControlNetCancellationSuite extends munit.FunSuite:
  override val munitTimeout: Duration=5.minutes
  test("worker interrupts real PyTorch ControlNet reading and accepts a later load") {
    val properties=new java.util.Properties()
    val config=Vector(Path.of(".superpowers/sd15-single-inpaint-test.properties"),
      Path.of("../.superpowers/sd15-single-inpaint-test.properties")).find(Files.isRegularFile(_))
    config.foreach(path => Using.resource(Files.newInputStream(path))(properties.load))
    val checkpoint=sys.env.get("SCALATORCH_SD15_CONTROLNET").orElse(Option(properties.getProperty("SCALATORCH_SD15_CONTROLNET")))
    assume(checkpoint.exists(_.toLowerCase(java.util.Locale.ROOT).endsWith(".pth")),"real PyTorch ControlNet required")
    val entered=new CountDownLatch(1)
    val release=new CountDownLatch(1)
    val checks=new AtomicInteger()
    val lines=new LinkedBlockingQueue[String]()
    val output=new java.io.OutputStream:
      def write(b: Int): Unit=throw UnsupportedOperationException()
      override def write(b: Array[Byte],off: Int,len: Int): Unit =
        lines.add(new String(b,off,len,java.nio.charset.StandardCharsets.UTF_8).trim)
    val backend=new NativeWorkerBackend:
      def supported=Set("load_control")
      def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson] =
        val cancelled=() =>
          if request.id=="first" && checks.incrementAndGet()==900 then
            entered.countDown()
            require(release.await(10,TimeUnit.SECONDS),"test did not release reader")
          context.cancelled()
        PretrainedControlNet.load(Path.of(checkpoint.get),cancelled=cancelled)
        Map("loaded"->NativeJson.Bool(true))
    val worker=new NativeWorker(output,backend)
    def next()=Option(lines.poll(90,TimeUnit.SECONDS)).getOrElse(fail("missing worker response"))
    try
      assert(next().contains("hello"))
      worker.handle("""{"id":"first","type":"load_control","owner_id":"cancel-test"}""")
      assert(entered.await(30,TimeUnit.SECONDS),"reader did not reach the cancellation point")
      worker.handle("""{"id":"stop","type":"interrupt","params":{"target_id":"first","owner_id":"cancel-test"}}""")
      release.countDown()
      val replies=Vector(next(),next())
      assert(replies.exists(_.contains("interrupted")))
      assert(replies.exists(_.contains("\"affected\":true")))
      worker.handle("""{"id":"again","type":"load_control"}""")
      assert(next().contains("\"loaded\":true"))
    finally
      release.countDown()
      worker.close()
  }
