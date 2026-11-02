package scalatorch.models

import scalatorch.models.worker.NativeJobControl

class NativeJobControlSuite extends munit.FunSuite:
  test("concurrent publication and interrupt have exactly one winner") {
    import java.util.concurrent.{Callable,CountDownLatch,Executors,TimeUnit}
    val executor=Executors.newFixedThreadPool(2)
    try
      for _ <- 0 until 32 do
        val c=new NativeJobControl
        val job=c.admit("a",None);c.startNext()
        val start=new CountDownLatch(1)
        val publication=executor.submit(new Callable[Boolean] {
          def call(): Boolean={ start.await();c.beginPublication(job) }
        })
        val interrupt=executor.submit(new Callable[scalatorch.models.worker.NativeInterrupt] {
          def call()={ start.await();c.interrupt(None,None) }
        })
        start.countDown()
        val published=publication.get(5,TimeUnit.SECONDS)
        val decision=interrupt.get(5,TimeUnit.SECONDS)
        assert(published != decision.affected)
        assertEquals(job.cancelled,decision.affected)
        assertEquals(decision.tooLate,published)
        c.finish(job)
    finally executor.shutdownNow()
  }
  test("late cancellation cannot cross request or owner boundaries") {
    val c=new NativeJobControl
    val a=c.admit("a",Some("owner-a"))
    assertEquals(c.startNext(),Some(a))
    c.finish(a)
    val b=c.admit("b",Some("owner-b"));c.startNext()
    assert(!c.interrupt(Some("a"),Some("owner-a")).affected)
    assert(!c.interrupt(None,Some("owner-a")).affected)
    assert(!b.cancelled)
    assert(c.interrupt(Some("b"),Some("owner-b")).affected)
    assert(b.cancelled)
  }
  test("protection and publication admission are atomic") {
    val c=new NativeJobControl
    val job=c.admit("a",None);c.startNext()
    assertEquals(c.setProtection(true).revision,1L)
    assertEquals(c.setProtection(true).revision,1L)
    assert(c.interrupt(None,None).blocked)
    assert(!job.cancelled)
    c.setProtection(false)
    assert(c.beginPublication(job))
    val late=c.interrupt(None,None)
    assert(late.tooLate && !late.accepted && !late.affected && !late.blocked)
    c.finish(job)
  }
  test("queued cancellation is delivered and shutdown closes admission") {
    val c=new NativeJobControl
    val a=c.admit("a",None);val b=c.admit("b",None)
    intercept[IllegalArgumentException](c.admit("a",None))
    assert(c.interrupt(Some("b"),None).affected)
    assertEquals(c.startNext(),Some(a))
    intercept[IllegalStateException](c.startNext())
    c.finish(a)
    assertEquals(c.startNext(),Some(b))
    assert(b.cancelled)
    assert(!c.beginPublication(b))
    c.finish(b)
    val pending=c.admit("c",None)
    assertEquals(c.shutdown(),Vector(pending))
    assert(pending.cancelled)
    intercept[IllegalStateException](c.admit("d",None))
    assertEquals(c.startNext(),None)
  }
