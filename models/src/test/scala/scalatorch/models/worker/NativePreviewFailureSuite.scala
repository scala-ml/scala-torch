package scalatorch.models.worker

class NativePreviewFailureSuite extends munit.FunSuite:
  test("a preview conversion failure does not abort work but cancellation propagates") {
    val events=scala.collection.mutable.ArrayBuffer.empty[String]
    var cancelled=false
    val context=new NativeWorkContext(()=>cancelled,()=>true,(kind,_)=>events+=kind)
    NativePreview.emitSafely(context,1) { throw new IllegalArgumentException("bad preview") }
    assert(!events.contains("preview"))
    assert(events.contains("log"))
    intercept[java.util.concurrent.CancellationException] {
      NativePreview.emitSafely(context,1) { throw new java.util.concurrent.CancellationException("cancelled") }
    }
    cancelled=true
    intercept[java.util.concurrent.CancellationException] {
      NativePreview.emitSafely(context,1) { throw new IllegalArgumentException("failed after cancellation") }
    }
  }
