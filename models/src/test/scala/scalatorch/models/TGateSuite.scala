package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.ffm.{CudaProbe,CudaSession}
import scalatorch.nn.{Linear,Parameter}

class TGateSuite extends munit.FunSuite:
  private val dummyLayer=
    val l=Linear(Parameter(Tensor.fromFloats(new Array[Float](16),Shape.from(Seq(4L,4L)))),None,None)
    DiffusionAttention(l,l,l,l,heads=1)

  test("TGateConfig validates gateStep bounds and state tracks step gating and CFG slots") {
    intercept[IllegalArgumentException](TGateConfig(0))
    intercept[IllegalArgumentException](TGateConfig(1001))
    val hasCuda=try CudaProbe.memoryInfo(0).totalBytes>0 catch case _: Throwable => false
    assume(hasCuda,"CUDA device required")
    CudaSession.withDevice(maxBytes=1L<<30) { session =>
      val state=new TGateState(TGateConfig(gateStep=3))
      try
        val sFull=Shape.from(Seq(2L,1L,2L))
        val slot0=state.withStep(0)(state.beginCall(1000f))
        assertEquals(slot0,0)
        assertEquals(state.shouldReuse(0,dummyLayer,sFull),None)
        assert(!state.shouldRecord(0,dummyLayer,None,Some(sFull)))
        // With DeepCache interval=3 and gateStep=3, step 0 is at gateStep-interval so it captures
        assert(state.shouldRecord(0,dummyLayer,Some(3),Some(sFull)))
        state.withStep(2) {
          state.withCall(800f) {
            val s=state.activeSlot.get
            assertEquals(s,0)
            assert(state.shouldRecord(s,dummyLayer,None,Some(sFull)))
            scala.util.Using.resource(session.uploadHalf(Tensor.fromFloats(Array(1f,1f,3f,3f),sFull))) { in =>
              val avg=session.tgateAvgHalf(in,hasPair=true)
              state.record(s,dummyLayer,avg)
            }
          }
        }
        state.withStep(3) {
          state.withCall(700f) {
            val cached=state.shouldReuse(state.activeSlot.get,dummyLayer,sFull)
            assert(cached.isDefined)
            // Single-branch shape (CFG=1 later in denoising) is also compatible with the averaged cache
            assert(state.shouldReuse(state.activeSlot.get,dummyLayer,Shape.from(Seq(1L,1L,2L))).isDefined)
            // Different spatial token count is incompatible, falls back cleanly, and enforces 2-step warmup
            assertEquals(state.shouldReuse(state.activeSlot.get,dummyLayer,Shape.from(Seq(2L,4L,2L))),None)
            // After spatial transition at step 3, effective gate moves to step 5 -> step 4 must also not reuse
            state.withStep(4) {
              assertEquals(state.shouldReuse(0,dummyLayer,Shape.from(Seq(2L,4L,2L))),None)
            }
            scala.util.Using.resource(session.uploadHalf(Tensor.fromFloats(Array(0.5f,0.5f,0.25f,0.25f),sFull))) { self =>
              val out=scala.util.Using.resource(session.tgateAddBroadcast(self,cached.get))(_.toCpu.toFloatArray.toSeq)
              assertEquals(out,Seq(2.5f,2.5f,2.25f,2.25f))
            }
          }
        }
      finally state.close()
    }
  }
