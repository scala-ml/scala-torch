package scalatorch.ffm

import scalatorch.*

class CudaBackwardSuite extends munit.FunSuite:
  test("rotary pullback validates offsets budgets and independently owned empty handles") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice(maxBytes=32) { session =>
      val x=session.fill(Shape.from(Seq(1L,1L,1L,4L)),1f)
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](session.ropeBackward(x))
      for (start,theta) <- Seq((-1,10000.0),(0,0.0),(0,Double.NaN),(0,Double.PositiveInfinity)) do
        intercept[IllegalArgumentException](session.ropeBackward(x,start,theta))
      assertEquals(session.allocatedBytes,before)
      x.close()
      val odd=session.fill(Shape.from(Seq(0L,1L,1L,3L)),0f)
      intercept[IllegalArgumentException](session.ropeBackward(odd))
      odd.close()
      val empty=session.fill(Shape.from(Seq(0L,2L,Long.MaxValue,4L)),0f)
      val result=session.ropeBackward(empty,Int.MaxValue)
      empty.close()
      assertEquals(result.toCpu.shape,Shape.from(Seq(0L,2L,Long.MaxValue,4L)))
      result.close()
      intercept[IllegalStateException](session.ropeBackward(result))
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("nonlinear pullbacks validate inputs and release inverse scratch on output failure") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice(maxBytes=64) { session =>
      val x=session.fill(Shape.from(Seq(2L,3L)),1f)
      val g=session.fill(x.shape,2f)
      val baseline=session.allocatedBytes
      // The 16-byte inverse scratch fits; the 12-byte weight gradient does not.
      intercept[IllegalArgumentException](session.rmsNormWeightBackward(x,g))
      assertEquals(session.allocatedBytes,baseline)
      for epsilon <- Seq(0.0,Double.NaN,Double.PositiveInfinity) do
        intercept[IllegalArgumentException](session.rmsNormWeightBackward(x,g,epsilon))
      intercept[IllegalArgumentException](session.rmsNormInputBackward(x,g,g))
      val scalar=session.fill(Shape.from(Seq.empty),1f)
      intercept[IllegalArgumentException](session.siluBackward(x,scalar))
      intercept[IllegalArgumentException](session.rmsNormWeightBackward(x,scalar))
      intercept[IllegalArgumentException](session.rmsNormWeightBackward(scalar,scalar))
      scalar.close();g.close();x.close()
      intercept[IllegalStateException](session.siluBackward(x,x))
      assertEquals(session.allocatedBytes,0L)
    }
  }
