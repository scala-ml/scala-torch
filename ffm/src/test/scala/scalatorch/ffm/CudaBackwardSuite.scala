package scalatorch.ffm

import scalatorch.*

class CudaBackwardSuite extends munit.FunSuite:
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
