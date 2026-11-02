package scalatorch.ffm

import scalatorch.*

class CudaFiniteSuite extends munit.FunSuite:
  test("resident finite validation handles IEEE extremes, empty tensors, lifetime and budget") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice(maxBytes=32) { session =>
      for values <- Seq(Array.emptyFloatArray,Array(0f,-0f,Float.MaxValue,-Float.MaxValue,Float.MinPositiveValue),
          Array(1f,Float.NaN),Array(Float.PositiveInfinity,0f),Array(0f,Float.NegativeInfinity)) do
        val x=session.upload(Tensor.fromFloats(values,Shape.from(Seq(values.length.toLong))))
        val baseline=session.allocatedBytes
        assertEquals(session.allFinite(x),values.forall(_.isFinite))
        assertEquals(session.allocatedBytes,baseline)
        x.close()
        intercept[IllegalStateException](session.allFinite(x))
      val full=session.fill(Shape.from(Seq(8L)),1f)
      intercept[IllegalArgumentException](session.allFinite(full))
      assertEquals(session.allocatedBytes,32L)
      full.close();assertEquals(session.allocatedBytes,0L)
    }
  }
