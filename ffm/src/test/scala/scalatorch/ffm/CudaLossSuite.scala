package scalatorch.ffm

import scalatorch.*

class CudaLossSuite extends munit.FunSuite:
  test("cross entropy validates targets and finite logits, including ignored rows") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice() { session =>
      val ids=Tensor.fromLongs(Array(-100L),Shape.from(Seq(1L)))
      for invalid <- Seq(Float.NaN,Float.PositiveInfinity,Float.NegativeInfinity) do
        val x=session.upload(Tensor.fromFloats(Array(1f,invalid),Shape.from(Seq(1L,2L))))
        val g=session.fill(Shape.from(Seq.empty),1f)
        val before=session.allocatedBytes
        intercept[IllegalArgumentException](session.crossEntropy(x,ids))
        intercept[IllegalArgumentException](session.crossEntropyBackward(x,ids,g))
        assertEquals(session.allocatedBytes,before)
        x.close();g.close()
      val x=session.fill(Shape.from(Seq(1L,2L)),1f)
      val before=session.allocatedBytes
      for id <- Seq(-1L,2L,Long.MaxValue) do
        intercept[IllegalArgumentException](session.crossEntropy(x,Tensor.fromLongs(Array(id),ids.shape)))
      intercept[IllegalArgumentException](session.crossEntropy(x,Tensor.fromLongs(Array(0L),Shape.from(Seq.empty))))
      val g=session.fill(Shape.from(Seq(1L)),1f)
      intercept[IllegalArgumentException](session.crossEntropyBackward(x,ids,g))
      g.close()
      assertEquals(session.allocatedBytes,before)
      x.close()
      intercept[IllegalStateException](session.crossEntropy(x,ids))
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("cross entropy preflights exact workspace and releases all temporary storage") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    val ids=Tensor.fromLongs(Array(0L,1L),Shape.from(Seq(2L)))
    for backward <- Seq(false,true); extra <- Seq(-1L,0L) do
      // Inputs 24+4; metadata8, Double stats48, status4, output4 or24.
      val budget=28L+8+48+4+(if backward then 24 else 4)+extra
      CudaSession.withDevice(maxBytes=budget) { session =>
        val x=session.fill(Shape.from(Seq(2L,3L)),0f)
        val seed=session.fill(Shape.from(Seq.empty),1f)
        def compute() = if backward then session.crossEntropyBackward(x,ids,seed) else session.crossEntropy(x,ids)
        if extra<0 then intercept[IllegalArgumentException](compute())
        else
          val output=compute()
          assertEquals(session.allocatedBytes,28L+output.shape.numel*4)
          assert(output.toCpu.toFloatArray.forall(_.isFinite))
          output.close()
        assertEquals(session.allocatedBytes,28L)
        x.close();seed.close();assertEquals(session.allocatedBytes,0L)
      }
  }
