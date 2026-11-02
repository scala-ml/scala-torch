package scalatorch.ffm

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class CudaAdamWSuite extends munit.FunSuite:
  private val config=AdamWConfig(.03,.8,.95,1e-7,.02)
  test("CUDA AdamW matches CPU Double moments and resumes exactly after host state restoration") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice() { session =>
      val initial=Tensor.fromFloats(Array(1f,-2f,3f,0f,1e-30f,-1e-30f),Shape.from(Seq(2L,3L)))
      var reference=initial
      var cpuState: Option[AdamWState[Tensor[Float]]]=None
      var weight=session.upload(initial)
      var state: Option[AdamWState[CudaTensor]]=None
      try
        for i <- 0 until 10 do
          val gradient=Tensor.fromFloats(Array(i*.02f,-.3f,1e20f,-1e20f,1e-30f,Float.MinPositiveValue),initial.shape)
          val g=session.upload(gradient)
          try
            val expected=CpuFloatBackend.adamW(reference,gradient,cpuState,config)
            val actual=session.adamW(weight,g,state,config)
            assertEquals(actual.value.toCpu.toFloatArray.toVector,expected.value.toFloatArray.toVector)
            val snapshot=actual.state.snapshot();val ref=expected.state.snapshot()
            assertEquals(snapshot.step,(i+1).toLong)
            assertEquals(snapshot.first.toDoubleArray.toVector,ref.first.toDoubleArray.toVector)
            assertEquals(snapshot.second.toDoubleArray.toVector,ref.second.toDoubleArray.toVector)
            val restored=session.restoreAdamW(snapshot)
            val a=session.adamW(actual.value,g,Some(actual.state),config)
            val b=session.adamW(actual.value,g,Some(restored),config)
            try
              assertEquals(a.value.toCpu.toFloatArray.toVector,b.value.toCpu.toFloatArray.toVector)
              assertEquals(a.state.snapshot().first.toDoubleArray.toVector,b.state.snapshot().first.toDoubleArray.toVector)
              assertEquals(a.state.snapshot().second.toDoubleArray.toVector,b.state.snapshot().second.toDoubleArray.toVector)
            finally
              a.value.close();a.state.close();b.value.close();b.state.close();restored.close()
            weight.close();state.foreach(_.close());cpuState.foreach(_.close())
            weight=actual.value;state=Some(actual.state);reference=expected.value;cpuState=Some(expected.state)
          finally g.close()
          assertEquals(session.allocatedBytes,initial.shape.numel*20)
      finally
        weight.close();state.foreach(_.close());cpuState.foreach(_.close())
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("Adam state and rejected updates preserve ownership, values and exact byte budgets") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    for budget <- Seq(59L,60L) do
      CudaSession.withDevice(maxBytes=budget) { session =>
        val w=session.fill(Shape.from(Seq(2L)),1f);val g=session.fill(w.shape,.2f)
        val before=session.allocatedBytes
        if budget==59 then intercept[IllegalArgumentException](session.adamW(w,g,None,config))
        else
          val next=session.adamW(w,g,None,config)
          assertEquals(session.allocatedBytes,56L)
          assertEquals(next.state.step,1L)
          next.state.close();next.state.close()
          intercept[IllegalStateException](next.state.snapshot())
          next.value.close()
        assertEquals(session.allocatedBytes,before)
        w.close();g.close();assertEquals(session.allocatedBytes,0L)
      }
    CudaSession.withDevice() { session =>
      val shape=Shape.from(Seq(2L))
      val weight=session.fill(shape,1f);val gradient=session.fill(shape,.1f)
      val original=session.adamW(weight,gradient,None,config)
      val saved=original.state.snapshot()
      val baseline=session.allocatedBytes
      for invalid <- Seq(Float.NaN,Float.PositiveInfinity,Float.NegativeInfinity) do
        val bad=session.fill(shape,invalid)
        val before=session.allocatedBytes
        intercept[IllegalArgumentException](session.adamW(weight,bad,Some(original.state),config))
        intercept[IllegalArgumentException](session.adamW(bad,gradient,Some(original.state),config))
        assertEquals(session.allocatedBytes,before);bad.close()
      intercept[IllegalArgumentException](session.adamW(weight,gradient,Some(original.state),AdamWConfig(Double.MaxValue,weightDecay=Double.MaxValue)))
      val wrong=session.reshape(gradient,Shape.from(Seq(1L,2L)))
      intercept[IllegalArgumentException](session.adamW(weight,wrong,None,config));wrong.close()
      for moments <- Seq(saved.copy(step=0),saved.copy(first=Tensor.fromDoubles(Array(Double.NaN,0),shape)),
          saved.copy(second=Tensor.fromDoubles(Array(-1.0,0),shape))) do
        intercept[IllegalArgumentException](session.restoreAdamW(moments))
      val exhausted=session.restoreAdamW(saved.copy(step=Long.MaxValue))
      intercept[ArithmeticException](session.adamW(weight,gradient,Some(exhausted),config));exhausted.close()
      assertEquals(session.allocatedBytes,baseline)
      assertEquals(original.state.snapshot().first.toDoubleArray.toVector,saved.first.toDoubleArray.toVector)
      assertEquals(original.state.snapshot().second.toDoubleArray.toVector,saved.second.toDoubleArray.toVector)
      assertEquals(weight.toCpu.toFloatArray.toVector,Vector(1f,1f))
      original.value.close();original.state.close();weight.close();gradient.close()
      val empty=session.fill(Shape.from(Seq(0L,2L)),0f)
      val next=session.adamW(empty,empty,None,config)
      assertEquals(next.value.shape,empty.shape);assertEquals(next.state.snapshot().first.shape,empty.shape)
      next.value.close();next.state.close();empty.close();assertEquals(session.allocatedBytes,0L)
    }
  }
