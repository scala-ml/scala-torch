package scalatorch.nn

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class BackendAdamWSuite extends munit.FunSuite:
  test("backend CPU AdamW matches existing optimizer weights and Double moment slots") {
    val initial=Tensor.fromFloats(Array(1f,-2f,3f,0f),Shape.from(Seq(2L,2L)))
    val p=Parameter(initial)
    val optimizer=new AdamW(Seq(p),.03,beta1=.8,beta2=.95,epsilon=1e-7,weightDecay=.02)
    val config=AdamWConfig(.03,.8,.95,1e-7,.02)
    var weight=initial
    var state: Option[AdamWState[Tensor[Float]]]=None
    try
      for i <- 0 until 8 do
        val gradient=Tensor.fromFloats(Array(i*.02f,-.3f,1e-30f,0f),initial.shape)
        p.zeroGrad();p.value.backward(gradient);optimizer.step()
        val next=CpuFloatBackend.adamW(weight,gradient,state,config)
        state.foreach(_.close());state=Some(next.state);weight=next.value
        assertEquals(weight.toFloatArray.toVector,p.data.toFloatArray.toVector)
        val actual=next.state.snapshot()
        val expected=optimizer.snapshot.head.get
        assertEquals(actual.step,expected.step)
        assertEquals(actual.first.toDoubleArray.toVector,expected.first)
        assertEquals(actual.second.toDoubleArray.toVector,expected.second)
    finally state.foreach(_.close())
  }
