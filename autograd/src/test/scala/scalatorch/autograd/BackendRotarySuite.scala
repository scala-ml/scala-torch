package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendRotarySuite extends munit.FunSuite:
  private val shape=Shape.from(Seq(2L,2L,3L,4L))
  private val input=Tensor.fromFloats(Array.tabulate(48)(i => (i%13-6)/7f),shape).transpose(0,1)
  private val seed=Tensor.fromFloats(Array.tabulate(48)(i => (i%7-3)/5f),shape).transpose(0,1)
  private def check[T](backend: TrainingBackend[T]): Unit =
    val x=backend.upload(input);val g=backend.upload(seed)
    try
      for (start,theta) <- Seq((0,10000.0),(7,10000.0),(Int.MaxValue,1.0),(1,1e-100)) do
        Using.resource(new BackendTape(backend)) { tape =>
          val a=tape.variable(x)
          a.rope(start,theta).backward(g)
          val reference=Variable(input,true)
          reference.rope(start,theta).backward(seed)
          val actual=backend.download(a.grad.get).toFloatArray
          actual.zip(reference.grad.get.toFloatArray).foreach((v,e) => assertEqualsDouble(v,e,2e-6))
          if start==7 then
            input.toFloatArray.indices.foreach { i =>
              def evaluate(delta: Float): Double =
                val values=input.toFloatArray;values(i)+=delta
                val result=CpuFloatBackend.rope(Tensor.fromFloats(values,shape),start,theta).toFloatArray
                result.zip(seed.toFloatArray).map((v,s) => v.toDouble*s).sum
              assertEqualsDouble(actual(i),(evaluate(.001f)-evaluate(-.001f))/.002,5e-5)
            }
        }
    finally { backend.release(g);backend.release(x) }
  test("CPU rotary gradients match autograd and finite differences") { check(CpuFloatBackend) }
  test("CUDA rotary gradients match autograd and finite differences") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      check(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("CUDA projection and rotary chain propagates input and weight gradients") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    val matrixShape=Shape.from(Seq(12L,4L))
    val weight=Tensor.fromFloats(Array.tabulate(16)(i => (i%5-2)/3f),Shape.from(Seq(4L,4L)))
    val cx=Variable(input,true);val cw=Variable(weight,true)
    cx.reshape(matrixShape).matmul(cw).reshape(shape).rope(11,100000).backward(seed)
    CudaSession.withDevice() { session =>
      val x=session.upload(input);val w=session.upload(weight);val g=session.upload(seed)
      Using.resource(new BackendTape(session)) { tape =>
        val a=tape.variable(x);val b=tape.variable(w)
        a.reshape(matrixShape).matmul(b).reshape(shape).rope(11,100000).backward(g)
        for (actual,expected) <- Seq(a.grad.get -> cx.grad.get,b.grad.get -> cw.grad.get) do
          actual.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((v,e) => assertEqualsDouble(v,e,3e-6))
      }
      g.close();w.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
