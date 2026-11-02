package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendEmbeddingSuite extends munit.FunSuite:
  private val table=Tensor.fromFloats(Array.tabulate(12)(i => (i-5)/7f),Shape.from(Seq(4L,3L)))
  private def check[T](backend: TrainingBackend[T]): Unit =
    val weight=backend.upload(table)
    try
      val cases=Seq(
        Tensor.fromLongs(Array(2L,1L,2L,3L,1L,2L),Shape.from(Seq(2L,3L))).transpose(0,1),
        Tensor.fromLongs(Array(1L),Shape.from(Seq.empty)),
        Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(2L,0L))))
      for ids <- cases do
        val shape=Shape.from(ids.shape.dims :+ 3L)
        val seed=Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => (i%7-3)/5f),shape)
        val g=backend.upload(seed)
        try Using.resource(new BackendTape(backend)) { tape =>
          val w=tape.variable(weight)
          w.embedding(ids).backward(g)
          val reference=Variable(table,true)
          reference.embedding(ids).backward(seed)
          val actual=backend.download(w.grad.get).toFloatArray
          assertEquals(actual.toVector,reference.grad.get.toFloatArray.toVector)
          table.toFloatArray.indices.foreach { i =>
            def evaluate(delta: Float): Double =
              val values=table.toFloatArray;values(i)+=delta
              CpuFloatBackend.embedding(Tensor.fromFloats(values,table.shape),ids).toFloatArray.zip(seed.toFloatArray).map((v,s) => v.toDouble*s).sum
            assertEqualsDouble(actual(i),(evaluate(.001f)-evaluate(-.001f))/.002,5e-5)
          }
        } finally backend.release(g)
    finally backend.release(weight)
  test("CPU embedding gradients match autograd and finite differences") { check(CpuFloatBackend) }
  test("CUDA embedding gradients match autograd and finite differences") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      check(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("CUDA tied embedding and output projection train with CPU update parity") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    val ids=Tensor.fromLongs(Array(2L,1L,2L),Shape.from(Seq(3L)))
    var reference=table
    CudaSession.withDevice() { session =>
      var weight=session.upload(table)
      val baseline=session.allocatedBytes
      var first=0f;var last=0f
      for step <- 0 until 20 do
        val cpu=Variable(reference,true)
        val output=cpu.embedding(ids).matmul(cpu.transpose(0,1))
        val expectedLoss=(output*output).mean()
        expectedLoss.backward()
        val next=Using.resource(new BackendTape(session)) { tape =>
          val w=tape.variable(weight)
          val prediction=w.embedding(ids).matmul(w.transpose(0,1))
          val loss=(prediction*prediction).mean()
          loss.backward()
          val value=loss.data.toCpu.toFloatArray.head
          if step==0 then first=value
          last=value
          assertEqualsDouble(value,expectedLoss.data.toFloatArray.head,3e-6)
          w.grad.get.toCpu.toFloatArray.zip(cpu.grad.get.toFloatArray).foreach((g,e) => assertEqualsDouble(g,e,3e-6))
          val delta=session.scale(w.grad.get,-.1f)
          try session.add(weight,delta) finally delta.close()
        }
        weight.close();weight=next
        reference=CpuFloatBackend.add(reference,CpuFloatBackend.scale(cpu.grad.get,-.1f))
        weight.toCpu.toFloatArray.zip(reference.toFloatArray).foreach((v,e) => assertEqualsDouble(v,e,3e-6))
        assertEquals(session.allocatedBytes,baseline)
      assert(last<first*.5,clues(first,last))
      println(s"CUDA tied embedding training loss: $first -> $last")
      weight.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
