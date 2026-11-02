package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendLossSuite extends munit.FunSuite:
  private def compare[T](backend: TrainingBackend[T]): Unit =
    val cases=Seq(
      (Array(1f,-2f,3f,4f,2f,-1f,0f,0f,0f),Seq(3L,3L),Array(2L,-7L,1L)),
      (Array(Float.MaxValue,Float.MaxValue,-Float.MaxValue),Seq(1L,3L),Array(0L)),
      (Array(0f,-110f,-120f),Seq(1L,3L),Array(0L)),
      (Array(1f,2f),Seq(1L,2L),Array(-7L)),
      (Array(-Float.MaxValue,Float.MaxValue),Seq(2L),Array(0L)),
      (Array.tabulate(4*513)(i => (math.sin(i*.17)*13).toFloat),Seq(4L,513L),Array(0L,512L,47L,-7L)),
      (Array.emptyFloatArray,Seq(0L,3L),Array.emptyLongArray))
    for (values,dims,labels) <- cases; reduction <- LossReduction.values do
      val source=Tensor.fromFloats(values,Shape.from(dims))
      val ids=Tensor.fromLongs(labels,Shape.from(dims.dropRight(1)))
      val reference=Variable(source,true)
      val expected=reference.crossEntropy(ids,reduction,-7L)
      val seed=Tensor.fromFloats(Array.fill(expected.data.shape.numel.toInt)(if values.contains(-110f) then Float.MaxValue else .75f),expected.data.shape)
      expected.backward(seed)
      val input=backend.upload(source);val upstream=backend.upload(seed)
      try Using.resource(new BackendTape(backend)) { tape =>
        val x=tape.variable(input)
        val loss=x.crossEntropy(ids,reduction,-7L)
        def near(actual: Array[Float],expected: Array[Float]): Unit =
          assertEquals(actual.length,expected.length)
          actual.zip(expected).foreach { (a,e) =>
            if e.isNaN then assert(a.isNaN)
            else if e.isInfinite then assertEquals(a,e)
            else assertEqualsDouble(a,e,math.max(1e-12,math.abs(e)*2e-6))
          }
        near(backend.download(loss.data).toFloatArray,expected.data.toFloatArray)
        loss.backward(upstream)
        near(backend.download(x.grad.get).toFloatArray,reference.grad.get.toFloatArray)
      } finally
        backend.release(upstream);backend.release(input)
  test("CPU backend cross entropy matches Variable for reductions and extremes") { compare(CpuFloatBackend) }
  test("CUDA backend cross entropy matches Variable for reductions and extremes") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      compare(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  private def differentiated[T](backend: TrainingBackend[T]): Unit =
    val source=Tensor.fromFloats(Array.tabulate(24)(i => math.sin(i*.71).toFloat),Shape.from(Seq(2L,3L,4L)))
    val ids=Tensor.fromLongs(Array(1L,2L,-3L,0L,3L,1L),Shape.from(Seq(3L,2L))).transpose(0,1)
    val input=backend.upload(source)
    try Using.resource(new BackendTape(backend)) { tape =>
      val x=tape.variable(input)
      val loss=x.crossEntropy(ids,LossReduction.Mean,-3)
      loss.backward()
      val first=backend.download(x.grad.get).toFloatArray
      source.toFloatArray.indices.foreach { i =>
        def evaluate(delta: Float): Double =
          val values=source.toFloatArray;values(i)+=delta
          Variable(Tensor.fromFloats(values,source.shape),false).crossEntropy(ids,LossReduction.Mean,-3).data.toFloatArray.head
        assertEqualsDouble(first(i),(evaluate(.002f)-evaluate(-.002f))/.004,3e-5)
      }
      loss.backward()
      backend.download(x.grad.get).toFloatArray.zip(first).foreach((a,b) => assertEqualsDouble(a,2*b,1e-7))
      assert(!tape.variable(input,false).crossEntropy(ids,ignoreIndex = -3).requiresGrad)
    } finally backend.release(input)
  test("CPU backend loss supports strided targets, finite differences and accumulation") { differentiated(CpuFloatBackend) }
  test("CUDA backend loss supports strided targets, finite differences and accumulation") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      differentiated(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("CUDA tied classifier trains with cross entropy and CPU update parity") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    val ids=Tensor.fromLongs(Array(0L,1L,2L,3L),Shape.from(Seq(4L)))
    var reference=Tensor.fromFloats(Array.tabulate(12)(i => math.sin(i*.7).toFloat*.3f),Shape.from(Seq(4L,3L)))
    CudaSession.withDevice() { session =>
      var weight=session.upload(reference)
      val baseline=session.allocatedBytes
      var first=0f;var last=0f
      for step <- 0 until 40 do
        val cpu=Variable(reference,true)
        val expected=cpu.embedding(ids).matmul(cpu.transpose(0,1)).crossEntropy(ids)
        expected.backward()
        val next=Using.resource(new BackendTape(session)) { tape =>
          val w=tape.variable(weight)
          val loss=w.embedding(ids).matmul(w.transpose(0,1)).crossEntropy(ids)
          loss.backward()
          val value=loss.data.toCpu.toFloatArray.head
          if step==0 then first=value
          last=value
          assertEqualsDouble(value,expected.data.toFloatArray.head,2e-6)
          w.grad.get.toCpu.toFloatArray.zip(cpu.grad.get.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-6))
          val delta=session.scale(w.grad.get,-.5f)
          try session.add(weight,delta) finally delta.close()
        }
        weight.close();weight=next
        reference=CpuFloatBackend.add(reference,CpuFloatBackend.scale(cpu.grad.get,-.5f))
        weight.toCpu.toFloatArray.zip(reference.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,3e-6))
        assertEquals(session.allocatedBytes,baseline)
      assert(last<first*.2,clues(first,last))
      println(s"CUDA cross entropy classifier loss: $first -> $last")
      weight.close();assertEquals(session.allocatedBytes,0L)
    }
  }
