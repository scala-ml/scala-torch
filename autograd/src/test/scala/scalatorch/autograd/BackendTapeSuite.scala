package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scala.util.Using
import scalatorch.ffm.CudaSession

class BackendTapeSuite extends munit.FunSuite:
  test("failed CUDA backward preserves prior gradients and releases scratch for retry") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice(maxBytes=256) { session =>
      Using.resource(new BackendTape(session)) { tape =>
        val source=session.upload(tensor(Array(2f,3f),2))
        val x=tape.variable(source)
        source.close() // The tape retains an independent handle.
        val loss=(x*x).sum()
        loss.backward()
        val baseline=session.allocatedBytes
        val filler=session.fill(Shape.from(Seq((256-baseline-20)/4)),0f)
        val before=session.allocatedBytes
        intercept[IllegalArgumentException](loss.backward())
        assertEquals(x.grad.get.toCpu.toFloatArray.toVector,Vector(4f,6f))
        assertEquals(session.allocatedBytes,before)
        filler.close()
        loss.backward()
        assertEquals(x.grad.get.toCpu.toFloatArray.toVector,Vector(8f,12f))
        assertEquals(session.allocatedBytes,baseline)
      }
      assertEquals(session.allocatedBytes,0L)
    }
  }
  private def tensor(values: Array[Float],dims: Long*)=Tensor.fromFloats(values,Shape.from(dims))
  private def checkGradients[T](backend: TrainingBackend[T]): Unit =
    val source=tensor(Array(.2f,-.4f,.6f,.8f,-.3f,.1f),2,3)
    val weights=tensor(Array(.5f,-.2f,.3f,.7f,-.6f,.4f),3,2)
    val x=backend.upload(source);val w=backend.upload(weights)
    try Using.resource(new BackendTape(backend)) { tape =>
      val a=tape.variable(x);val b=tape.variable(w)
      val transformed=a.matmul(b).transpose(0,1).reshape(Shape.from(Seq(4L)))
      val loss=(transformed*transformed).mean()
      loss.backward()
      val cpuA=Variable(source,true);val cpuB=Variable(weights,true)
      val cpuY=cpuA.matmul(cpuB).transpose(0,1).reshape(Shape.from(Seq(4L)))
      (cpuY*cpuY).mean().backward()
      for (actual,expected) <- Seq(a.grad.get -> cpuA.grad.get,b.grad.get -> cpuB.grad.get) do
        backend.download(actual).toFloatArray.zip(expected.toFloatArray).foreach((g,e) => assertEqualsDouble(g,e,2e-6))
      val epsilon=.001f
      val analytic=backend.download(a.grad.get).toFloatArray
      source.toFloatArray.indices.foreach { i =>
        def evaluate(delta: Float): Float =
          val changed=source.toFloatArray;changed(i)+=delta
          val y=CpuFloatBackend.matmul(tensor(changed,2,3),weights).toFloatArray
          y.map(v => v*v).sum/y.length
        val numerical=(evaluate(epsilon)-evaluate(-epsilon))/(2*epsilon)
        assertEqualsDouble(analytic(i),numerical,3e-5)
      }
    } finally { backend.release(w);backend.release(x) }

  test("CPU backend matrix and layout gradients match reference and finite differences") {
    checkGradients(CpuFloatBackend)
  }
  test("CUDA backend matrix and layout gradients match reference and finite differences") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      checkGradients(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("explicit seeds, frozen leaves, empty sums, closed and thread-confined tapes") {
    val tape=new BackendTape(CpuFloatBackend)
    val x=tape.variable(tensor(Array(2f,3f),2))
    val constant=tape.variable(tensor(Array(4f,5f),2),requiresGrad=false)
    val y=x*constant
    intercept[IllegalArgumentException](y.backward())
    intercept[IllegalArgumentException](y.backward(tensor(Array(1f),1)))
    y.backward(tensor(Array(2f,-1f),2))
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(8f,-5f))
    assertEquals(constant.grad,None)
    intercept[IllegalArgumentException](constant.sum().backward())
    val empty=tape.variable(tensor(Array.emptyFloatArray,0,2))
    empty.sum().backward()
    assertEquals(empty.grad.get.shape,Shape.from(Seq(0L,2L)))
    intercept[IllegalArgumentException](empty.mean())
    var failure: Throwable=null
    val worker=new Thread(() => { try x.zeroGrad() catch case e: Throwable => failure=e })
    worker.start();worker.join()
    assert(failure.isInstanceOf[IllegalStateException])
    tape.close();tape.close()
    intercept[IllegalStateException](x.grad)
    intercept[IllegalStateException](y.backward())
  }
  test("CUDA linear regression trains with resident gradients and updates") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      val x=session.upload(tensor(Array(1f,-1f,1f,0f,1f,1f,1f,2f),4,2))
      val target=session.upload(tensor(Array(-1f,1f,3f,5f),4,1))
      var weight=session.fill(Shape.from(Seq(2L,1L)),0f)
      val baseline=session.allocatedBytes
      var initial=0f;var last=0f
      for step <- 0 until 100 do
        val next=Using.resource(new BackendTape(session)) { tape =>
          val w=tape.variable(weight)
          val residual=tape.variable(x,false).matmul(w)-tape.variable(target,false)
          val loss=(residual*residual).mean()
          loss.backward()
          if step==0 then
            initial=loss.data.toCpu.toFloatArray.head
            assertEquals(w.grad.get.toCpu.toFloatArray.toVector,Vector(-4f,-7f))
          if step==99 then last=loss.data.toCpu.toFloatArray.head
          val delta=session.scale(w.grad.get,-.1f)
          try session.add(weight,delta) finally delta.close()
        }
        weight.close();weight=next
        assertEquals(session.allocatedBytes,baseline)
      val learned=weight.toCpu.toFloatArray
      assertEqualsDouble(learned(0),1,1e-4);assertEqualsDouble(learned(1),2,1e-4)
      assert(last<1e-8,clues(initial,last))
      println(s"CUDA resident linear regression loss: $initial -> $last weights=${learned.toVector}")
      weight.close();target.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("shared edges accumulate leaf gradients across repeated backward") {
    Using.resource(new BackendTape(CpuFloatBackend)) { tape =>
      val x=tape.variable(Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(3L))))
      val loss=(x*x+x).sum()
      loss.backward()
      assertEquals(x.grad.get.toFloatArray.toVector,Vector(3f,5f,7f))
      loss.backward()
      assertEquals(x.grad.get.toFloatArray.toVector,Vector(6f,10f,14f))
      x.zeroGrad()
      assertEquals(x.grad,None)
    }
  }
