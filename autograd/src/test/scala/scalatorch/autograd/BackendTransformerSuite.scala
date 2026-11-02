package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendTransformerSuite extends munit.FunSuite:
  private def tensor(a: Array[Float],dims: Long*)=Tensor.fromFloats(a,Shape.from(dims))
  private def check[T](backend: TrainingBackend[T]): Unit =
    val input=tensor(Array(.2f,-.4f,.6f,.8f,-.3f,.1f),2,3)
    val weight=tensor(Array(.7f,1.2f,-.5f),3)
    val upstream=tensor(Array(.1f,-.2f,.3f,.4f,.5f,-.6f),2,3)
    val x=backend.upload(input);val w=backend.upload(weight);val g=backend.upload(upstream)
    try Using.resource(new BackendTape(backend)) { tape =>
      val a=tape.variable(x);val b=tape.variable(w)
      a.rmsNorm(b,.03).silu().backward(g)
      val ca=Variable(input,true);val cb=Variable(weight,true)
      ca.rmsNorm(cb,.03).silu().backward(upstream)
      for (actual,expected) <- Seq(a.grad.get -> ca.grad.get,b.grad.get -> cb.grad.get) do
        backend.download(actual).toFloatArray.zip(expected.toFloatArray).foreach((v,e) => assertEqualsDouble(v,e,2e-6))
      for (source,gradient,isInput) <- Seq((input,backend.download(a.grad.get),true),(weight,backend.download(b.grad.get),false)) do
        val epsilon=.001f
        source.toFloatArray.indices.foreach { i =>
          def evaluate(delta: Float): Double =
            val changed=source.toFloatArray;changed(i)+=delta
            val candidate=Tensor.fromFloats(changed,source.shape)
            val y=CpuFloatBackend.silu(CpuFloatBackend.rmsNorm(if isInput then candidate else input,if isInput then weight else candidate,.03))
            y.toFloatArray.zip(upstream.toFloatArray).map((v,g) => v.toDouble*g).sum
          assertEqualsDouble(gradient.toFloatArray(i),(evaluate(epsilon)-evaluate(-epsilon))/(2*epsilon),5e-5)
        }
    } finally { backend.release(g);backend.release(w);backend.release(x) }
  private def boundaries[T](backend: TrainingBackend[T]): Unit =
    def compare(a: T,e: Tensor[Float]): Unit =
      backend.download(a).toFloatArray.zip(e.toFloatArray).foreach { (v,w) =>
        if w.isNaN then assert(v.isNaN)
        else if w.isInfinite then assertEquals(v,w)
        else assertEqualsDouble(v,w,2e-6*math.max(1.0,math.abs(w.toDouble)))
      }
    val values=tensor(Array(Float.NegativeInfinity,Float.PositiveInfinity,Float.NaN,-1000f,-100f,-1.278f,-0f,0f,Float.MaxValue,1e-30f),10)
    val seed=tensor(Array(Float.NaN,2f,1f,Float.NaN,1f,2f,3f,4f,5f,6f),10)
    val cpu=Variable(values,true);cpu.silu().backward(seed)
    val x=backend.upload(values);val g=backend.upload(seed)
    try
      val out=backend.siluBackward(x,g)
      try
        compare(out,cpu.grad.get)
        assertEquals(backend.download(out).toFloatArray.take(1).toVector,Vector(0f))
      finally backend.release(out)
    finally { backend.release(g);backend.release(x) }
    for source <- Seq(tensor(Array(0f,0f,0f,Float.MaxValue,-Float.MaxValue,1e-30f),2,3),tensor(Array.emptyFloatArray,0,3)) do
      val scale=tensor(Array(1f,2f,-3f),3)
      val upstream=Tensor.fromFloats(Array.fill(source.shape.numel.toInt)(2f),source.shape)
      val a=backend.upload(source);val b=backend.upload(scale);val s=backend.upload(upstream)
      try
        for trainInput <- Seq(true,false) do
          Using.resource(new BackendTape(backend)) { tape =>
            val p=tape.variable(a,trainInput);val q=tape.variable(b,!trainInput)
            p.rmsNorm(q).backward(s)
            val cp=Variable(source,trainInput);val cq=Variable(scale,!trainInput)
            cp.rmsNorm(cq).backward(upstream)
            if trainInput then
              compare(p.grad.get,cp.grad.get);assertEquals(q.grad,None)
            else
              compare(q.grad.get,cq.grad.get);assertEquals(p.grad,None)
          }
      finally { backend.release(s);backend.release(b);backend.release(a) }
  test("CPU transformer pullbacks match existing autograd") { check(CpuFloatBackend) }
  test("CUDA transformer pullbacks match existing autograd") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      check(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("CPU nonlinear pullbacks preserve frozen empty and extreme cases") { boundaries(CpuFloatBackend) }
  test("CUDA nonlinear pullbacks preserve frozen empty and extreme cases") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      boundaries(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("resident CUDA gated MLP training matches CPU gradients and updates") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    val random=new java.util.Random(19)
    def matrix(rows: Int,cols: Int)=tensor(Array.fill(rows*cols)((random.nextGaussian()*.4).toFloat),rows,cols)
    val input=matrix(3,4)
    var cpuWeights=Vector(tensor(Array.fill(4)(1f),4),matrix(4,6),matrix(4,6),matrix(6,4))
    CudaSession.withDevice() { session =>
      val x=session.upload(input)
      var weights=cpuWeights.map(session.upload)
      val baseline=session.allocatedBytes
      var first=0f;var last=0f
      for step <- 0 until 20 do
        val cpu=cpuWeights.map(Variable(_,true))
        val normalized=Variable(input,false).rmsNorm(cpu(0))
        val result=(normalized.matmul(cpu(1)).silu()*normalized.matmul(cpu(2))).matmul(cpu(3))
        val expectedLoss=(result*result).mean()
        expectedLoss.backward()
        val next=Using.resource(new BackendTape(session)) { tape =>
          val parameters=weights.map(tape.variable(_))
          val n=tape.variable(x,false).rmsNorm(parameters(0))
          val y=(n.matmul(parameters(1)).silu()*n.matmul(parameters(2))).matmul(parameters(3))
          val loss=(y*y).mean()
          loss.backward()
          val value=loss.data.toCpu.toFloatArray.head
          if step==0 then first=value
          last=value
          assertEqualsDouble(value,expectedLoss.data.toFloatArray.head,2e-6)
          parameters.zip(cpu).foreach { (actual,reference) =>
            actual.grad.get.toCpu.toFloatArray.zip(reference.grad.get.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,3e-6))
          }
          parameters.zip(weights).map { (p,w) =>
            val delta=session.scale(p.grad.get,-.05f)
            try session.add(w,delta) finally delta.close()
          }
        }
        weights.foreach(_.close());weights=next
        cpuWeights=cpuWeights.zip(cpu).map((w,p) => CpuFloatBackend.add(w,CpuFloatBackend.scale(p.grad.get,-.05f)))
        weights.zip(cpuWeights).foreach { (a,e) =>
          a.toCpu.toFloatArray.zip(e.toFloatArray).foreach((v,w) => assertEqualsDouble(v,w,3e-6))
        }
        assertEquals(session.allocatedBytes,baseline)
      assert(last<first,clues(first,last))
      println(s"CUDA gated MLP training loss: $first -> $last")
      weights.foreach(_.close());x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
