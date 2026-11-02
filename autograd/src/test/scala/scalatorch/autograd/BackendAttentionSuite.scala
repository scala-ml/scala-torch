package scalatorch.autograd

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend
import scalatorch.ffm.CudaSession
import scala.util.Using

class BackendAttentionSuite extends munit.FunSuite:
  private def tensor(shape: Shape,seed: Int): Tensor[Float] =
    Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => ((i*7+seed)%17-8)/9f),shape)
  private def compare(a: Array[Float],b: Array[Float],tol: Double = 3e-6): Unit =
    assertEquals(a.length,b.length)
    a.zip(b).foreach((x,y) => assertEqualsDouble(x,y,tol*math.max(1,math.abs(y.toDouble))))
  private def check[T](backend: TrainingBackend[T]): Unit =
    for start <- Seq(0,2,Int.MaxValue) do
      val q=tensor(Shape.from(Seq(2L,4L,2L,4L)),1)
      val k=tensor(Shape.from(Seq(2L,2L,3L,4L)),3)
      val v=tensor(Shape.from(Seq(2L,2L,3L,2L)),5)
      val seed=tensor(Shape.from(Seq(2L,4L,2L,2L)),7)
      val source=Vector(q,k,v);val handles=source.map(backend.upload);val g=backend.upload(seed)
      try Using.resource(new BackendTape(backend)) { tape =>
        val p=handles.map(tape.variable(_));val cpu=source.map(Variable(_,true))
        val result=p(0).causalAttention(p(1),p(2),start)
        result.backward(g)
        cpu(0).causalAttention(cpu(1),cpu(2),start).backward(seed)
        p.zip(cpu).foreach((a,b) => compare(backend.download(a.grad.get).toFloatArray,b.grad.get.toFloatArray))
        if start==2 then
          for target <- source.indices; index <- source(target).toFloatArray.indices do
            def evaluate(delta: Float): Double =
              val changed=source(target).toFloatArray;changed(index)+=delta
              val inputs=source.updated(target,Tensor.fromFloats(changed,source(target).shape))
              CpuFloatBackend.causalAttention(inputs(0),inputs(1),inputs(2),start).toFloatArray.zip(seed.toFloatArray).map((a,g) => a.toDouble*g).sum
            assertEqualsDouble(backend.download(p(target).grad.get).toFloatArray(index),
              (evaluate(.001f)-evaluate(-.001f))/.002,1e-4)
        result.backward(g)
        p.zip(cpu).foreach((a,b) => compare(backend.download(a.grad.get).toFloatArray,b.grad.get.toFloatArray.map(_*2f)))
      } finally { backend.release(g);handles.reverseIterator.foreach(backend.release) }
  test("CPU attention pullbacks match GQA reference offsets and repeated backward") { check(CpuFloatBackend) }
  test("CUDA attention pullbacks match GQA reference offsets and repeated backward") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      check(session)
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("joint pullback runs once for tied edges and once again for a new seed") {
    class Counting extends TrainingBackend[Tensor[Float]]:
      export CpuFloatBackend.{crossEntropy,crossEntropyBackward,upload,download,shape,release,reshape,transpose,embedding,matmul,add,multiply,scale,fill,sum,expandScalar,
        silu,rmsNorm,rope,causalAttention,attentionCache,embeddingBackward,ropeBackward,siluBackward,rmsNormInputBackward,rmsNormWeightBackward}
      var calls=0
      def causalAttentionBackward(q: Tensor[Float],k: Tensor[Float],v: Tensor[Float],g: Tensor[Float],start: Int = 0,
          maxScores: Long = 16777216L): (Tensor[Float],Tensor[Float],Tensor[Float]) =
        calls+=1;CpuFloatBackend.causalAttentionBackward(q,k,v,g,start,maxScores)
    val backend=new Counting
    val source=tensor(Shape.from(Seq(1L,2L,2L,2L)),4)
    val cpu=Variable(source,true);cpu.causalAttention(cpu,cpu).sum().backward()
    Using.resource(new BackendTape(backend)) { tape =>
      val x=tape.variable(source)
      val loss=x.causalAttention(x,x).sum()
      loss.backward()
      assertEquals(backend.calls,1)
      compare(x.grad.get.toFloatArray,cpu.grad.get.toFloatArray)
      loss.backward(CpuFloatBackend.fill(Shape.from(Seq.empty),2f))
      assertEquals(backend.calls,2)
      compare(x.grad.get.toFloatArray,cpu.grad.get.toFloatArray.map(_*3f))
    }
  }
  test("CUDA attention masks future NaNs and handles empty frozen queries") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    CudaSession.withDevice() { session =>
      val q=session.upload(Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(1L,2L,1L,2L))))
      val k=session.upload(Tensor.fromFloats(Array(.2f,.3f,Float.NaN,Float.NaN),Shape.from(Seq(1L,1L,2L,2L))))
      val v=session.upload(Tensor.fromFloats(Array(.4f,.5f,Float.NaN,Float.NaN),k.shape))
      val g=session.upload(Tensor.fromFloats(Array(1f,2f,3f,4f),q.shape))
      Using.resource(new BackendTape(session)) { tape =>
        val a=tape.variable(q);val b=tape.variable(k);val c=tape.variable(v)
        a.causalAttention(b,c).backward(g)
        assertEquals(a.grad.get.toCpu.toFloatArray.toVector,Vector.fill(4)(0f))
        assertEquals(b.grad.get.toCpu.toFloatArray.toVector,Vector.fill(4)(0f))
        assertEquals(c.grad.get.toCpu.toFloatArray.toVector,Vector(4f,6f,0f,0f))
      }
      val empty=session.fill(Shape.from(Seq(1L,2L,0L,2L)),0f)
      Using.resource(new BackendTape(session)) { tape =>
        val a=tape.variable(empty,false);val b=tape.variable(k);val c=tape.variable(v)
        a.causalAttention(b,c).backward(empty)
        assertEquals(a.grad,None)
        assert(b.grad.get.toCpu.toFloatArray.forall(_==0f))
        assert(c.grad.get.toCpu.toFloatArray.forall(_==0f))
      }
      empty.close();g.close();v.close();k.close();q.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("resident rotary GQA training matches CPU gradients and updates") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA autograd gate disabled")
    var reference=Vector(tensor(Shape.from(Seq(1L,2L,3L,4L)),1),tensor(Shape.from(Seq(1L,1L,3L,4L)),2),
      tensor(Shape.from(Seq(1L,1L,3L,2L)),3))
    CudaSession.withDevice() { session =>
      var weights=reference.map(session.upload)
      val baseline=session.allocatedBytes
      var first=0f;var last=0f
      for step <- 0 until 20 do
        val cpu=reference.map(Variable(_,true))
        val output=cpu(0).rope(5).causalAttention(cpu(1).rope(5),cpu(2))
        val expected=(output*output).mean();expected.backward()
        val next=Using.resource(new BackendTape(session)) { tape =>
          val p=weights.map(tape.variable(_))
          val out=p(0).rope(5).causalAttention(p(1).rope(5),p(2))
          val loss=(out*out).mean();loss.backward()
          val current=loss.data.toCpu.toFloatArray.head
          if step==0 then first=current
          last=current
          assertEqualsDouble(current,expected.data.toFloatArray.head,3e-6)
          p.zip(cpu).foreach((a,b) => compare(a.grad.get.toCpu.toFloatArray,b.grad.get.toFloatArray))
          p.zip(weights).map { (a,w) =>
            val delta=session.scale(a.grad.get,-.1f)
            try session.add(w,delta) finally delta.close()
          }
        }
        weights.foreach(_.close());weights=next
        reference=reference.zip(cpu).map((w,p) => CpuFloatBackend.add(w,CpuFloatBackend.scale(p.grad.get,-.1f)))
        weights.zip(reference).foreach((a,b) => compare(a.toCpu.toFloatArray,b.toFloatArray))
        assertEquals(session.allocatedBytes,baseline)
      assert(last<first*.8,clues(first,last))
      println(s"CUDA rotary GQA training loss: $first -> $last")
      weights.foreach(_.close())
      assertEquals(session.allocatedBytes,0L)
    }
  }
