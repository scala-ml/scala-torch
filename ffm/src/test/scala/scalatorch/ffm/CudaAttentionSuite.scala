package scalatorch.ffm

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class CudaAttentionSuite extends munit.FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def tensor(dims: Seq[Long], seed: Int) =
    val shape=Shape.from(dims)
    Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => ((i*13+seed)%23-11).toFloat/7),shape)
  private def compare(actual: Array[Float], expected: Array[Float]): Unit =
    assertEquals(actual.length,expected.length)
    actual.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,3e-6*math.max(1.0,math.abs(e.toDouble))))

  test("CUDA split-half RoPE matches CPU for batches heads positions and extreme theta") {
    gpu()
    CudaSession.withDevice() { session =>
      val input=tensor(Seq(2,3,7,8),3)
      for (start,theta) <- Seq((0,10000.0),(37,100000.0),(Int.MaxValue,1.0), (1,1e-100)) do
        val output=session.rope(session.upload(input),start,theta)
        compare(output.toCpu.toFloatArray,CpuFloatBackend.rope(input,start,theta).toFloatArray)
    }
  }
  test("CUDA causal GQA matches CPU with offsets unequal value widths and masked future keys") {
    gpu()
    CudaSession.withDevice() { session =>
      for (b,h,kv,t,s,d,dv,start) <- Seq((2,4,2,3,5,6,4,0),(1,3,1,2,5,4,7,3),(1,1,1,1,1,2,2,Int.MaxValue)) do
        val q=tensor(Seq(b,h,t,d).map(_.toLong),1)
        val k=tensor(Seq(b,kv,s,d).map(_.toLong),2)
        val v=tensor(Seq(b,kv,s,dv).map(_.toLong),3)
        val output=session.causalAttention(session.upload(q),session.upload(k),session.upload(v),start)
        compare(output.toCpu.toFloatArray,CpuFloatBackend.causalAttention(q,k,v,start).toFloatArray)
      val q=Tensor.fromFloats(Array(1e30f,1e30f),Shape.from(Seq(1L,1L,1L,2L)))
      val k=Tensor.fromFloats(Array(1e30f,1e30f,-1e30f,-1e30f),Shape.from(Seq(1L,1L,2L,2L)))
      val v=Tensor.fromFloats(Array(3f,4f,100f,200f),Shape.from(Seq(1L,1L,2L,2L)))
      compare(session.causalAttention(session.upload(q),session.upload(k),session.upload(v),1).toCpu.toFloatArray,Array(3f,4f))
      val masked=Tensor.fromFloats(Array(0f,0f,Float.NaN,Float.NaN),k.shape)
      val dq=session.upload(q);val dk=session.upload(masked);val dv=session.upload(v)
      compare(session.causalAttention(dq,dk,dv).toCpu.toFloatArray,Array(3f,4f))
      assert(session.causalAttention(dq,dk,dv,1).toCpu.toFloatArray.forall(_.isNaN))
    }
  }
  test("attention temporaries respect budgets and empty outputs preserve shape") {
    gpu()
    CudaSession.withDevice(maxBytes=31) { session =>
      val x=session.upload(tensor(Seq(1,1,1,2),0))
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](session.rope(x))
      intercept[IllegalArgumentException](session.causalAttention(x,x,x,maxScoreElements=0))
      assertEquals(session.allocatedBytes,before)
      val empty=session.upload(tensor(Seq(0,2,3,4),1))
      val out=session.causalAttention(empty,empty,empty)
      assertEquals(out.shape,empty.shape)
      assert(out.toCpu.toFloatArray.isEmpty)
      out.close();empty.close()
      val bigger=session.upload(tensor(Seq(1,1,2,2),2))
      val occupied=session.allocatedBytes
      intercept[IllegalArgumentException](session.causalAttention(bigger,bigger,bigger))
      assertEquals(session.allocatedBytes,occupied)
    }
  }

  test("device KV capacity strides match full GQA and reject gaps") {
    gpu()
    CudaSession.withDevice() { session =>
      val q=tensor(Seq(2,4,3,4),1);val k=tensor(Seq(2,2,3,4),2);val v=tensor(Seq(2,2,3,6),3)
      val before=session.allocatedBytes
      val cache=session.attentionCache(2,2,7,4,6)
      val cacheBytes=session.allocatedBytes
      for (start,end) <- Seq((0,2),(2,3)) do
        val dq=session.upload(q.slice(2,start,end));val dk=session.upload(k.slice(2,start,end));val dv=session.upload(v.slice(2,start,end))
        val result=cache.attend(dq,dk,dv,start)
        compare(result.toCpu.toFloatArray,CpuFloatBackend.causalAttention(q.slice(2,start,end),k.slice(2,0,end),v.slice(2,0,end),start).toFloatArray)
        result.close();dq.close();dk.close();dv.close()
        assertEquals(session.allocatedBytes,cacheBytes)
      val one=session.upload(q.slice(2,0,1));val key=session.upload(k.slice(2,0,1));val value=session.upload(v.slice(2,0,1))
      intercept[IllegalArgumentException](cache.attend(one,key,value,4))
      val repeated=cache.attend(one,key,value,0)
      compare(repeated.toCpu.toFloatArray,CpuFloatBackend.causalAttention(q.slice(2,0,1),k.slice(2,0,1),v.slice(2,0,1)).toFloatArray)
      repeated.close();one.close();key.close();value.close();cache.close()
      assertEquals(session.allocatedBytes,before)
    }
    CudaSession.withDevice(maxBytes=32) { session =>
      val cache=session.attentionCache(1,1,2,1,1)
      val q=session.upload(Tensor.fromFloats(Array(7f),Shape.from(Seq(1L,1L,1L,1L))))
      val filler=session.upload(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L))))
      // Append completes and scratch fits, but output allocation fails.
      intercept[IllegalArgumentException](cache.attend(q,q,q,0))
      assertEquals(session.allocatedBytes,24L)
      filler.close()
      val retried=cache.attend(q,q,q,0)
      compare(retried.toCpu.toFloatArray,Array(7f))
      retried.close();q.close();cache.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
