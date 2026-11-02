package scalatorch.ffm

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class CudaDenseAttentionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def tensor(dims: Seq[Long],seed: Int): Tensor[Float]=
    val shape=Shape.from(dims)
    val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
  test("dense attention sees all keys for every query and matches CPU GQA reference") {
    gpu()
    CudaSession.withDevice() { session =>
      for (b,h,hk,t,s,d,dv) <- Vector((2,4,2,3,7,6,5),(1,2,1,9,3,4,7),(1,1,1,1,4,2,2)) do
        val q=tensor(Seq(b,h,t,d).map(_.toLong),1)
        val k=tensor(Seq(b,hk,s,d).map(_.toLong),2)
        val v=tensor(Seq(b,hk,s,dv).map(_.toLong),3)
        val expected=CpuFloatBackend.causalAttention(q,k,v,s-1).toFloatArray
        val dq=session.upload(q);val dk=session.upload(k);val dvv=session.upload(v)
        val out=session.scaledDotProductAttention(dq,dk,dvv)
        out.toCpu.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a.toDouble,e.toDouble,3e-6*math.max(1.0,math.abs(e.toDouble))))
        out.close();dvv.close();dk.close();dq.close()
      val q=session.fill(Shape.from(Seq(1L,1L,2L,1L)),0f)
      val k=session.fill(Shape.from(Seq(1L,1L,3L,1L)),0f)
      val v=session.upload(Tensor.fromFloats(Array(0f,3f,9f),k.shape))
      val out=session.scaledDotProductAttention(q,k,v)
      assertEquals(out.toCpu.toFloatArray.toVector,Vector(4f,4f))
      out.close();v.close();k.close();q.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("dense attention propagates future NaNs and validates workspace and empty queries") {
    gpu()
    CudaSession.withDevice() { session =>
      val q=session.fill(Shape.from(Seq(1L,1L,1L,1L)),1f)
      val k=session.upload(Tensor.fromFloats(Array(0f,Float.NaN),Shape.from(Seq(1L,1L,2L,1L))))
      val v=session.fill(k.shape,2f)
      val causal=session.causalAttention(q,k,v)
      val dense=session.scaledDotProductAttention(q,k,v)
      assertEquals(causal.toCpu.toFloatArray.toVector,Vector(2f))
      assert(dense.toCpu.toFloatArray.forall(_.isNaN))
      dense.close();causal.close()
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](session.scaledDotProductAttention(q,k,v,maxScoreElements=1))
      assertEquals(session.allocatedBytes,before)
      v.close();k.close();q.close()
      val empty=session.fill(Shape.from(Seq(0L,2L,3L,4L)),0f)
      val result=session.scaledDotProductAttention(empty,empty,empty)
      assertEquals(result.shape,empty.shape)
      assert(result.toCpu.toFloatArray.isEmpty)
      result.close();empty.close()
      assertEquals(session.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=20) { session =>
      val x=session.fill(Shape.from(Seq(1L,1L,2L,1L)),1f)
      intercept[IllegalArgumentException](session.scaledDotProductAttention(x,x,x))
      assertEquals(session.allocatedBytes,8L)
      x.close()
    }
  }
