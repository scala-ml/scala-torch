package scalatorch.ffm

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class CudaDenseAttentionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def tensor(dims: Seq[Long],seed: Int): Tensor[Float]=
    val shape=Shape.from(dims)
    val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
  private def denseReference(query: Tensor[Float],key: Tensor[Float],value: Tensor[Float]): Array[Float]=
    val qd=query.shape.dims.map(_.toInt);val kd=key.shape.dims.map(_.toInt);val vd=value.shape.dims.map(_.toInt)
    val q=query.toFloatArray;val k=key.toFloatArray;val v=value.toFloatArray
    val out=new Array[Float](qd(0)*qd(1)*qd(2)*vd(3))
    val groups=qd(1)/kd(1);val scale=1.0/math.sqrt(qd(3).toDouble)
    for b<-0 until qd(0);h<-0 until qd(1);t<-0 until qd(2) do
      val kh=h/groups;val qbase=((b*qd(1)+h)*qd(2)+t)*qd(3)
      val scores=Array.tabulate(kd(2)) { s =>
        var dot=0.0;var d=0
        while d<qd(3) do
          dot+=q(qbase+d).toDouble*k(((b*kd(1)+kh)*kd(2)+s)*kd(3)+d).toDouble
          d+=1
        dot*scale
      }
      val maximum=scores.max
      val weights=scores.map(x=>math.exp(x-maximum));val total=weights.sum
      for c<-0 until vd(3) do
        var sum=0.0
        for s<-0 until kd(2) do sum+=(weights(s)/total)*v(((b*kd(1)+kh)*vd(2)+s)*vd(3)+c).toDouble
        out(((b*qd(1)+h)*qd(2)+t)*vd(3)+c)=sum.toFloat
    out
  test("dense attention sees all keys for every query and matches CPU GQA reference") {
    gpu()
    val profileEnabled=sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1")
    val diagnostics=new java.io.ByteArrayOutputStream()
    val previous=System.err
    val profileStream=new java.io.PrintStream(diagnostics,true,java.nio.charset.StandardCharsets.UTF_8)
    try
      if profileEnabled then System.setErr(profileStream)
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
        val boundaryQ=session.fill(Shape.from(Seq(1L,1L,1L,1L)),0f)
        val boundaryK=session.fill(Shape.from(Seq(1L,1L,4096L,1L)),0f)
        val boundaryV=session.upload(Tensor.fromFloats(Array.tabulate(4096)(_.toFloat),boundaryK.shape))
        val boundaryOut=session.scaledDotProductAttention(boundaryQ,boundaryK,boundaryV,maxScoreElements=4096)
        assertEqualsDouble(boundaryOut.toCpu.toFloatArray.head.toDouble,2047.5,1e-1)
        boundaryOut.close();boundaryV.close();boundaryK.close();boundaryQ.close()
        assertEquals(session.allocatedBytes,0L)
      }
    finally
      if profileEnabled then
        System.setErr(previous)
        profileStream.close()
    if profileEnabled then
      val summary=diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8)
      val fusedEnabled=sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION").contains("1")
      val expected=if fusedEnabled then "attention_fused_fp32" else "attention_scores_fp32"
      assert(summary.contains(expected),s"supported dense attention did not dispatch $expected: $summary")
      if fusedEnabled then assert(!summary.contains("attention_scores_fp32"),s"fused path also used legacy score kernels: $summary")
      else assert(!summary.contains("attention_fused_fp32"),s"experimental fused path must remain opt-in: $summary")
  }

  test("two-query fused attention reuses K and V for GQA pairs and preserves odd-token fallback") {
    gpu()
    val profileEnabled=sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1")
    val fusedEnabled=sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION").contains("1")
    val multiQueryEnabled=sys.env.get("SCALATORCH_CUDA_MULTI_QUERY_ATTENTION").contains("1")
    val diagnostics=new java.io.ByteArrayOutputStream()
    val previous=System.err
    val profileStream=new java.io.PrintStream(diagnostics,true,java.nio.charset.StandardCharsets.UTF_8)
    try
      if profileEnabled then System.setErr(profileStream)
      CudaSession.withDevice() { session =>
        val cases=Vector((1,2,1,4,7,3,2),(1,2,1,3,5,2,3),(1,1,1,2,4096,1,1))
        for (b,h,hk,t,s,d,dv) <- cases do
          val q=tensor(Seq(b,h,t,d).map(_.toLong),100+t)
          val k=tensor(Seq(b,hk,s,d).map(_.toLong),200+s)
          val v=tensor(Seq(b,hk,s,dv).map(_.toLong),300+dv)
          val expected=denseReference(q,k,v)
          val dq=session.upload(q);val dk=session.upload(k);val dvv=session.upload(v)
          val out=session.scaledDotProductAttention(dq,dk,dvv,maxScoreElements=1L<<20)
          out.toCpu.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a.toDouble,e.toDouble,4e-6*math.max(1.0,math.abs(e.toDouble))))
          out.close();dvv.close();dk.close();dq.close()
        assertEquals(session.allocatedBytes,0L)
      }
    finally
      if profileEnabled then
        System.setErr(previous)
        profileStream.close()
    if profileEnabled then
      val summary=diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8)
      if fusedEnabled && multiQueryEnabled then
        assert(summary.contains("attention_fused_multiquery_fp32"),s"eligible even-token dense attention did not dispatch the two-query CTA: $summary")
        assert(summary.contains("attention_fused_fp32"),s"odd-token dense attention did not fall back to the single-query CTA: $summary")
      else
        assert(!summary.contains("attention_fused_multiquery_fp32"),s"two-query CTA must require both explicit opt-ins: $summary")
  }

  test("dense attention falls back to sliced kernels above the fused key bound") {
    gpu()
    val diagnostics=new java.io.ByteArrayOutputStream()
    val previous=System.err
    val profileStream=new java.io.PrintStream(diagnostics,true,java.nio.charset.StandardCharsets.UTF_8)
    System.setErr(profileStream)
    try
      CudaSession.withDevice() { session =>
        val q=session.fill(Shape.from(Seq(1L,1L,1L,1L)),1f)
        val k=session.upload(Tensor.fromFloats(Array.fill(4097)(0f),Shape.from(Seq(1L,1L,4097L,1L))))
        val v=session.upload(Tensor.fromFloats(Array.tabulate(4097)(_.toFloat),Shape.from(Seq(1L,1L,4097L,1L))))
        val out=session.scaledDotProductAttention(q,k,v,maxScoreElements=4097)
        assertEqualsDouble(out.toCpu.toFloatArray.head.toDouble,2048.0,3e-2)
        out.close();v.close();k.close();q.close()
        assertEquals(session.allocatedBytes,0L)
      }
    finally
      System.setErr(previous)
      profileStream.close()
    val summary=diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8)
    if sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1") then
      assert(summary.contains("attention_scores_fp32"),s"long dense attention did not use sliced fallback: $summary")
      assert(!summary.contains("attention_fused_fp32"),s"long dense attention exceeded the fused bound: $summary")
  }

  test("dense attention falls back when value width exceeds fused CTA channels") {
    gpu()
    val diagnostics=new java.io.ByteArrayOutputStream()
    val previous=System.err
    val profileStream=new java.io.PrintStream(diagnostics,true,java.nio.charset.StandardCharsets.UTF_8)
    System.setErr(profileStream)
    try
      CudaSession.withDevice() { session =>
        val q=session.fill(Shape.from(Seq(1L,1L,1L,1L)),1f)
        val k=session.fill(Shape.from(Seq(1L,1L,2L,1L)),0f)
        val v=session.fill(Shape.from(Seq(1L,1L,2L,257L)),2f)
        val out=session.scaledDotProductAttention(q,k,v)
        assertEquals(out.toCpu.toFloatArray.toVector,Vector.fill(257)(2f))
        out.close();v.close();k.close();q.close()
        assertEquals(session.allocatedBytes,0L)
      }
    finally
      System.setErr(previous)
      profileStream.close()
    if sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1") then
      val summary=diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8)
      assert(summary.contains("attention_scores_fp32"),s"wide values did not use sliced fallback: $summary")
      assert(!summary.contains("attention_fused_fp32"),s"wide values exceeded the fused CTA channel bound: $summary")
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
      if sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION").contains("1") then
        val out=session.scaledDotProductAttention(x,x,x)
        assertEquals(out.toCpu.toFloatArray.toVector,Vector(1f,1f))
        out.close()
      else
        intercept[IllegalArgumentException](session.scaledDotProductAttention(x,x,x))
      assertEquals(session.allocatedBytes,8L)
      x.close()
    }
  }
