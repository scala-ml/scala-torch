package scalatorch.ffm

import scalatorch.*

class CudaSlicedAttentionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def data(dims: Seq[Long],seed: Int): Tensor[Float] =
    val shape=Shape.from(dims);val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
  test("explicit work limit permits 4096 tokens and eight heads with bounded scratch") {
    gpu()
    CudaSession.withDevice(maxBytes=20L*1024*1024) { s =>
      // SD1.5 512px self-attention token/head geometry; width one isolates the score matrix.
      val shape=Shape.from(Seq(1L,8L,4096L,1L))
      val q=s.fill(shape,0f);val k=s.fill(shape,0f);val v=s.fill(shape,2f)
      val before=s.allocatedBytes
      intercept[IllegalArgumentException](s.scaledDotProductAttention(q,k,v))
      assertEquals(s.allocatedBytes,before)
      val out=s.scaledDotProductAttention(q,k,v,maxScoreElements=134217728L)
      assert(out.toCpu.toFloatArray.forall(_==2f))
      out.close()
      assertEquals(s.allocatedBytes,before)
      v.close();k.close();q.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("query slices preserve exact dense outputs across batch head and GQA boundaries") {
    gpu()
    CudaSession.withDevice() { s =>
      val q=s.upload(data(Seq(2L,4L,7L,5L),1))
      val k=s.upload(data(Seq(2L,2L,11L,5L),2))
      val v=s.upload(data(Seq(2L,2L,11L,3L),3))
      // All keys visible: preserved causal kernel supplies the serial reference.
      val full=s.causalAttention(q,k,v,queryStart=10)
      val expected=full.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits);full.close()
      val before=s.allocatedBytes
      for rows <- Seq(1,2,5,17,64) do
        val out=s.scaledDotProductAttention(q,k,v,maxWorkspaceBytes=rows*11L*8+7)
        assertEquals(out.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits),expected)
        out.close()
        assertEquals(s.allocatedBytes,before)
      intercept[IllegalArgumentException](s.scaledDotProductAttention(q,k,v,maxWorkspaceBytes=87))
      intercept[IllegalArgumentException](s.scaledDotProductAttention(q,k,v,maxScoreElements=1,maxWorkspaceBytes=88))
      assertEquals(s.allocatedBytes,before)
      v.close();k.close();q.close()
      val qn=s.fill(Shape.from(Seq(2L,2L,3L,1L)),1f)
      val kn=s.upload(Tensor.fromFloats(Array(0f,Float.NaN,0f,1f,0f,1f,0f,1f),Shape.from(Seq(2L,2L,2L,1L))))
      val vn=s.upload(Tensor.fromFloats(Array(1f,2f,Float.PositiveInfinity,1f,-0f,0f,3f,4f),kn.shape))
      val dense=s.causalAttention(qn,kn,vn,queryStart=1)
      val sliced=s.scaledDotProductAttention(qn,kn,vn,maxWorkspaceBytes=16)
      assertEquals(sliced.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits),
        dense.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits))
      sliced.close();dense.close();vn.close();kn.close();qn.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("parallel weighted outputs match serial attention at diffusion head widths") {
    gpu()
    CudaSession.withDevice() { s =>
      for width <- Seq(40L,80L,160L) do
        val q=s.upload(data(Seq(1L,8L,17L,width),11))
        val k=s.upload(data(Seq(1L,2L,77L,width),12))
        val v=s.upload(data(Seq(1L,2L,77L,width),13))
        val reference=s.causalAttention(q,k,v,queryStart=76)
        val parallel=s.scaledDotProductAttention(q,k,v,maxWorkspaceBytes=77L*8*3)
        assertEquals(parallel.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits),
          reference.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits))
        parallel.close();reference.close();v.close();k.close();q.close()
        assertEquals(s.allocatedBytes,0L)
    }
  }
  test("full spatial key count at width40 matches the original serial kernel") {
    gpu()
    CudaSession.withDevice() { s =>
      val q=s.upload(data(Seq(1L,1L,512L,40L),21))
      val k=s.upload(data(Seq(1L,1L,4096L,40L),22))
      val v=s.upload(data(Seq(1L,1L,4096L,40L),23))
      val start=System.nanoTime()
      val reference=s.causalAttention(q,k,v,queryStart=4095)
      val middle=System.nanoTime()
      val parallel=s.scaledDotProductAttention(q,k,v)
      val end=System.nanoTime()
      println(s"attention 512x4096 width40 serialMs=${(middle-start)/1e6} parallelMs=${(end-middle)/1e6}")
      assertEquals(parallel.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits),
        reference.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits))
      parallel.close();reference.close();v.close();k.close();q.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("automatic query slicing fits a 64 KiB workspace instead of an 8 MiB score matrix") {
    gpu()
    val qs=Shape.from(Seq(1L,2L,512L,4L));val ks=Shape.from(Seq(1L,2L,1024L,4L))
    val vs=Shape.from(Seq(1L,2L,1024L,2L));val os=Shape.from(Seq(1L,2L,512L,2L))
    val inputs=(qs.numel+ks.numel+vs.numel)*4
    CudaSession.withDevice(maxBytes=inputs+os.numel*4+65536) { s =>
      val q=s.fill(qs,0f);val k=s.fill(ks,0f);val v=s.fill(vs,2f)
      val out=s.scaledDotProductAttention(q,k,v)
      assertEquals(out.shape,os)
      assert(out.toCpu.toFloatArray.forall(_==2f))
      out.close()
      assertEquals(s.allocatedBytes,inputs)
      q.close();k.close();v.close()
    }
  }
  test("slice workspace preflight rejects insufficient total memory and keeps empty shape semantics") {
    gpu()
    CudaSession.withDevice(maxBytes=43) { s =>
      val x=s.fill(Shape.from(Seq(1L,1L,3L,1L)),1f)
      intercept[IllegalArgumentException](s.scaledDotProductAttention(x,x,x))
      assertEquals(s.allocatedBytes,12L)
      x.close()
      val empty=s.fill(Shape.from(Seq(0L,1L,3L,1L)),0f)
      val result=s.scaledDotProductAttention(empty,empty,empty,maxWorkspaceBytes=0)
      assertEquals(result.shape,empty.shape)
      result.close()
      intercept[IllegalArgumentException](s.scaledDotProductAttention(empty,empty,empty,maxWorkspaceBytes = -1))
      empty.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
