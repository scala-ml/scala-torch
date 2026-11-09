package scalatorch.ffm

import munit.FunSuite
import scalatorch.*

/** Float16 activation kernels compared with the Float32 CUDA path (same inputs, half-precision tolerance). */
class CudaHalfSuite extends FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")

  private def data(count: Int,seed: Long,scale: Double=1.0): Array[Float] =
    val random=new java.util.Random(seed)
    Array.fill(count)((random.nextGaussian()*scale).toFloat)
  private def tensor(values: Array[Float],dims: Long*): Tensor[Float] = Tensor.fromFloats(values,Shape.from(dims.toSeq))

  /** Runs `op` on Float32 and Float16 copies of the inputs and compares the Float32 view of the results. */
  private def compare(inputs: Seq[Tensor[Float]],tolerance: Double)(op: (CudaSession,Seq[CudaTensor]) => CudaTensor): Unit =
    CudaSession.withDevice(maxBytes=1L<<30) { s =>
      assume(s.hasCublas,"cuBLAS unavailable")
      val full=inputs.map(s.upload)
      val half=inputs.map(s.uploadHalf)
      val a=op(s,full);val b=op(s,half)
      try
        assert(!a.half && b.half)
        assertEquals(a.shape,b.shape)
        val x=a.toCpu.toFloatArray;val y=b.toCpu.toFloatArray
        var worst=0.0
        for i <- x.indices do worst=math.max(worst,math.abs(x(i)-y(i))/(1.0+math.abs(x(i))))
        assert(worst<tolerance,s"max relative error $worst")
      finally { b.close();a.close();half.foreach(_.close());full.foreach(_.close()) }
    }

  test("add, broadcast add, scale and silu") {
    gpu()
    val x=data(2*3*5*7,1);val y=data(2*3*5*7,2);val bias=data(7,3)
    compare(Seq(tensor(x,2,3,5,7),tensor(y,2,3,5,7)),2e-3)((s,v) => s.add(v(0),v(1)))
    compare(Seq(tensor(x,2,3,5,7),tensor(bias,7)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,2,3,5,7),tensor(data(3,4),3,1,1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,2,3,5,7)),2e-3)((s,v) => s.scale(v(0),0.37f))
    compare(Seq(tensor(x,2,3,5,7)),2e-3)((s,v) => s.silu(v(0)))
  }

  test("geglu, transpose, cat and nearest resize") {
    gpu()
    compare(Seq(tensor(data(4*10,5),4,10)),3e-3)((s,v) => s.geglu(v(0)))
    compare(Seq(tensor(data(2*3*4*5,6),2,3,4,5)),1e-3)((s,v) => s.transpose(v(0),1,3))
    compare(Seq(tensor(data(2*3*4,7),2,3,4),tensor(data(2*2*4,8),2,2,4)),1e-3)((s,v) => s.cat(Seq(v(0),v(1)),1))
    compare(Seq(tensor(data(1*3*4*5,9),1,3,4,5)),1e-3)((s,v) => s.resizeNearest2d(v(0),8,10))
  }

  test("GroupNorm and LayerNorm with half affine parameters") {
    gpu()
    val x=data(2*8*6*6,10,3.0);val w=data(8,11);val b=data(8,12)
    compare(Seq(tensor(x,2,8,6,6),tensor(w,8),tensor(b,8)),6e-3)((s,v) => s.groupNorm(v(0),4,1e-5,Some(v(1)),Some(v(2))))
    compare(Seq(tensor(x,2,8,6,6)),6e-3)((s,v) => s.groupNorm(v(0),2))
    compare(Seq(tensor(data(11*64,13,2.0),11,64),tensor(data(64,14),64),tensor(data(64,15),64)),6e-3)((s,v) =>
      s.layerNorm(v(0),1e-5,Some(v(1)),Some(v(2))))
  }

  test("transposed-B GEMM and non-causal attention") {
    gpu()
    compare(Seq(tensor(data(37*129,16),37,129),tensor(data(70*129,17,0.2),70,129)),8e-3)((s,v) => s.matmulTransposedB(v(0),v(1)))
    val (b,h,t,kt,d)=(2,3,19,77,16)
    compare(Seq(tensor(data(b*h*t*d,18),b.toLong,h.toLong,t.toLong,d.toLong),
        tensor(data(b*h*kt*d,19),b.toLong,h.toLong,kt.toLong,d.toLong),tensor(data(b*h*kt*d,20),b.toLong,h.toLong,kt.toLong,d.toLong)),1e-2)(
      (s,v) => s.scaledDotProductAttention(v(0),v(1),v(2)))
  }

  test("convolution: 3x3 with padding, strided 3x3, 1x1 and grouped") {
    gpu()
    val x=data(2*8*12*12,21)
    compare(Seq(tensor(x,2,8,12,12),tensor(data(16*8*3*3,22,0.2),16,8,3,3)),1e-2)((s,v) =>
      s.conv2dFast(v(0),v(1),Conv2dConfig(padding=(1,1))))
    compare(Seq(tensor(x,2,8,12,12),tensor(data(16*8*3*3,23,0.2),16,8,3,3)),1e-2)((s,v) =>
      s.conv2dFast(v(0),v(1),Conv2dConfig(stride=(2,2),padding=(1,1))))
    compare(Seq(tensor(x,2,8,12,12),tensor(data(24*8,24,0.2),24,8,1,1)),1e-2)((s,v) => s.conv2dFast(v(0),v(1)))
    compare(Seq(tensor(x,2,8,12,12),tensor(data(8*4*3*3,25,0.2),8,4,3,3)),1e-2)((s,v) =>
      s.conv2dFast(v(0),v(1),Conv2dConfig(padding=(1,1),groups=2)))
  }

  test("GELU, QuickGELU and causal attention") {
    gpu()
    compare(Seq(tensor(data(5*33,26,2.0),5,33)),3e-3)((s,v) => s.geluErf(v(0)))
    compare(Seq(tensor(data(5*33,27,2.0),5,33)),3e-3)((s,v) => s.quickGelu(v(0)))
    val (b,h,t,d)=(2,3,23,16)
    compare(Seq(tensor(data(b*h*t*d,28),b.toLong,h.toLong,t.toLong,d.toLong),
        tensor(data(b*h*t*d,29),b.toLong,h.toLong,t.toLong,d.toLong),tensor(data(b*h*t*d,30),b.toLong,h.toLong,t.toLong,d.toLong)),1e-2)(
      (s,v) => s.causalAttention(v(0),v(1),v(2)))
  }

  test("half attention with a tiny scratch budget (heads and queries chunked) equals the unchunked result") {
    gpu()
    val (b,h,t,kt,d)=(2,3,53,77,16)
    val q=tensor(data(b*h*t*d,31),b.toLong,h.toLong,t.toLong,d.toLong)
    val k=tensor(data(b*h*kt*d,32),b.toLong,h.toLong,kt.toLong,d.toLong)
    val v=tensor(data(b*h*kt*d,33),b.toLong,h.toLong,kt.toLong,d.toLong)
    def run(scratchBytes: Long): Array[Float] =
      val previous=CudaSession.attentionScratchBytes
      CudaSession.attentionScratchBytes=scratchBytes
      try CudaSession.withDevice(maxBytes=1L<<30) { s =>
        assume(s.hasCublas,"cuBLAS unavailable")
        val (qa,ka,va)=(s.uploadHalf(q),s.uploadHalf(k),s.uploadHalf(v))
        val out=s.scaledDotProductAttention(qa,ka,va,maxScoreElements=1L)
        try out.toCpu.toFloatArray finally { out.close();qa.close();ka.close();va.close() }
      } finally CudaSession.attentionScratchBytes=previous
    val whole=run(1L<<30)
    // 77 keys x 2 bytes = 154 B per query row: 1 KiB holds 6 rows, so a head (53 queries) needs 9 query blocks
    val chunked=run(1024L)
    assertEquals(chunked.length,whole.length)
    var worst=0.0
    for i <- whole.indices do worst=math.max(worst,math.abs(whole(i)-chunked(i)))
    assert(worst<1e-3,s"max abs difference $worst")
  }

  test("uploadHalfBits matches uploadHalf") {
    gpu()
    val x=data(3*7*5,41,3.0)
    CudaSession.withDevice(maxBytes=1L<<30) { s =>
      val arena=java.lang.foreign.Arena.ofConfined()
      try
        val seg=arena.allocate(x.length*2L,2)
        val sb=seg.asByteBuffer().order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()
        x.indices.foreach(i => sb.put(i,java.lang.Float.floatToFloat16(x(i))))
        val a=s.uploadHalf(tensor(x,3,7,5));val b=s.uploadHalfBits(Shape.from(Seq(3L,7L,5L)),seg)
        try
          assert(a.half && b.half)
          assertEquals(a.toCpu.toFloatArray.toSeq,b.toCpu.toFloatArray.toSeq)
        finally { a.close();b.close() }
      finally arena.close()
    }
  }

  test("AutoBudget uses the free device memory instead of a fixed number; 'auto' parses") {
    gpu()
    assertEquals(CudaSession.parseBudget("auto"),CudaSession.AutoBudget)
    assertEquals(CudaSession.parseBudget(" 1048576 "),1048576L)
    CudaSession.withDevice(maxBytes=CudaSession.AutoBudget) { s =>
      assert(s.maxBytes>=268435456L && s.maxBytes<Long.MaxValue,s"effective budget ${s.maxBytes}")
      val big=s.fillHalf(Shape.from(Seq(64L,1024L,1024L)),1f) // 128 MiB, beyond a tiny fixed budget but fine here
      big.close()
    }
    CudaSession.withDevice(maxBytes=1024L) { s =>
      intercept[IllegalArgumentException](s.fillHalf(Shape.from(Seq(1024L,1024L)),1f))
    }
  }
