package scalatorch.ffm

import munit.FunSuite
import scala.util.Using
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

  test("broadcast add: bias along one axis (fast path), along two axes, and a broadcast first operand") {
    gpu()
    val x=data(2*6*5*7,21)
    // one axis: channel bias of a rank-4 tensor, the batch axis, the last axis of a rank-3 tensor, a scalar
    compare(Seq(tensor(x,2,6,5,7),tensor(data(6,22),1,6,1,1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,2,6,5,7),tensor(data(2,23),2,1,1,1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,2,30,7),tensor(data(7,24),7)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,2,6,5,7),tensor(data(1,25),1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(x,1,2,6,5,7),tensor(data(5,26),5,1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    // two axes (time embedding [N,C,1,1]) and a first operand that is itself broadcast: the general path
    compare(Seq(tensor(x,2,6,5,7),tensor(data(12,27),2,6,1,1)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(data(6,28),1,6,1,1),tensor(x,2,6,5,7)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
    compare(Seq(tensor(data(2*5,29),2,1,5,1),tensor(data(6*7,30),1,6,1,7)),2e-3)((s,v) => s.addBroadcast(v(0),v(1)))
  }

  test("GroupNorm over wide groups (1024-thread statistics) and softmax over long and short rows") {
    gpu()
    val x=data(2*4*48*48,31,2.0)
    compare(Seq(tensor(x,2,4,48,48),tensor(data(4,32),4),tensor(data(4,33),4)),6e-3)((s,v) => s.groupNorm(v(0),2,1e-5,Some(v(1)),Some(v(2))))
    // rows of 700 keys (more than one element per thread) and of 5 keys (most threads see nothing); large scores exercise the rescaling
    for (queries,keys,scale) <- Seq((9,700,1.0),(9,700,6.0),(300,5,3.0)) do
      val q=tensor(data(2*queries*8,34,scale),1,2,queries,8);val k=tensor(data(2*keys*8,35,scale),1,2,keys,8)
      val value=tensor(data(2*keys*8,36),1,2,keys,8)
      compare(Seq(q,k,value),2e-2)((s,v) => s.scaledDotProductAttention(v(0),v(1),v(2)))
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
    // rows of the single-kernel LayerNorm: narrower than a block, several values per thread, the widest it takes, without affine
    // parameters, and a width above its limit (statistics and apply apart)
    for (rows,width) <- Seq((5,7),(3,640),(4,1280),(2,2048),(2,2304)) do
      compare(Seq(tensor(data(rows*width,21,2.0),rows.toLong,width.toLong),tensor(data(width,22),width.toLong),tensor(data(width,23),width.toLong)),6e-3)(
        (s,v) => s.layerNorm(v(0),1e-5,Some(v(1)),Some(v(2))))
      compare(Seq(tensor(data(2*rows*width,24,2.0),2,rows.toLong,width.toLong)),6e-3)((s,v) => s.layerNorm(v(0)))
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

  test("convolution of a batch as one GEMM: 3x3 padded, strided, dilated and 5x3 kernels over three and five samples") {
    gpu()
    for (n,ci,co,h,w) <- Seq((3,5,7,9,11),(5,4,6,8,8)) do
      val x=tensor(data(n*ci*h*w,41+n),n.toLong,ci.toLong,h.toLong,w.toLong)
      compare(Seq(x,tensor(data(co*ci*9,42,0.3),co.toLong,ci.toLong,3,3)),1e-2)((s,v) => s.conv2dFast(v(0),v(1),Conv2dConfig(padding=(1,1))))
      compare(Seq(x,tensor(data(co*ci*9,43,0.3),co.toLong,ci.toLong,3,3)),1e-2)((s,v) =>
        s.conv2dFast(v(0),v(1),Conv2dConfig(stride=(2,2),padding=(1,1))))
      compare(Seq(x,tensor(data(co*ci*9,44,0.3),co.toLong,ci.toLong,3,3)),1e-2)((s,v) =>
        s.conv2dFast(v(0),v(1),Conv2dConfig(padding=(2,2),dilation=(2,2))))
      compare(Seq(x,tensor(data(co*ci*15,45,0.3),co.toLong,ci.toLong,5,3)),1e-2)((s,v) => s.conv2dFast(v(0),v(1),Conv2dConfig(padding=(2,0))))
  }

  test("fused attention: widths 16 to 160 (also not multiples of 16), different key and query counts, value width, large scores") {
    gpu()
    // With SCALATORCH_CUDA_FUSED_ATTENTION_HALF=0 this compares the cuBLAS path on the same shapes.
    println(s"fused attention kernel under test: ${!sys.env.get("SCALATORCH_CUDA_FUSED_ATTENTION_HALF").contains("0")}")
    // (heads, queries, keys, width, value width, score scale)
    for (h,t,kt,d,dv,scale) <- Seq((3,16,16,16,16,1.0),(2,32,48,40,40,1.0),(5,64,64,64,64,1.0),(2,48,160,80,80,1.0),
        (1,16,32,160,160,1.0),(3,32,64,24,56,1.0),(2,64,256,40,40,4.0),(2,16,1024,64,64,2.0),
        // every kernel variant: 1 / 2 / 4 query tiles per warp x 3 / 4 / 5 chunks, with and without a width remainder
        // (widths 33 and 50 are not multiples of 8: they must take the cuBLAS path, a fragment load there would be misaligned)
        (3,16,64,48,48,1.0),(3,32,64,33,33,1.0),(2,64,128,48,48,3.0),(3,16,48,50,50,1.0),(2,32,96,64,64,1.0),(2,128,64,56,56,1.0),
        (3,16,32,80,80,1.0),(2,32,64,72,72,2.0),(2,64,64,80,80,1.0),(7,96,4096,40,40,1.0),
        // key counts that are not multiples of 16 (a text context has 77 tokens): the last key tile is masked
        (3,32,77,64,64,1.0),(2,64,77,40,40,3.0),(2,16,17,80,80,1.0),(3,128,93,48,48,1.0),(2,32,154,56,56,2.0),(2,16,31,72,72,1.0)) do
      val q=tensor(data(h*t*d,51,scale),1,h.toLong,t.toLong,d.toLong);val k=tensor(data(h*kt*d,52,scale),1,h.toLong,kt.toLong,d.toLong)
      val v=tensor(data(h*kt*dv,53),1,h.toLong,kt.toLong,dv.toLong)
      compare(Seq(q,k,v),2e-2)((s,x) => s.scaledDotProductAttention(x(0),x(1),x(2)))
  }

  test("linear layer with the bias folded into the product equals product plus broadcast add") {
    gpu()
    for (m,k,n) <- Seq((37,129,70),(1,16,5),(300,64,64)) do
      CudaSession.withDevice(maxBytes=1L<<30) { s =>
        assume(s.hasCublas,"cuBLAS unavailable")
        val x=s.uploadHalf(tensor(data(m*k,61),m.toLong,k.toLong));val w=s.uploadHalf(tensor(data(n*k,62,0.2),n.toLong,k.toLong))
        val bias=s.uploadHalf(tensor(data(n,63,3.0),n.toLong))
        val product=s.matmulTransposedB(x,w);val expected=s.addBroadcast(product,bias)
        val fused=s.linearWithBias(x,w,bias).getOrElse(fail("the fused path did not apply"))
        assertEquals(fused.shape,expected.shape)
        val a=s.castToFloat(expected).toCpu.toFloatArray;val b=s.castToFloat(fused).toCpu.toFloatArray
        var worst=0.0
        for i <- a.indices do worst=math.max(worst,math.abs(a(i)-b(i))/(1.0+math.abs(a(i))))
        assert(worst<4e-3,s"max relative error $worst")
        // Float32 inputs are not handled here
        val full=s.upload(tensor(data(m*k,61),m.toLong,k.toLong))
        assertEquals(s.linearWithBias(full,w,bias),None)
        Seq(x,w,bias,product,expected,fused,full).foreach(_.close())
      }
  }

  test("residual folded into a linear layer and bias folded into GEGLU equal the separate passes") {
    gpu()
    for (m,k,n) <- Seq((37,129,70),(1,16,6),(300,64,64)) do
      CudaSession.withDevice(maxBytes=1L<<30) { s =>
        assume(s.hasCublas,"cuBLAS unavailable")
        def floats(t: CudaTensor)=s.castToFloat(t).toCpu.toFloatArray
        def worst(a: Array[Float],b: Array[Float])=a.indices.map(i => math.abs(a(i)-b(i))/(1.0+math.abs(a(i)))).max
        val x=s.uploadHalf(tensor(data(m*k,61),m.toLong,k.toLong));val w=s.uploadHalf(tensor(data(n*k,62,0.2),n.toLong,k.toLong))
        val bias=s.uploadHalf(tensor(data(n,63,3.0),n.toLong));val residual=s.uploadHalf(tensor(data(m*n,64,2.0),m.toLong,n.toLong))
        val plain=s.linearWithBias(x,w,bias).getOrElse(fail("the fused bias path did not apply"))
        val expected=s.add(residual,plain)
        val fused=s.linearWithBias(x,w,bias,Some(residual)).getOrElse(fail("the residual path did not apply"))
        assertEquals(fused.shape,expected.shape)
        val e1=worst(floats(expected),floats(fused))
        assert(e1<4e-3,s"residual: max relative error $e1")
        // a residual of another shape is refused
        val other=s.uploadHalf(tensor(data(m*n,65),(m*n).toLong))
        assertEquals(s.linearWithBias(x,w,bias,Some(other)),None)
        // GEGLU: bias added inside the kernel against a broadcast add before it
        val product=s.matmulTransposedB(x,w)
        val withBias=s.addBroadcast(product,bias);val reference=s.geglu(withBias,fast=true)
        assert(s.canGegluBias(product,bias))
        val gated=s.gegluBias(product,bias)
        assertEquals(gated.shape,reference.shape)
        val e2=worst(floats(reference),floats(gated))
        assert(e2<4e-3,s"geglu: max relative error $e2")
        Seq(x,w,bias,residual,plain,expected,fused,other,product,withBias,reference,gated).foreach(_.close())
      }
  }

  test("tensor-core fragment layout of this device") {
    gpu()
    CudaSession.withDevice(maxBytes=1L<<28) { s =>
      val layout=s.wmmaLayout()
      for (kind,name) <- Seq(0->"accumulator",1->"matrix_a",2->"matrix_b col",3->"matrix_b row") do
        val part=layout.slice(kind*256,kind*256+256)
        assertEquals(part.sorted.toSeq,(0 until 256).toSeq,s"$name does not cover the matrix")
        println(s"[wmma] $name: "+(0 until 6).map(lane => s"lane $lane: "+part.slice(lane*8,lane*8+8).map(v => s"(${v/16},${v%16})").mkString("")).mkString(" | "))
      println("[wmma] matrix_a equals accumulator: "+(layout.slice(0,256).toSeq==layout.slice(256,512).toSeq))
    }
  }

  test("tokens-major attention equals heads-major attention on transposed copies") {
    gpu()
    CudaSession.withDevice(maxBytes=1L<<30) { s =>
      assume(s.hasCublas,"cuBLAS unavailable")
      assume(s.canAttendTokensMajor(64,64,40),"fused attention is switched off or not usable on this device")
      // (batch, heads, queries, keys, width): every kernel variant, one and several batch members
      for (b,h,t,kt,d) <- Seq((1,8,64,64,40),(2,8,32,48,40),(3,5,16,32,64),(2,10,128,64,64),(2,4,48,160,80),(1,3,64,4096,56),
          (2,10,64,77,64),(2,8,32,77,40),(1,4,16,93,80),(3,6,128,17,56)) do
        assert(s.canAttendTokensMajor(t,kt,d))
        def tokens(count: Int,seed: Long)=s.uploadHalf(tensor(data(b*count*h*d,seed),b.toLong,count.toLong,(h*d).toLong))
        val q=tokens(t,71);val k=tokens(kt,72);val v=tokens(kt,73)
        def headsMajor(x: CudaTensor,count: Int): CudaTensor =
          Using.resource(s.reshape(x,Shape.from(Seq(b.toLong,count.toLong,h.toLong,d.toLong))))(view => s.transpose(view,1,2))
        val expected=Using.resource(headsMajor(q,t)) { qh => Using.resource(headsMajor(k,kt)) { kh => Using.resource(headsMajor(v,kt)) { vh =>
          Using.resource(s.scaledDotProductAttention(qh,kh,vh)) { out => Using.resource(s.transpose(out,1,2))(p => s.castToFloat(p).toCpu.toFloatArray) }
        } } }
        val actual=Using.resource(s.attentionTokensMajor(q,k,v,h))(out => s.castToFloat(out).toCpu.toFloatArray)
        assertEquals(actual.length,expected.length)
        // both go through the same kernels on the same numbers in another layout: equal bit for bit
        assert(actual.sameElements(expected),s"layouts differ for ($b,$h,$t,$kt,$d)")
        Seq(q,k,v).foreach(_.close())
      // queries, keys and values as the three parts of one tensor: the same result as three tensors
      for (b,h,t,d) <- Seq((1,8,64,40),(2,10,32,64),(3,4,48,80),(2,8,128,40)) do
        def tokens(seed: Long)=s.uploadHalf(tensor(data(b*t*h*d,seed),b.toLong,t.toLong,(h*d).toLong))
        val q=tokens(81);val k=tokens(82);val v=tokens(83)
        val separate=Using.resource(s.attentionTokensMajor(q,k,v,h))(out => s.castToFloat(out).toCpu.toFloatArray)
        val packed=Using.resource(s.cat(Seq(q,k,v),2)) { all =>
          assertEquals(all.shape,Shape.from(Seq(b.toLong,t.toLong,3L*h*d)))
          Using.resource(s.attentionPacked(all,h))(out => s.castToFloat(out).toCpu.toFloatArray)
        }
        assert(packed.sameElements(separate),s"packed differs for ($b,$h,$t,$d)")
        Seq(q,k,v).foreach(_.close())
      assert(s.canAttendTokensMajor(64,77,40) && !s.canAttendTokensMajor(60,64,40) && !s.canAttendTokensMajor(64,64,160) && !s.canAttendTokensMajor(64,64,50))
    }
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

  test("shared weights are uploaded once, seen by later sessions and by other threads, and freed on release") {
    gpu()
    val x=data(7*9,51,2.0)
    val source=tensor(x,7,9)
    val key=new Object
    val expected=CudaSession.withDevice(maxBytes=1L<<30) { s => Using.resource(s.uploadHalf(source))(_.toCpu.toFloatArray.toSeq) }
    def readShared(half: Boolean): Seq[Float] = CudaSession.withDevice(maxBytes=1L<<30) { s =>
      Using.resource(s.sharedUpload(key,source,half))(t => (if t.half then Using.resource(s.castToFloat(t))(_.toCpu.toFloatArray) else t.toCpu.toFloatArray).toSeq)
    }
    assertEquals(readShared(true),expected)           // first session uploads
    assertEquals(readShared(true),expected)           // second session reuses the device memory
    val fromThread=new java.util.concurrent.atomic.AtomicReference[Seq[Float]]()
    val t=new Thread(() => fromThread.set(readShared(true)));t.start();t.join()
    assertEquals(fromThread.get(),expected)           // another thread uses the same weights
    assertEquals(readShared(false),x.toSeq)           // the Float32 copy is a separate entry
    CudaSession.releaseSharedWeights(0)
    assertEquals(readShared(true),expected)           // after release it is uploaded again
    CudaSession.releaseSharedWeights(0)
  }
