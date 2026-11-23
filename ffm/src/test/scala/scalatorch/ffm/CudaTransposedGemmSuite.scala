package scalatorch.ffm

import munit.FunSuite
import scalatorch.*

class CudaTransposedGemmSuite extends FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")

  private def check(half: Boolean,tolerance: Double): Unit =
    val (m,n,k)=(37,70,129)
    val random=new java.util.Random(3)
    val x=Array.fill(m*k)(random.nextGaussian().toFloat)
    val w=Array.fill(n*k)((random.nextGaussian()*0.2).toFloat)
    val out=CudaSession.withDevice(maxBytes=1L<<28) { s =>
      assume(s.hasCublas,"cuBLAS unavailable")
      val cx=s.upload(Tensor.fromFloats(x,Shape.from(Seq(m.toLong,k.toLong))))
      val cw=s.upload(Tensor.fromFloats(w,Shape.from(Seq(n.toLong,k.toLong))))
      val packed=if half then Some(s.toHalf(cw)) else None
      val r=packed.fold(s.matmulTransposedB(cx,cw))(p => s.matmulTransposedBHalf(cx,p,n,k))
      try
        assertEquals(r.shape,Shape.from(Seq(m.toLong,n.toLong)))
        r.toCpu.toFloatArray
      finally { r.close();packed.foreach(_.close());cw.close();cx.close() }
    }
    var worst=0.0
    for i <- 0 until m; j <- 0 until n do
      val expected=(0 until k).map(p => x(i*k+p).toDouble*w(j*k+p)).sum
      worst=math.max(worst,math.abs(expected-out(i*n+j)))
    assert(worst<tolerance,s"max error $worst")

  test("x times transposed weight matches the reference in Float32") { gpu();check(half=false,tolerance=1e-4) }
  test("Float16 tensor-core GEMM stays within half-precision error of the reference") { gpu();check(half=true,tolerance=0.08) }
