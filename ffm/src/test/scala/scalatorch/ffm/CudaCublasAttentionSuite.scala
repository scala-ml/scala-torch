package scalatorch.ffm

import munit.FunSuite
import scalatorch.*

class CudaCublasAttentionSuite extends FunSuite:
  test("batched cuBLAS attention matches a double-precision reference") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    val (b,h,qt,kt,d,dv)=(2,3,37,517,40,48)
    val random=new java.util.Random(9)
    def data(n: Int)=Array.fill(n)(random.nextGaussian().toFloat)
    val q=data(b*h*qt*d);val k=data(b*h*kt*d);val v=data(b*h*kt*dv)
    def t(a: Array[Float],tokens: Int,w: Int)=Tensor.fromFloats(a,Shape.from(Seq(b.toLong,h.toLong,tokens.toLong,w.toLong)))
    val out=CudaSession.withDevice(maxBytes=1L<<30) { s =>
      val (cq,ck,cv)=(s.upload(t(q,qt,d)),s.upload(t(k,kt,d)),s.upload(t(v,kt,dv)))
      val r=s.scaledDotProductAttention(cq,ck,cv)
      try r.toCpu.toFloatArray finally { r.close();cv.close();ck.close();cq.close() }
    }
    var worst=0.0
    for bh <- 0 until b*h; i <- 0 until qt do
      val scores=Array.tabulate(kt)(j => (0 until d).map(x => q((bh*qt+i)*d+x).toDouble*k((bh*kt+j)*d+x)).sum/math.sqrt(d))
      val m=scores.max;val e=scores.map(x => math.exp(x-m));val z=e.sum
      for c <- 0 until dv do
        val expected=(0 until kt).map(j => e(j)/z*v((bh*kt+j)*dv+c)).sum
        worst=math.max(worst,math.abs(expected-out((bh*qt+i)*dv+c)))
    assert(worst<2e-4,s"max error $worst")
  }
