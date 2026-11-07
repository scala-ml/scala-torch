package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.ffm.CudaSession

class CudaGegluAccuracySuite extends munit.FunSuite:
  test("GEGLU matches double precision erf reference over the convergence boundary") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val n=20001
    val gate=Array.tabulate(n)(i=>((i-n/2)*0.001f))
    val input=Tensor.fromFloats(Array.fill(n)(1f)++gate,Shape.from(Seq(1L,n.toLong*2)))
    CudaSession.withDevice() { session =>
      val x=session.upload(input);val y=session.geglu(x)
      val actual=y.toCpu.toFloatArray
      for i <- gate.indices do
        val v=gate(i).toDouble
        val expected=(v*0.5*org.apache.commons.numbers.gamma.Erfc.value(-v/math.sqrt(2.0))).toFloat
        assert(math.abs(actual(i).toDouble-expected)<=1e-6,s"input=$v actual=${actual(i)} expected=$expected")
      y.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
