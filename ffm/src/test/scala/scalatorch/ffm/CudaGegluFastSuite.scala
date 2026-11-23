package scalatorch.ffm

import scalatorch.*

class CudaGegluFastSuite extends munit.FunSuite:
  test("fast GEGLU error stays within 2e-7 of the value scaled by the gate magnitude") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice() { session =>
      val rows=64;val width=257
      val rng=new scala.util.Random(7)
      val values=Array.tabulate(rows*width*2)(i => (rng.nextGaussian()*(if i%97==0 then 12.0 else 2.5)).toFloat)
      val x=session.upload(Tensor.fromFloats(values,Shape.from(Seq(rows.toLong,width*2L))))
      val exact=session.geglu(x);val fast=session.geglu(x,fast=true)
      val a=exact.toCpu.toFloatArray;val b=fast.toCpu.toFloatArray
      var worst=0.0
      a.indices.foreach { i =>
        val row=i/width;val col=i%width
        val v=math.abs(values(row*width*2+col).toDouble);val g=math.abs(values(row*width*2+width+col).toDouble)
        worst=math.max(worst,math.abs(a(i)-b(i)).toDouble/(v*(g+1)+1e-9))
      }
      assert(worst<2e-7,s"worst scaled error $worst")
      exact.close();fast.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
