package scalatorch.autograd

import scalatorch.*
import scalatorch.ffm.CudaSession

class CudaGegluSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def reference(input: Tensor[Float]): Tensor[Float] =
    val x=Variable(input);val axis=input.shape.rank-1;val width=input.shape.dims.last/2
    (x.narrow(axis,0,width)*x.narrow(axis,width,width).gelu()).data
  test("resident GEGLU matches Gaussian-error CPU reference across tails and IEEE values") {
    gpu()
    val gates=Array.tabulate(30721)(i => (i-15360).toFloat/1024)++Array(0f,-0f,Float.MinPositiveValue,
      -Float.MinPositiveValue,Float.MaxValue,-Float.MaxValue,Float.PositiveInfinity,Float.NegativeInfinity,Float.NaN)
    val input=Tensor.fromFloats(gates.zipWithIndex.flatMap((g,i) => Array(if i%2==0 then 1f else -3.7f,g)),
      Shape.from(Seq(gates.length.toLong,2L)))
    val expected=reference(input).toFloatArray
    CudaSession.withDevice() { s =>
      val x=s.upload(input);val result=s.geglu(x)
      assertEquals(result.shape,Shape.from(Seq(gates.length.toLong,1L)))
      result.toCpu.toFloatArray.zip(expected).zipWithIndex.foreach { case ((a,e),i) =>
        if e.isNaN then assert(a.isNaN)
        else if e==0f || e.isInfinity then assertEquals(java.lang.Float.floatToRawIntBits(a),java.lang.Float.floatToRawIntBits(e))
        else assertEqualsDouble(a,e,2.0*math.ulp(e),s"gate=${gates(i)}")
      }
      result.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("GEGLU packs rows correctly and validates layout lifetime and budget") {
    gpu()
    CudaSession.withDevice() { s =>
      val base=Tensor.fromFloats(Array.tabulate(120)(i => (i%19-9).toFloat/3),Shape.from(Seq(2L,6L,10L)))
      for input <- Seq(base,base.transpose(1,2)) do
        val x=s.upload(input);val out=s.geglu(x)
        val expected=reference(input)
        assertEquals(out.shape,expected.shape)
        out.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2.0*math.ulp(e)))
        out.close();x.close()
      for shape <- Seq(Shape.from(Seq.empty),Shape.from(Seq(2L,3L)),Shape.from(Seq(1L,0L))) do
        val x=s.fill(shape,0f)
        intercept[IllegalArgumentException](s.geglu(x))
        x.close()
      val empty=s.fill(Shape.from(Seq(0L,4L)),0f)
      val result=s.geglu(empty)
      assertEquals(result.shape,Shape.from(Seq(0L,2L)))
      result.close();empty.close()
      intercept[IllegalStateException](s.geglu(empty))
      assertEquals(s.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=8) { s =>
      val x=s.fill(Shape.from(Seq(2L)),1f)
      intercept[IllegalArgumentException](s.geglu(x))
      assertEquals(s.allocatedBytes,8L)
      x.close()
    }
  }
