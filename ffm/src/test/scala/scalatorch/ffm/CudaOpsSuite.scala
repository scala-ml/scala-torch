package scalatorch.ffm

import scalatorch.*
import scalatorch.cpu.CpuFloatBackend

class CudaOpsSuite extends munit.FunSuite:
  private def gpu(): Unit = assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def closeEnough(actual: Array[Float], expected: Array[Float], tolerance: Double = 2e-6): Unit =
    assertEquals(actual.length,expected.length)
    actual.zip(expected).foreach { (a,e) =>
      if e.isNaN then assert(a.isNaN)
      else if e.isInfinite then assertEquals(a,e)
      else assertEqualsDouble(a.toDouble,e.toDouble,tolerance*math.max(1.0,math.abs(e.toDouble)))
    }
  private def tensor(values: Array[Float], dims: Long*) = Tensor.fromFloats(values,Shape.from(dims))

  test("CUDA elementwise operations agree across blocks, extremes and empty shapes") {
    gpu()
    CudaSession.withDevice() { session =>
      val backend: FloatBackend[CudaTensor]=session
      val values=Array.tabulate(137)(i => (i-68).toFloat/3) ++ Array(Float.NegativeInfinity,Float.PositiveInfinity,Float.NaN,-0f,-100f,100f)
      val x=tensor(values,values.length.toLong)
      val y=tensor(Array.fill(values.length)(2f),values.length.toLong)
      val a=backend.upload(x);val b=backend.upload(y)
      closeEnough(backend.download(backend.add(a,b)).toFloatArray,CpuFloatBackend.add(x,y).toFloatArray)
      closeEnough(backend.download(backend.multiply(a,b)).toFloatArray,CpuFloatBackend.multiply(x,y).toFloatArray)
      val activated=backend.download(backend.silu(a)).toFloatArray
      closeEnough(activated,CpuFloatBackend.silu(x).toFloatArray)
      assertEquals(java.lang.Float.floatToRawIntBits(activated(137)),0x80000000)
      val empty=backend.upload(tensor(Array.emptyFloatArray,0,3))
      assertEquals(backend.shape(backend.silu(empty)),Shape.from(Seq(0L,3L)))
    }
  }

  test("CUDA RMSNorm agrees for strided inputs, large finite values and empty rows") {
    gpu()
    CudaSession.withDevice() { session =>
      val normal=tensor(Array.tabulate(70*5)(i => (i%17-8).toFloat),5,70).transpose(0,1)
      val huge=tensor(Array(Float.MaxValue,-Float.MaxValue,1e-30f,-1e-30f),2,2)
      for x <- Seq(normal,huge,tensor(Array.emptyFloatArray,0,2)) do
        val w=tensor(Array.tabulate(x.shape.dims.last.toInt)(i => (i+1).toFloat),x.shape.dims.last)
        val expected=CpuFloatBackend.rmsNorm(x,w,1e-6).toFloatArray
        val out=session.rmsNorm(session.upload(x),session.upload(w),1e-6)
        closeEnough(out.toCpu.toFloatArray,expected)
    }
  }

  test("the same backend-generic gated MLP and residual runs on CPU and CUDA") {
    gpu()
    def run[T](backend: FloatBackend[T]): Array[Float] =
      val owned=scala.collection.mutable.ArrayBuffer.empty[T]
      def keep(x: T): T = { owned+=x;x }
      try
        val input=keep(backend.upload(tensor(Array(1f,-2f,3f,4f),2,2)))
        val w=keep(backend.upload(tensor(Array(.1f,.2f,-.3f,.4f,.5f,-.6f),2,3)))
        val down=keep(backend.upload(tensor(Array(.1f,-.2f,.3f,-.4f,.5f,-.6f),3,2)))
        val scale=keep(backend.upload(tensor(Array(1f,2f),2)))
        val normalized=keep(backend.rmsNorm(input,scale,1e-6))
        val gate=keep(backend.matmul(normalized,w))
        val activated=keep(backend.silu(gate))
        val product=keep(backend.multiply(activated,gate))
        val projected=keep(backend.matmul(product,down))
        backend.download(keep(backend.add(input,projected))).toFloatArray
      finally owned.reverseIterator.foreach(backend.release)
    val expected=run(CpuFloatBackend)
    CudaSession.withDevice() { session =>
      closeEnough(run(session),expected)
      assertEquals(session.allocatedBytes,0L)
    }
  }

  test("invalid shape, epsilon, lifetime and budget fail without allocating output") {
    gpu()
    CudaSession.withDevice(maxBytes=32) { session =>
      val x=session.upload(tensor(Array(1f,2f,3f,4f),2,2))
      val w=session.upload(tensor(Array(1f,1f),2))
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](session.add(x,w))
      intercept[IllegalArgumentException](session.multiply(x,w))
      intercept[IllegalArgumentException](session.rmsNorm(x,x,1e-6))
      for e <- Seq(0.0,Double.NaN,Double.PositiveInfinity) do
        intercept[IllegalArgumentException](session.rmsNorm(x,w,e))
      intercept[IllegalArgumentException](session.silu(x))
      assertEquals(session.allocatedBytes,before)
      w.close()
      intercept[IllegalStateException](session.rmsNorm(x,w,1e-6))
      x.close()
      intercept[IllegalStateException](session.silu(x))
    }
  }
