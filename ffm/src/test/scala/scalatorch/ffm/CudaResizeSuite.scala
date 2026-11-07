package scalatorch.ffm

import scalatorch.*

class CudaResizeSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def bits(x: Tensor[Float])=x.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits)
  test("resident nearest resize matches CPU for both modes and preserves IEEE bits") {
    gpu()
    CudaSession.withDevice() { s =>
      val values=Array(0f,-0f,Float.PositiveInfinity,Float.NegativeInfinity,
        java.lang.Float.intBitsToFloat(0x7fc01234),1f,2f,3f,4f,5f,6f,7f)
      val input=Tensor.fromFloats(values,Shape.from(Seq(1L,2L,2L,3L)))
      for cpu <- Seq(input,input.transpose(2,3),input.reshape(Shape.from(Seq(2L,1L,2L,3L)))); mode <- NearestMode.values;
          (h,w) <- Seq((5,7),(1,2),(2,3),(3,1)) do
        val device=s.upload(cpu)
        val result=s.resizeNearest2d(device,h,w,mode)
        device.close()
        assertEquals(bits(result.toCpu),bits(Resize2d.nearest(cpu,h,w,mode)))
        result.close()
      val wide=Tensor.fromFloats(Array.tabulate(65537)(_.toFloat),Shape.from(Seq(1L,1L,1L,65537L)))
      val dw=s.upload(wide)
      for mode <- NearestMode.values do
        val result=s.resizeNearest2d(dw,1,65539,mode)
        assertEquals(bits(result.toCpu),bits(Resize2d.nearest(wide,1,65539,mode)))
        result.close()
      dw.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("nearest resize validates shape lifetime and budget and supports empty batches") {
    gpu()
    CudaSession.withDevice(maxBytes=64) { s =>
      val x=s.fill(Shape.from(Seq(1L,1L,2L,2L)),1f)
      intercept[IllegalArgumentException](s.resizeNearest2d(x,0,2))
      intercept[IllegalArgumentException](s.resizeNearest2d(x,Int.MaxValue,2))
      intercept[IllegalArgumentException](s.resizeNearest2d(x,4,4))
      assertEquals(s.allocatedBytes,16L)
      x.close()
      intercept[IllegalStateException](s.resizeNearest2d(x,2,2))
      for shape <- Seq(Shape.from(Seq(2L,2L)),Shape.from(Seq(1L,0L,2L,2L))) do
        val bad=s.fill(shape,0f)
        intercept[IllegalArgumentException](s.resizeNearest2d(bad,2,2))
        bad.close()
      val empty=s.fill(Shape.from(Seq(0L,2L,3L,4L)),0f)
      val result=s.resizeNearest2d(empty,5,7)
      assertEquals(result.shape,Shape.from(Seq(0L,2L,5L,7L)))
      assert(result.toCpu.toFloatArray.isEmpty)
      empty.close();result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("upsample and convolution compose without intermediate host tensors") {
    gpu()
    val input=Tensor.fromFloats(Array.tabulate(24)(i => i.toFloat/8),Shape.from(Seq(1L,2L,3L,4L)))
    val weight=Tensor.fromFloats(Array.tabulate(36)(i => (i%7-3).toFloat/16),Shape.from(Seq(2L,2L,3L,3L)))
    val config=Conv2dConfig(padding=(1,1))
    val expected=Conv2dOps.forward(Resize2d.nearest(input,7,9),weight,config)
    CudaSession.withDevice() { s =>
      val x=s.upload(input);val w=s.upload(weight)
      val enlarged=s.resizeNearest2d(x,7,9)
      val result=s.conv2d(enlarged,w,config)
      enlarged.close();x.close();w.close()
      assertEquals(bits(result.toCpu),bits(expected))
      result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
