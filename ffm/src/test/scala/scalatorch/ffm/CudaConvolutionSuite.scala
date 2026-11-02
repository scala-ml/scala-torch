package scalatorch.ffm

import scalatorch.*

class CudaConvolutionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"set SCALATORCH_CUDA_TESTS=1 for hardware tests")
  test("resident CUDA convolution matches CPU grouped padded strided and dilated reference") {
    gpu()
    CudaSession.withDevice() { session =>
      for config <- Vector(Conv2dConfig(),Conv2dConfig(padding=(1,2)),
          Conv2dConfig(stride=(2,3),padding=(2,1)),Conv2dConfig(dilation=(2,2),padding=(2,2)),
          Conv2dConfig(groups=2,padding=(1,1)),Conv2dConfig(groups=4,padding=(1,1))) do
        val shape=Shape.from(Seq(2L,4L,7L,9L))
        val weightShape=Shape.from(Seq(8L,4L/config.groups,3L,2L))
        val random=new java.util.Random(42)
        val input=Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
        val weights=Tensor.fromFloats(Array.fill(weightShape.numel.toInt)(random.nextGaussian().toFloat),weightShape)
        val expected=Conv2dOps.forward(input,weights,config)
        val x=session.upload(input);val w=session.upload(weights)
        val y=session.conv2d(x,w,config)
        assertEquals(y.shape,expected.shape)
        assertEquals(y.toCpu.toFloatArray.toVector,expected.toFloatArray.toVector)
        y.close();w.close();x.close()
        assertEquals(session.allocatedBytes,0L)
    }
  }
  test("invalid convolution geometry and allocation failure do not leak memory") {
    gpu()
    CudaSession.withDevice(maxBytes=128) { session =>
      val x=session.fill(Shape.from(Seq(1L,1L,3L,3L)),1f)
      val w=session.fill(Shape.from(Seq(1L,1L,1L,1L)),2f)
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](session.conv2d(x,w,Conv2dConfig(groups=2)))
      intercept[IllegalArgumentException](session.conv2d(x,w,Conv2dConfig(padding=(10,10))))
      assertEquals(session.allocatedBytes,before)
      w.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("convolution handles empty batches noncontiguous uploads and IEEE padding") {
    gpu()
    CudaSession.withDevice() { session =>
      val weights=Tensor.fromFloats(Array(2f),Shape.from(Seq(1L,1L,1L,1L)))
      for input <- Vector(
          Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0L,1L,3L,2L))),
          Tensor.fromFloats(Array(Float.NaN,Float.PositiveInfinity,-0f,1f,2f,3f),Shape.from(Seq(1L,1L,3L,2L))).transpose(2,3)) do
        val config=Conv2dConfig(padding=(1,1))
        val expected=Conv2dOps.forward(input,weights,config)
        val x=session.upload(input);val w=session.upload(weights);val result=session.conv2d(x,w,config)
        assertEquals(result.shape,expected.shape)
        result.toCpu.toFloatArray.zip(expected.toFloatArray).foreach { (actual,reference) =>
          if reference.isNaN then assert(actual.isNaN)
          else assertEquals(java.lang.Float.floatToRawIntBits(actual),java.lang.Float.floatToRawIntBits(reference))
        }
        result.close();w.close()
        intercept[IllegalStateException](session.conv2d(x,w))
        x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
