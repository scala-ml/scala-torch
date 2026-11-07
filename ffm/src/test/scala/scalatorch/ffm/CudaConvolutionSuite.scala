package scalatorch.ffm

import scalatorch.*

class CudaConvolutionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"set SCALATORCH_CUDA_TESTS=1 for hardware tests")
  test("fast convolution handles batched grouped chunks with a constrained workspace") {
    gpu()
    for groups <- Vector(2,4) do
      val config=Conv2dConfig(groups=groups,padding=(1,2))
      val shape=Shape.from(Seq(2L,4L,7L,9L))
      val weightShape=Shape.from(Seq(8L,4L/groups,3L,2L))
      val random=new java.util.Random(1234)
      val input=Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
      val weights=Tensor.fromFloats(Array.fill(weightShape.numel.toInt)(random.nextGaussian().toFloat),weightShape)
      val expected=Conv2dOps.forward(input,weights,config)
      CudaSession.withDevice(maxBytes=8192) { session =>
        val x=session.upload(input);val w=session.upload(weights)
        val result=session.conv2dFast(x,w,config)
        assertEquals(result.shape,expected.shape)
        val actual=result.toCpu.toFloatArray
        val reference=expected.toFloatArray
        val differences=actual.zip(reference).map((a,b)=>math.abs(a.toDouble-b))
        val index=differences.indices.maxBy(differences(_))
        val error=differences(index)
        assert(error<1e-4,s"grouped chunked convolution max error $error at index $index, groups=$groups, actual=${actual(index)}, expected=${reference(index)}")
        result.close();w.close();x.close()
        assertEquals(session.allocatedBytes,0L)
      }
  }
  test("fast convolution dispatches the wider GEMM tile and handles all reduction tails") {
    gpu()
    assume(sys.env.get("SCALATORCH_CUDA_PROFILE").contains("1"),"set SCALATORCH_CUDA_PROFILE=1 to assert kernel dispatch")
    val config=Conv2dConfig(padding=(1,1))
    val shape=Shape.from(Seq(1L,7L,13L,15L))
    val weightShape=Shape.from(Seq(11L,7L,3L,2L))
    val random=new java.util.Random(721)
    val input=Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
    val weights=Tensor.fromFloats(Array.fill(weightShape.numel.toInt)(random.nextGaussian().toFloat),weightShape)
    val expected=Conv2dOps.forward(input,weights,config)
    val diagnostics=new java.io.ByteArrayOutputStream()
    val previous=System.err
    try
      System.setErr(new java.io.PrintStream(diagnostics,true,java.nio.charset.StandardCharsets.UTF_8))
      CudaSession.withDevice() { session =>
        val x=session.upload(input);val w=session.upload(weights)
        val result=session.conv2dFast(x,w,config)
        assertEquals(result.shape,expected.shape)
        val error=result.toCpu.toFloatArray.zip(expected.toFloatArray).map((a,b)=>math.abs(a.toDouble-b)).max
        assert(error<1e-4,s"wider tiled convolution max error $error")
        result.close();w.close();x.close()
        assertEquals(session.allocatedBytes,0L)
      }
    finally System.setErr(previous)
    assert(diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8).contains("gemm_conv_n32"),
      s"convolution did not dispatch gemm_conv_n32: ${diagnostics.toString(java.nio.charset.StandardCharsets.UTF_8)}")
  }
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
