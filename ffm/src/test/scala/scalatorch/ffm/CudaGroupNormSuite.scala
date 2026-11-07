package scalatorch.ffm

import scalatorch.*

class CudaGroupNormSuite extends munit.FunSuite:
  test("large groups parallelize output while preserving reference reduction and affine") {
    gpu()
    val dims=Shape.from(Seq(2L,4L,64L,64L))
    val values=Array.tabulate(dims.numel.toInt)(i=>(math.sin(i*.017)*3+math.cos(i*.003)).toFloat)
    val input=Tensor.fromFloats(values,dims)
    val weights=Array(1.1f,-.4f,2f,.25f);val biases=Array(.2f,-.1f,1f,-2f)
    val reference=GroupNormOps.forward(input,2).toFloatArray
    CudaSession.withDevice() { session =>
      val x=session.upload(input);val w=session.upload(Tensor.fromFloats(weights,Shape.from(Seq(4L))))
      val b=session.upload(Tensor.fromFloats(biases,Shape.from(Seq(4L))))
      val output=session.groupNorm(x,2,weight=Some(w),bias=Some(b))
      output.toCpu.toFloatArray.zipWithIndex.foreach { (actual,i) =>
        val channel=(i/4096)%4
        val expected=reference(i)*weights(channel)+biases(channel)
        assertEquals(java.lang.Float.floatToRawIntBits(actual),java.lang.Float.floatToRawIntBits(expected))
      }
      output.close();b.close();w.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("large GroupNorm falls back when its statistics scratch does not fit") {
    gpu()
    val shape=Shape.from(Seq(1L,2L,64L,64L))
    val input=Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i=>math.sin(i*.01).toFloat),shape)
    val expected=GroupNormOps.forward(input,1).toFloatArray
    val bytes=shape.numel*4
    CudaSession.withDevice(maxBytes=2*bytes) { session =>
      val x=session.upload(input)
      val output=session.groupNorm(x,1)
      assertEquals(session.allocatedBytes,2*bytes)
      assertEquals(output.toCpu.toFloatArray.toVector.map(java.lang.Float.floatToRawIntBits),
        expected.toVector.map(java.lang.Float.floatToRawIntBits))
      output.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  test("resident GroupNorm matches CPU normalization and separate affine rounding") {
    gpu()
    CudaSession.withDevice() { session =>
      for dims <- Vector(Seq(2L,4L),Seq(2L,4L,3L,5L),Seq(0L,4L,2L)); groups <- Vector(1,2,4) do
        val shape=Shape.from(dims)
        val random=new java.util.Random(42)
        val input=Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
        val weight=Array(0.5f,-1.5f,2f,0.1f);val bias=Array(1f,0f,-2f,3f)
        val x=session.upload(input)
        val w=session.upload(Tensor.fromFloats(weight,Shape.from(Seq(4L))))
        val b=session.upload(Tensor.fromFloats(bias,Shape.from(Seq(4L))))
        val spatial=dims.drop(2).product.toInt
        val normalized=GroupNormOps.forward(input,groups).toFloatArray
        for (useWeight,useBias) <- Vector((false,false),(true,false),(false,true),(true,true)) do
          val result=session.groupNorm(x,groups,weight=Option.when(useWeight)(w),bias=Option.when(useBias)(b))
          val expected=normalized.indices.map { i =>
            val channel=(i/spatial)%4
            val scaled=if useWeight then normalized(i)*weight(channel) else normalized(i)
            if useBias then scaled+bias(channel) else scaled
          }.toVector
          assertEquals(result.shape,shape)
          assertEquals(result.toCpu.toFloatArray.toVector,expected)
          result.close()
        b.close();w.close();x.close()
        assertEquals(session.allocatedBytes,0L)
    }
  }
  test("GroupNorm noncontiguous and extreme inputs preserve CPU IEEE behavior") {
    gpu()
    CudaSession.withDevice() { session =>
      for values <- Vector(Array(1e8f,1f,-1e8f,2f,3f,4f,5f,6f),Array.fill(8)(Float.MaxValue),
          Array(Float.NaN,1f,2f,3f,Float.PositiveInfinity,Float.NegativeInfinity,0f,-0f)) do
        val input=Tensor.fromFloats(values,Shape.from(Seq(1L,2L,2L,2L))).transpose(2,3)
        val x=session.upload(input);val y=session.groupNorm(x,2,epsilon=1e-6)
        y.toCpu.toFloatArray.zip(GroupNormOps.forward(input,2,1e-6).toFloatArray).foreach { (a,e) =>
          if e.isNaN then assert(a.isNaN) else assertEquals(java.lang.Float.floatToRawIntBits(a),java.lang.Float.floatToRawIntBits(e))
        }
        y.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("GroupNorm rejects invalid geometry affine parameters and budgets without leaks") {
    gpu()
    CudaSession.withDevice(maxBytes=32) { session =>
      val x=session.fill(Shape.from(Seq(1L,2L,2L,2L)),1f)
      for groups <- Vector(0,3) do intercept[IllegalArgumentException](session.groupNorm(x,groups))
      for epsilon <- Vector(0.0,Double.NaN,Double.PositiveInfinity) do intercept[IllegalArgumentException](session.groupNorm(x,1,epsilon))
      intercept[IllegalArgumentException](session.groupNorm(x,1,weight=Some(x)))
      intercept[IllegalArgumentException](session.groupNorm(x,1))
      assertEquals(session.allocatedBytes,32L)
      x.close()
      intercept[IllegalStateException](session.groupNorm(x,1))
      assertEquals(session.allocatedBytes,0L)
    }
  }
  test("LayerNorm retains its output after temporary views close and matches final-axis reference") {
    gpu()
    CudaSession.withDevice() { session =>
      for dims <- Vector(Seq(4L),Seq(2L,3L,4L),Seq(0L,4L)) do
        val shape=Shape.from(dims)
        val input=Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => (i%7-3).toFloat),shape)
        val x=session.upload(input)
        val bias=session.fill(Shape.from(Seq(4L)),0.25f)
        val result=session.layerNorm(x,bias=Some(bias))
        val expected=GroupNormOps.forward(input.reshape(Shape.from(Seq(shape.numel/4,4L))),1).toFloatArray.map(_+0.25f)
        assertEquals(result.shape,shape)
        assertEquals(result.toCpu.toFloatArray.toVector,expected.toVector)
        result.close();bias.close();x.close()
        assertEquals(session.allocatedBytes,0L)
      val scalar=session.fill(Shape.from(Seq.empty),1f)
      intercept[IllegalArgumentException](session.layerNorm(scalar))
      scalar.close()
    }
  }
  test("convolution GroupNorm SiLU chain keeps activations resident") {
    gpu()
    val random=new java.util.Random(19)
    val shape=Shape.from(Seq(1L,4L,5L,5L))
    val weightShape=Shape.from(Seq(4L,4L,3L,3L))
    val input=Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
    val weights=Tensor.fromFloats(Array.fill(weightShape.numel.toInt)(random.nextGaussian().toFloat),weightShape)
    val config=Conv2dConfig(padding=(1,1))
    val expected=scalatorch.cpu.CpuFloatBackend.silu(GroupNormOps.forward(Conv2dOps.forward(input,weights,config),2)).toFloatArray
    CudaSession.withDevice() { session =>
      val x=session.upload(input);val w=session.upload(weights)
      val convolution=session.conv2d(x,w,config)
      val normalized=session.groupNorm(convolution,2)
      convolution.close()
      val activated=session.silu(normalized)
      normalized.close()
      activated.toCpu.toFloatArray.zip(expected).foreach { (a,e) =>
        assertEqualsDouble(a.toDouble,e.toDouble,2.0*math.ulp(e).toDouble)
      }
      activated.close();w.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
