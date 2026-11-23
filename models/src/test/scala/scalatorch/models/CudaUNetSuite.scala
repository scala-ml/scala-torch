package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scalatorch.nn.Module
import java.nio.{ByteBuffer,ByteOrder}

class CudaUNetSuite extends munit.FunSuite:
  private def loaded(): ConditionalUNet2d =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val dir=sys.props.get("scalatorch.unet.dir")
    assume(dir.nonEmpty,"public tiny U-Net required")
    PretrainedUNet.load(java.nio.file.Path.of(dir.get))
  private def input(batch: Int,h: Int,w: Int)=
    Tensor.fromFloats(Array.tabulate(batch*4*h*w)(i => math.sin(i*.17).toFloat),Shape.from(Seq(batch,4,h,w).map(_.toLong)))
  private def context(batch: Int)=
    Tensor.fromFloats(Array.tabulate(batch*3*32)(i => math.cos(i*.13).toFloat),Shape.from(Seq(batch.toLong,3L,32L)))
  private def times(values: Float*)=Tensor.fromFloats(values.toArray,Shape.from(Seq(values.size.toLong)))
  private def compare(actual: Tensor[Float],expected: Array[Float]): Unit =
    val values=actual.toFloatArray
    assertEquals(values.length,expected.length)
    val error=values.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
    println(s"CUDA complete U-Net maxAbsError=$error")
    assert(error<3e-5)
  test("complete resident U-Net matches independent Candle references at odd and even resolutions") {
    val cpu=loaded()
    CudaSession.withDevice() { s =>
      val model=CudaUNet.upload(s,cpu)
      val c=s.upload(context(1));val before=s.allocatedBytes
      for (h,w,t) <- Seq((3,5,7.5f),(4,4,0f),(4,6,19f)) do
        val x=s.upload(input(1,h,w));val out=model(x,times(t),c)
        assertEquals(out.shape,x.shape)
        val stream=getClass.getResourceAsStream(s"/tiny-sd/output-${h}x$w-f32.bin")
        val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
        compare(out.toCpu,Array.fill(bytes.remaining()/4)(bytes.getFloat()))
        out.close();x.close()
        assertEquals(s.allocatedBytes,before)
      model.close();c.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("batched timesteps and explicit bottom-right downsampling preserve CPU behavior") {
    val original=loaded()
    val first=original.downBlocks.head
    val down=first.downsample.get
    val unpadded=down.copy(conv=down.conv.copy(config=Conv2dConfig(stride=(2,2))))
    val asymmetric=original.copy(downBlocks=original.downBlocks.updated(0,first.copy(downsample=Some(unpadded))))
    val x=input(2,3,5);val c=context(2)
    for cpu <- Seq(original,asymmetric); t <- Seq(times(7.5f),times(7.5f,123f)) do
      val expected=Autograd.noGrad(cpu(Variable(x),t,Variable(c))).data
      CudaSession.withDevice() { s =>
        val model=CudaUNet.upload(s,cpu)
        val dx=s.upload(x);val dc=s.upload(c)
        val result=model(dx,t,dc)
        model.close();dx.close();dc.close()
        assertEquals(result.shape,expected.shape)
        compare(result.toCpu,expected.toFloatArray)
        result.close()
        assertEquals(s.allocatedBytes,0L)
      }
  }
  test("invalid conditioning and cancellation release temporaries while preserving the model") {
    val cpu=loaded()
    CudaSession.withDevice() { s =>
      var armed=false;var checks=0
      val model=CudaUNet.upload(s,cpu,() => { if armed then checks+=1;armed && checks>=20 })
      val x=s.upload(input(1,3,5));val c=s.upload(context(1));val before=s.allocatedBytes
      intercept[IllegalArgumentException](model(x,times(Float.NaN),c))
      intercept[IllegalArgumentException](model(x,times(1f,2f),c))
      intercept[IllegalArgumentException](model(x,times(1f),x))
      assertEquals(s.allocatedBytes,before)
      armed=true
      intercept[java.util.concurrent.CancellationException](model(x,times(7.5f),c))
      assertEquals(s.allocatedBytes,before)
      armed=false
      val output=model(x,times(7.5f),c)
      assert(output.toCpu.toFloatArray.forall(_.isFinite))
      output.close();model.close();c.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
    val weightBytes=Module.parameters(cpu).map(_.data.shape.numel*4).sum
    CudaSession.withDevice(maxBytes=weightBytes+100000) { s =>
      val model=CudaUNet.upload(s,cpu)
      val x=s.upload(input(1,3,5));val c=s.upload(context(1));val before=s.allocatedBytes
      intercept[IllegalArgumentException](model(x,times(7.5f),c))
      assertEquals(s.allocatedBytes,before)
      model.close();c.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("a scheduled guidance on the batched path matches the CPU predictor; at exactly 1 it is the positive prediction alone") {
    val cpu=loaded()
    val positive=context(1);val negative=Tensor.fromFloats(Array.tabulate(3*32)(i => math.sin(i*.31).toFloat),Shape.from(Seq(1L,3L,32L)))
    val sample=input(1,4,6)
    def worst(a: Tensor[Float],b: Tensor[Float]): Double = a.toFloatArray.zip(b.toFloatArray).map((x,y) => math.abs(x.toDouble-y)).max
    for target <- Seq(1f,3f,0.5f,7.5f) do
      val expected=UNetExecution.Cpu.withPredictor(cpu,positive,Some(negative),() => false)(_.apply(sample,times(123f),7.5f,scheduled=Some(target)))
      val actual=UNetExecution.Cuda().withPredictor(cpu,positive,Some(negative),() => false)(_.apply(sample,times(123f),7.5f,scheduled=Some(target)))
      assertEquals(actual.shape,expected.shape)
      val error=worst(actual,expected)
      println(s"CUDA scheduled guidance target=$target maxAbsError=$error")
      assert(error<1e-4)
    // guidance 1 needs no negative pass: the result equals a run without a negative prompt
    val alone=UNetExecution.Cuda().withPredictor(cpu,positive,None,() => false)(_.apply(sample,times(123f),1f))
    val scheduled=UNetExecution.Cuda().withPredictor(cpu,positive,Some(negative),() => false)(_.apply(sample,times(123f),7.5f,scheduled=Some(1f)))
    assert(worst(alone,scheduled)<1e-6)
  }

  test("classifier-free guidance as one batch of two matches the CPU predictor") {
    val cpu=loaded()
    val positive=context(1);val negative=Tensor.fromFloats(Array.tabulate(3*32)(i => math.sin(i*.31).toFloat),Shape.from(Seq(1L,3L,32L)))
    for guidance <- Seq(1.5f,7.5f); t <- Seq(7.5f,123f) do
      val sample=input(1,4,6)
      val expected=UNetExecution.Cpu.withPredictor(cpu,positive,Some(negative),() => false)(_.apply(sample,times(t),guidance))
      val actual=UNetExecution.Cuda().withPredictor(cpu,positive,Some(negative),() => false)(_.apply(sample,times(t),guidance))
      assertEquals(actual.shape,expected.shape)
      val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
      println(s"CUDA batched guidance g=$guidance t=$t maxAbsError=$error")
      assert(error<1e-4)
  }
  /** Float32 kernels for a comparison, whatever precision an earlier suite of this test JVM left selected. */
  private def float32[A](body: => A): A =
    val previous=CudaDiffusionRuntime.fp16Default
    CudaDiffusionRuntime.setHalfPrecision(Some(false))
    try body finally CudaDiffusionRuntime.setHalfPrecision(Some(previous))
  test("a sample batch of three with one prompt matches three single-sample predictor calls") { float32 {
    val cpu=loaded()
    val positive=context(1);val negative=Tensor.fromFloats(Array.tabulate(3*32)(i => math.sin(i*.31).toFloat),Shape.from(Seq(1L,3L,32L)))
    val samples=input(3,4,6)
    def single(index: Int)=samples.slice(0,index,index+1).contiguous
    // guidance above one: one U-Net batch of six; guidance one without a negative prompt: one batch of three
    for (guidance,unconditional) <- Seq((7.5f,Some(negative)),(1.5f,Some(negative)),(1f,None)); t <- Seq(7.5f,123f) do
      UNetExecution.Cuda().withPredictor(cpu,positive,unconditional,() => false) { predict =>
        val batched=predict(samples,times(t),guidance)
        assertEquals(batched.shape,samples.shape)
        val expected=(0 until 3).flatMap(i => predict(single(i),times(t),guidance).toFloatArray).toArray
        val error=batched.toFloatArray.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
        println(s"CUDA batched sampling g=$guidance t=$t maxAbsError=$error")
        assert(error<1e-4)
        // the three samples are different images, not one repeated
        assert(expected.take(expected.length/3).zip(expected.drop(expected.length/3)).exists((a,b) => math.abs(a-b)>1e-3))
      }
  } }
  test("a sample batch that does not fit reports the memory failure to the caller") {
    val cpu=loaded()
    val negative=Tensor.fromFloats(Array.tabulate(3*32)(i => math.sin(i*.31).toFloat),Shape.from(Seq(1L,3L,32L)))
    def run(budget: Long,batch: Int)=UNetExecution.Cuda(maxBytes=budget).withPredictor(cpu,context(1),Some(negative),() => false)(
      _.apply(input(batch,4,6),times(7.5f),7.5f))
    // the smallest power-of-two budget that serves one image
    val budget=Iterator.iterate(1L<<16)(_*2).takeWhile(_<=(1L<<32)).find { bytes =>
      try { run(bytes,1);true } catch case error: Throwable if CudaUNet.memoryFailure(error) => false
    }.get
    val error=intercept[Throwable](run(budget,1024))
    assert(CudaUNet.memoryFailure(error),error.toString)
    assert(run(budget,1).toFloatArray.forall(_.isFinite))
  }
  test("a sample batch falls back to the caller when the step is not plain") {
    val cpu=loaded()
    val negative=Tensor.fromFloats(Array.tabulate(3*32)(i => math.sin(i*.31).toFloat),Shape.from(Seq(1L,3L,32L)))
    UNetExecution.Cuda().withPredictor(cpu,context(1),Some(negative),() => false) { predict =>
      intercept[BatchedSamplingUnsupported](predict(input(3,4,6),times(7.5f),7.5f,rescale=0.5f))
      // the single-image general path still serves the same step
      assert(predict(input(1,4,6),times(7.5f),7.5f,rescale=0.5f).toFloatArray.forall(_.isFinite))
    }
    // A CFG schedule alone does not leave the batched path (without a negative prompt it has nothing to scale).
    val plainBatch=UNetExecution.Cuda(maxBytes=1L<<30).withPredictor(cpu,context(1),None,() => false)(_.apply(input(2,4,6),times(7.5f),1f))
    val scheduledBatch=UNetExecution.Cuda(maxBytes=1L<<30).withPredictor(cpu,context(1),None,() => false)(
      _.apply(input(2,4,6),times(7.5f),1f,scheduled=Some(2f)))
    assertEquals(scheduledBatch.shape,plainBatch.shape)
    assert(scheduledBatch.toFloatArray.zip(plainBatch.toFloatArray).forall((a,b) => a==b))
  }
