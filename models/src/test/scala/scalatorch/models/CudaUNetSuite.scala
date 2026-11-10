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
