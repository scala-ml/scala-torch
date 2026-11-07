package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scalatorch.nn.*

class CudaDiffusionAttentionSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def data(dims: Seq[Long],seed: Int): Tensor[Float] =
    val shape=Shape.from(dims);val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)(random.nextGaussian().toFloat),shape)
  private def model(context: Int): DiffusionAttention =
    DiffusionAttention(Linear.init(6,8,1),Linear.init(context,8,2,bias=false),Linear.init(context,8,3),Linear.init(8,6,4),2)
  test("projected resident attention matches CPU self and unequal-length cross attention") {
    gpu()
    for cross <- Seq(false,true) do
      val cpu=model(if cross then 5 else 6)
      val x=data(Seq(2L,3L,6L),1)
      val context=if cross then Some(data(Seq(2L,7L,5L),2)) else None
      val expected=Autograd.noGrad(cpu(Variable(x),context.map(Variable(_)))).data
      CudaSession.withDevice() { s =>
        val runtime=CudaDiffusionRuntime.upload(s,cpu)
        val dx=s.upload(x);val dc=context.map(s.upload)
        val before=s.allocatedBytes
        for _ <- 0 until 2 do
          val result=runtime.attention(dx,cpu,dc)
          assertEquals(result.shape,expected.shape)
          result.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,4e-6))
          result.close()
          assertEquals(s.allocatedBytes,before)
        val result=runtime.attention(dx,cpu,dc)
        runtime.close();dx.close();dc.foreach(_.close())
        assertEquals(result.toCpu.shape,expected.shape)
        result.close()
        assertEquals(s.allocatedBytes,0L)
      }
  }
  test("attention validates shapes and score budget before allocating projections") {
    gpu()
    val cpu=model(6).copy(maxScoreElements=1)
    CudaSession.withDevice() { s =>
      val runtime=CudaDiffusionRuntime.upload(s,cpu)
      val x=s.upload(data(Seq(2L,3L,6L),1))
      val before=s.allocatedBytes
      intercept[IllegalArgumentException](runtime.attention(x,cpu,None))
      assertEquals(s.allocatedBytes,before)
      val invalid=s.fill(Shape.from(Seq(2L,3L,5L)),0f)
      val withInvalid=s.allocatedBytes
      intercept[IllegalArgumentException](runtime.attention(invalid,cpu,None))
      assertEquals(s.allocatedBytes,withInvalid)
      invalid.close()
      runtime.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
    val normal=model(6)
    val weightBytes=Module.parameters(normal).map(_.data.shape.numel*4).sum
    CudaSession.withDevice(maxBytes=weightBytes+144+500) { s =>
      val runtime=CudaDiffusionRuntime.upload(s,normal)
      val x=s.upload(data(Seq(2L,3L,6L),1))
      val before=s.allocatedBytes
      intercept[IllegalArgumentException](runtime.attention(x,normal,None))
      assertEquals(s.allocatedBytes,before)
      runtime.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("loaded VAE middle block runs residual attention residual entirely on GPU") {
    gpu()
    val configured=sys.props.get("scalatorch.vae.dir")
    assume(configured.nonEmpty,"set scalatorch.vae.dir to tiny-sd-pipe/vae")
    val vae=PretrainedVae.load(java.nio.file.Path.of(configured.get))
    val mid=vae.decoder.midBlock
    val x=data(Seq(1L,mid.first.norm1.channels.toLong,3L,5L),8)
    val expected=Autograd.noGrad(mid(Variable(x))).data
    CudaSession.withDevice() { s =>
      val runtime=CudaDiffusionRuntime.upload(s,mid)
      val input=s.upload(x)
      val a=runtime.resnet(input,mid.first,None)
      val b=runtime.vaeAttention(a,mid.attention.get);a.close()
      val result=runtime.resnet(b,mid.second,None);b.close()
      input.close();runtime.close()
      assertEquals(result.shape,expected.shape)
      result.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-5))
      result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
