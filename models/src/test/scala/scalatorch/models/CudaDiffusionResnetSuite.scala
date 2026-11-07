package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession
import scalatorch.nn.*

class CudaDiffusionResnetSuite extends munit.FunSuite:
  private def gpu(): Unit=assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
  private def data(dims: Seq[Long],seed: Int): Tensor[Float] =
    val shape=Shape.from(dims);val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)((random.nextGaussian()*.1).toFloat),shape)
  private def conv(in: Int,out: Int,k: Int): Conv2d =
    Conv2d(Parameter(data(Seq(out,in,k,k).map(_.toLong),3)),Some(Parameter(data(Seq(out.toLong),4))),Conv2dConfig(padding=(k/2,k/2)))
  private def model(conditioned: Boolean): DiffusionResnet2d =
    DiffusionResnet2d(GroupNorm.init(2,1),conv(2,4,3),if conditioned then Some(Linear.init(3,4,5)) else None,
      GroupNorm.init(4,2),conv(4,4,3),Some(conv(2,4,1)),1.4)
  test("uploaded residual blocks match CPU with and without timestep conditioning") {
    gpu()
    for variant <- 0 until 3 do
      val conditioned=variant>0
      val original=model(conditioned)
      val cpu=if variant==2 then
        val projection=original.timeEmbProj.get.withLoRA(2,3.0,7)
        projection.lora.get.b.load(data(Seq(4L,2L),8))
        original.copy(timeEmbProj=Some(projection))
      else original
      val input=data(Seq(2L,2L,3L,5L),1)
      val time=if conditioned then Some(data(Seq(2L,3L),2)) else None
      val expected=Autograd.noGrad(cpu(Variable(input),time.map(Variable(_)))).data.toFloatArray
      CudaSession.withDevice() { s =>
        val resident=CudaDiffusionResnet.upload(s,cpu)
        val x=s.upload(input);val t=time.map(s.upload)
        val before=s.allocatedBytes
        for _ <- 0 until 2 do
          val out=resident(x,t)
          out.toCpu.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,3e-6))
          out.close()
          assertEquals(s.allocatedBytes,before)
        // Upload is a snapshot: later CPU parameter mutation cannot alter the resident block.
        cpu.conv2.bias.get.load(Tensor.fromFloats(Array.fill(4)(100f),Shape.from(Seq(4L))))
        val out=resident(x,t)
        resident.close()
        out.toCpu.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,3e-6))
        intercept[IllegalStateException](resident(x,t))
        out.close();t.foreach(_.close());x.close()
        assertEquals(s.allocatedBytes,0L)
      }
  }
  test("residuals loaded from VAE safetensors chain on GPU") {
    gpu()
    val configured=sys.props.get("scalatorch.vae.dir")
    assume(configured.nonEmpty,"set scalatorch.vae.dir to tiny-sd-pipe/vae")
    val directory=java.nio.file.Path.of(configured.get)
    val vae=PretrainedVae.load(directory)
    val first=vae.decoder.midBlock.first;val second=vae.decoder.midBlock.second
    val input=data(Seq(1L,first.norm1.channels.toLong,3L,5L),9)
    val expected=Autograd.noGrad(second(first(Variable(input)))).data.toFloatArray
    CudaSession.withDevice() { s =>
      val a=CudaDiffusionResnet.upload(s,first);val b=CudaDiffusionResnet.upload(s,second)
      val x=s.upload(input)
      val middle=a(x);x.close()
      val result=b(middle);middle.close()
      a.close();b.close()
      result.toCpu.toFloatArray.zip(expected).foreach((a,e) => assertEqualsDouble(a,e,5e-6))
      result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("residual upload and execution failures release temporary allocations") {
    gpu()
    val cpu=model(true)
    CudaSession.withDevice(maxBytes=100) { s =>
      intercept[IllegalArgumentException](CudaDiffusionResnet.upload(s,cpu))
      assertEquals(s.allocatedBytes,0L)
    }
    val weightBytes=Module.parameters(cpu).map(_.data.shape.numel*4).sum
    CudaSession.withDevice(maxBytes=weightBytes+256) { s =>
      val resident=CudaDiffusionResnet.upload(s,cpu)
      val x=s.fill(Shape.from(Seq(1L,2L,2L,2L)),1f)
      val t=s.fill(Shape.from(Seq(1L,3L)),1f)
      val before=s.allocatedBytes
      intercept[IllegalArgumentException](resident(x,None))
      assertEquals(s.allocatedBytes,before)
      intercept[IllegalArgumentException](resident(x,Some(t)))
      assertEquals(s.allocatedBytes,before)
      resident.close();t.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
