package scalatorch.models

import scalatorch.*
import scalatorch.ffm.CudaSession
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.Path
import scala.util.Using

class CudaSdxlUNetSuite extends munit.FunSuite:
  private def model(): ConditionalUNet2d =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"GPU required")
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL required")
    PretrainedUNet.load(Path.of(directory.get).resolve("unet"))
  private def tensor(name: String,dims: Long*): Tensor[Float] =
    val stream=getClass.getResourceAsStream(s"/tiny-sdxl/$name.bin")
    val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
    Tensor.fromFloats(Array.fill(bytes.remaining()/4)(bytes.getFloat()),Shape.from(dims))
  private val time=Tensor.fromFloats(Array(500f),Shape.from(Seq(1L)))
  private def closeTo(actual: Tensor[Float],expected: Tensor[Float]): Unit =
    assertEquals(actual.shape,expected.shape)
    val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
    println(s"CUDA SDXL maxAbsError=$error")
    assert(error<2e-5)

  test("resident and offloaded SDXL CUDA match independent conditioned oracle") {
    val loaded=model()
    for offload <- Seq(false,true); batch <- Seq(1L,2L) do
      val suffix=if batch==1 then "" else "-b2"
      val source=tensor("sample"+suffix,batch,4,8,8)
      val context=tensor("context"+suffix,batch,6,64)
      val added=DiffusionAddedConditioning(tensor("pooled"+suffix,batch,32),tensor("time-ids"+suffix,batch,6))
      CudaSession.withDevice() { session =>
        Using.Manager { use =>
          val gpu=use(CudaUNet.upload(session,loaded,offloadWeights=offload))
          val x=use(session.upload(source));val c=use(session.upload(context))
          val before=session.allocatedBytes
          intercept[IllegalArgumentException](gpu(x,time,c))
          assertEquals(session.allocatedBytes,before)
          val out=use(gpu(x,time,c,Some(added)))
          closeTo(out.toCpu,tensor("unet-output"+suffix,batch,4,8,8))
          out.close()
          assertEquals(session.allocatedBytes,before)
        }.get
        assertEquals(session.allocatedBytes,0L)
      }
  }

  test("CFG predictor retains distinct positive and negative SDXL additions across calls") {
    val loaded=model()
    val sample=tensor("sample",1,4,8,8)
    val context=tensor("context",1,6,64)
    val positive=DiffusionAddedConditioning(tensor("pooled",1,32),tensor("time-ids",1,6))
    val negative=DiffusionAddedConditioning(tensor("pooled-b2",2,32).slice(0,1,2),tensor("time-ids-b2",2,6).slice(0,1,2))
    val expected=UNetExecution.Cpu.withPredictor(loaded,context,Some(context),() => false,
      Some(positive),Some(negative))(_.apply(sample,time,7.5f))
    val rescaled=UNetExecution.Cpu.withPredictor(loaded,context,Some(context),() => false,
      Some(positive),Some(negative))(_.apply(sample,time,7.5f,0,.7f))
    val scheduled=UNetExecution.Cpu.withPredictor(loaded,context,Some(context),() => false,
      Some(positive),Some(negative))(_.apply(sample,time,7.5f,0,.7f,Some(4f)))
    val detail=DetailDaemon.Adjustment("cond",1.1f,450f)
    val detailed=UNetExecution.Cpu.withPredictor(loaded,context,Some(context),() => false,
      Some(positive),Some(negative))(_.apply(sample,time,7.5f,0,.7f,Some(4f),Some(detail)))
    UNetExecution.Cuda(offloadWeights=true).withPredictor(loaded,context,Some(context),() => false,
      Some(positive),Some(negative)) { predictor =>
      closeTo(predictor(sample,time,7.5f),expected)
      closeTo(predictor(sample,time,7.5f),expected)
      closeTo(predictor(sample,time,7.5f,0,.7f),rescaled)
      closeTo(predictor(sample,time,7.5f,0,.7f,Some(4f)),scheduled)
      closeTo(predictor(sample,time,7.5f,0,.7f,Some(4f),Some(detail)),detailed)
    }
  }

  test("SDXL additions clean failed and cancelled GPU work without consuming inputs") {
    val loaded=model()
    val source=tensor("sample",1,4,8,8);val context=tensor("context",1,6,64)
    val added=DiffusionAddedConditioning(tensor("pooled",1,32),tensor("time-ids",1,6))
    for budget <- Seq(8192L,16777216L) do
      CudaSession.withDevice(maxBytes=budget) { session =>
        var armed=false;var checks=0
        val gpu=CudaUNet.upload(session,loaded,() => {if armed then checks+=1;armed && checks>=10},offloadWeights=true)
        val x=session.upload(source);val c=session.upload(context)
        val scalar=session.fill(Shape.from(Seq.empty),0f)
        intercept[IllegalArgumentException](gpu(scalar,time,c,Some(added)))
        scalar.close()
        val before=session.allocatedBytes
        val wrong=added.copy(timeIds=Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))))
        intercept[IllegalArgumentException](gpu(x,time,c,Some(wrong)))
        assertEquals(session.allocatedBytes,before)
        if budget==8192L then intercept[IllegalArgumentException](gpu(x,time,c,Some(added)))
        else
          armed=true
          intercept[java.util.concurrent.CancellationException](gpu(x,time,c,Some(added)))
          armed=false
        assertEquals(session.allocatedBytes,before)
        assertEquals(x.toCpu.toFloatArray.toVector,source.toFloatArray.toVector)
        if budget>8192 then
          val out=gpu(x,time,c,Some(added));out.close()
          assertEquals(session.allocatedBytes,before)
        gpu.close();x.close();c.close()
        assertEquals(session.allocatedBytes,0L)
      }
  }
