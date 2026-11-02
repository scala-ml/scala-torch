package scalatorch.models

import scalatorch.*
import scalatorch.ffm.CudaSession
import java.nio.{ByteBuffer,ByteOrder}

class CudaVaeEncoderSuite extends munit.FunSuite:
  private def model(): VaeEncoder =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val directory=sys.props.get("scalatorch.vae.dir")
    assume(directory.nonEmpty,"tiny VAE required")
    PretrainedVae.load(java.nio.file.Path.of(directory.get)).encoder

  test("resident and offloaded encoder moments match independent Candle references") {
    val encoder=model()
    for offload <- Seq(false,true) do
      CudaSession.withDevice() { session =>
        val gpu=CudaVaeEncoder.upload(session,encoder,offloadWeights=offload)
        val baseline=session.allocatedBytes
        for (b,h,w) <- Seq((1,6,10),(1,7,11),(2,8,8)) do
          val input=session.upload(Tensor.fromFloats(Array.tabulate(b*3*h*w)(i => math.sin(i*.13).toFloat),
            Shape.from(Seq(b,3,h,w).map(_.toLong))))
          val output=gpu(input)
          input.close()
          val stream=getClass.getResourceAsStream(s"/tiny-vae/encode-b$b-${h}x$w-f32.bin")
          val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
          val expected=Array.fill(bytes.remaining()/4)(bytes.getFloat())
          val actual=output.toCpu.toFloatArray
          assertEquals(output.shape,Shape.from(Seq(b,8,h/2,w/2).map(_.toLong)))
          assertEquals(actual.length,expected.length)
          val error=actual.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
          println(s"CUDA VAE encode offload=$offload b$b ${h}x$w maxAbsError=$error")
          assert(error<2e-5)
          output.close()
          assertEquals(session.allocatedBytes,baseline)
        gpu.close()
        assertEquals(session.allocatedBytes,0L)
      }
  }

  test("encoder failure and cancellation release temporaries and allow reuse") {
    val encoder=model()
    CudaSession.withDevice(maxBytes=2*1024*1024) { session =>
      var armed=false
      var calls=0
      val gpu=CudaVaeEncoder.upload(session,encoder,() => {if armed then calls+=1;armed && calls>=10},offloadWeights=true)
      val input=session.fill(Shape.from(Seq(1L,3L,6L,10L)),.5f)
      val bad=session.fill(Shape.from(Seq(1L,4L,6L,10L)),.5f)
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](gpu(bad))
      assertEquals(session.allocatedBytes,before)
      armed=true
      intercept[java.util.concurrent.CancellationException](gpu(input))
      assertEquals(session.allocatedBytes,before)
      armed=false
      val result=gpu(input)
      assert(result.toCpu.toFloatArray.forall(_.isFinite))
      result.close();gpu.close()
      intercept[IllegalStateException](gpu(input))
      input.close();bad.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }

  test("encoder handles omitted quant projection and attention and bounded allocation failure") {
    val base=model()
    val encoder=base.copy(quantConv=None,midBlock=base.midBlock.copy(attention=None))
    val input=Tensor.fromFloats(Array.tabulate(3*7*11)(i => math.sin(i*.13).toFloat),Shape.from(Seq(1L,3L,7L,11L)))
    val expected=VaeExecution.Cpu.encode(encoder,input,() => false)
    val actual=VaeExecution.Cuda(offloadWeights=true).encode(encoder,input,() => false)
    assertEquals(actual.shape,expected.shape)
    actual.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-5))
    CudaSession.withDevice(maxBytes=8192) { session =>
      val gpu=CudaVaeEncoder.upload(session,encoder,offloadWeights=true)
      val x=session.upload(input)
      val before=session.allocatedBytes
      intercept[IllegalArgumentException](gpu(x))
      assertEquals(session.allocatedBytes,before)
      gpu.close();x.close()
      assertEquals(session.allocatedBytes,0L)
    }
  }
