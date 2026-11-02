package scalatorch.models

import scalatorch.*
import scalatorch.ffm.CudaSession
import scalatorch.nn.Module
import java.nio.{ByteBuffer,ByteOrder}

class CudaVaeDecoderSuite extends munit.FunSuite:
  private def loaded(): VaeModel =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val directory=sys.props.get("scalatorch.vae.dir")
    assume(directory.nonEmpty,"set scalatorch.vae.dir to tiny-sd-pipe/vae")
    PretrainedVae.load(java.nio.file.Path.of(directory.get))
  test("complete resident VAE decoder matches independent Candle image references") {
    val model=loaded()
    CudaSession.withDevice() { s =>
      val decoder=CudaVaeDecoder.upload(s,model.decoder)
      val weights=s.allocatedBytes
      for (b,h,w) <- Seq((1,3,5),(1,4,4),(2,3,5)) do
        val input=Tensor.fromFloats(Array.tabulate(b*4*h*w)(i => math.cos(i*.17).toFloat),Shape.from(Seq(b,4,h,w).map(_.toLong)))
        val x=s.upload(input)
        val result=decoder(x)
        x.close()
        val stream=getClass.getResourceAsStream(s"/tiny-vae/decode-b$b-${h}x$w-f32.bin")
        val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
        val expected=Array.fill(bytes.remaining()/4)(bytes.getFloat())
        assertEquals(result.shape,Shape.from(Seq(b,3,h*2,w*2).map(_.toLong)))
        val actual=result.toCpu.toFloatArray
        assertEquals(actual.length,expected.length)
        val max=actual.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
        println(s"CUDA VAE decode b$b ${h}x$w maxAbsError=$max")
        assert(max<4e-5,s"Candle VAE discrepancy: $max")
        result.close()
        assertEquals(s.allocatedBytes,weights)
      decoder.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("decoder rejects invalid geometry and cleans failed inference without discarding weights") {
    val model=loaded()
    val bytes=Module.parameters(model.decoder).map(_.data.shape.numel*4).sum
    CudaSession.withDevice(maxBytes=bytes+1024) { s =>
      val decoder=CudaVaeDecoder.upload(s,model.decoder)
      val x=s.fill(Shape.from(Seq(1L,4L,3L,5L)),1f)
      val bad=s.fill(Shape.from(Seq(1L,3L,3L,5L)),1f)
      val before=s.allocatedBytes
      intercept[IllegalArgumentException](decoder(bad))
      assertEquals(s.allocatedBytes,before)
      intercept[IllegalArgumentException](decoder(x))
      assertEquals(s.allocatedBytes,before)
      decoder.close()
      intercept[IllegalStateException](decoder(x))
      x.close();bad.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("decoder supports omitted post-quant convolution and middle attention") {
    val base=loaded().decoder
    val model=base.copy(postQuantConv=None,midBlock=base.midBlock.copy(attention=None))
    val input=Tensor.fromFloats(Array.tabulate(60)(i => math.sin(i*.1).toFloat),Shape.from(Seq(1L,4L,3L,5L)))
    val expected=scalatorch.autograd.Autograd.noGrad(model(scalatorch.autograd.Variable(input))).data
    CudaSession.withDevice() { s =>
      val decoder=CudaVaeDecoder.upload(s,model)
      val x=s.upload(input)
      val result=decoder(x)
      decoder.close();x.close()
      assertEquals(result.shape,expected.shape)
      result.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-5))
      result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
