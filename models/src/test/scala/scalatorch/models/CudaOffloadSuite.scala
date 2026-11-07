package scalatorch.models

import scalatorch.*
import scalatorch.ffm.CudaSession
import scalatorch.nn.Module

class CudaOffloadSuite extends munit.FunSuite:
  test("offloaded VAE preserves resident output with less VRAM than its weights") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val directory=sys.props.get("scalatorch.vae.dir")
    assume(directory.nonEmpty,"public tiny VAE required")
    val model=PretrainedVae.load(java.nio.file.Path.of(directory.get)).decoder
    val parameters=Module.parameters(model)
    val total=parameters.map(_.data.shape.numel*4).sum
    val budget=parameters.map(_.data.shape.numel*4).max+1024*1024L
    assert(budget<total)
    val input=Tensor.fromFloats(Array.tabulate(60)(i => math.sin(i*.1).toFloat),Shape.from(Seq(1L,4L,3L,5L)))
    val expected=VaeExecution.Cuda().decode(model,input,() => false).toFloatArray
    CudaSession.withDevice(maxBytes=budget) { s =>
      intercept[IllegalArgumentException](CudaVaeDecoder.upload(s,model))
      assertEquals(s.allocatedBytes,0L)
      var cancel=false
      var checks=0
      val decoder=CudaVaeDecoder.upload(s,model,() => { checks+=1;cancel && checks>10 },offloadWeights=true)
      assertEquals(s.allocatedBytes,0L)
      val x=s.upload(input)
      checks=0;cancel=true
      intercept[java.util.concurrent.CancellationException](decoder(x))
      assertEquals(s.allocatedBytes,input.shape.numel*4)
      cancel=false
      val result=decoder(x)
      assertEquals(result.toCpu.toFloatArray.toVector,expected.toVector)
      result.close()
      assert(s.allocatedBytes>=input.shape.numel*4)
      assert(s.allocatedBytes<=budget)
      decoder.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
    CudaSession.withDevice(maxBytes=1024) { s =>
      val decoder=CudaVaeDecoder.upload(s,model,offloadWeights=true)
      val x=s.upload(input)
      intercept[IllegalArgumentException](decoder(x))
      assertEquals(s.allocatedBytes,input.shape.numel*4)
      decoder.close()
      intercept[IllegalStateException](decoder(x))
      x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("offloaded U-Net retains parameter snapshots and fits below resident weight size") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val directory=sys.props.get("scalatorch.unet.dir")
    assume(directory.nonEmpty,"public tiny U-Net required")
    val model=PretrainedUNet.load(java.nio.file.Path.of(directory.get))
    val parameters=Module.parameters(model)
    val total=parameters.map(_.data.shape.numel*4).sum
    val budget=parameters.map(_.data.shape.numel*4).max+1024*1024L
    assert(budget<total)
    val times=Tensor.fromFloats(Array(7.5f),Shape.from(Seq(1L)))
    CudaSession.withDevice() { s =>
      val x=s.fill(Shape.from(Seq(1L,4L,3L,5L)),.2f)
      val context=s.fill(Shape.from(Seq(1L,3L,32L)),.3f)
      val resident=CudaUNet.upload(s,model)
      val output=resident(x,times,context)
      val expected=output.toCpu.toFloatArray.toVector
      output.close();resident.close();context.close();x.close()
      CudaSession.withDevice(maxBytes=budget) { small =>
        val offloaded=CudaUNet.upload(small,model,offloadWeights=true)
        assertEquals(small.allocatedBytes,0L)
        val p=parameters.head
        p.load(Tensor.fromFloats(Array.fill(p.data.shape.numel.toInt)(0f),p.data.shape))
        val input=small.fill(Shape.from(Seq(1L,4L,3L,5L)),.2f)
        val ctx=small.fill(Shape.from(Seq(1L,3L,32L)),.3f)
        var cachedAllocation=0L
        for iteration <- 0 until 2 do
          val out=offloaded(input,times,ctx)
          assertEquals(out.toCpu.toFloatArray.toVector,expected)
          out.close()
          assert(small.allocatedBytes>=(input.shape.numel+ctx.shape.numel)*4)
          assert(small.allocatedBytes<=budget)
          if iteration==0 then cachedAllocation=small.allocatedBytes
          else assertEquals(small.allocatedBytes,cachedAllocation)
        offloaded.close();ctx.close();input.close()
        assertEquals(small.allocatedBytes,0L)
      }
    }
  }
