package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.ffm.CudaSession

class CudaSpatialTransformerSuite extends munit.FunSuite:
  private def loaded(): DiffusionSpatialTransformer =
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val dir=sys.props.get("scalatorch.unet.dir")
    assume(dir.nonEmpty,"public tiny U-Net required")
    PretrainedUNet.load(java.nio.file.Path.of(dir.get)).downBlocks.flatMap(_.layers).flatMap(_.attention).head
  private def data(dims: Seq[Long],seed: Int): Tensor[Float] =
    val shape=Shape.from(dims);val random=new java.util.Random(seed)
    Tensor.fromFloats(Array.fill(shape.numel.toInt)((random.nextGaussian()*.3).toFloat),shape)
  private def compare(actual: Tensor[Float],expected: Tensor[Float]): Unit =
    assertEquals(actual.shape,expected.shape)
    val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
    println(s"CUDA spatial Transformer maxAbsError=$error")
    assert(error<2e-5)
  test("loaded U-Net GEGLU feed-forward and token Transformer match CPU") {
    val spatial=loaded();val block=spatial.blocks.head
    val input=data(Seq(2L,5L,block.norm1.width.toLong),1)
    val context=data(Seq(2L,7L,block.attn2.toK.weight.data.shape.dims(1)),2)
    val expectedFf=Autograd.noGrad(block.feedForward(Variable(input))).data
    val expected=Autograd.noGrad(block(Variable(input),Variable(context))).data
    CudaSession.withDevice() { s =>
      val runtime=CudaDiffusionRuntime.upload(s,block)
      val x=s.upload(input);val c=s.upload(context)
      val before=s.allocatedBytes
      val ff=runtime.feedForward(x,block.feedForward)
      compare(ff.toCpu,expectedFf);ff.close()
      val result=runtime.transformer(x,block,c)
      compare(result.toCpu,expected);result.close()
      assertEquals(s.allocatedBytes,before)
      runtime.close();c.close();x.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
  test("complete image-space Transformer stays resident and validates context before allocation") {
    val model=loaded()
    val input=data(Seq(2L,model.norm.channels.toLong,3L,5L),3)
    val context=data(Seq(2L,7L,model.blocks.head.attn2.toK.weight.data.shape.dims(1)),4)
    val expected=Autograd.noGrad(model(Variable(input),Variable(context))).data
    CudaSession.withDevice() { s =>
      val runtime=CudaDiffusionRuntime.upload(s,model)
      val x=s.upload(input);val c=s.upload(context)
      val before=s.allocatedBytes
      for _ <- 0 until 2 do
        val result=runtime.spatialTransformer(x,model,c)
        compare(result.toCpu,expected);result.close()
        assertEquals(s.allocatedBytes,before)
      intercept[IllegalArgumentException](runtime.spatialTransformer(x,model,x))
      assertEquals(s.allocatedBytes,before)
      val result=runtime.spatialTransformer(x,model,c)
      runtime.close();c.close();x.close()
      compare(result.toCpu,expected);result.close()
      assertEquals(s.allocatedBytes,0L)
    }
  }
