package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.io.SafeTensors
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}

class SdxlUNetSuite extends munit.FunSuite:
  private def tensor(name: String,shape: Shape): Tensor[Float] =
    val stream=getClass.getResourceAsStream(s"/tiny-sdxl/$name.bin")
    val data=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
    Tensor.fromFloats(Array.fill(data.remaining()/4)(data.getFloat()),shape)

  test("SDXL CPU text-time conditioning matches independent Diffusers outputs") {
    val directory=sys.props.get("scalatorch.sdxl.dir")
    assume(directory.nonEmpty,"tiny SDXL required")
    val model=PretrainedUNet.load(Path.of(directory.get).resolve("unet"))
    for batch <- Seq(1L,2L) do
      val suffix=if batch==1 then "" else "-b2"
      val sample=tensor("sample"+suffix,Shape.from(Seq(batch,4L,8L,8L)))
      val context=tensor("context"+suffix,Shape.from(Seq(batch,6L,64L)))
      val pooled=tensor("pooled"+suffix,Shape.from(Seq(batch,32L)))
      val ids=tensor("time-ids"+suffix,Shape.from(Seq(batch,6L)))
      val time=Tensor.fromFloats(Array(500f),Shape.from(Seq(1L)))
      val added=DiffusionAddedConditioning(pooled,ids)
      val expected=tensor("unet-output"+suffix,sample.shape)
      val actual=Autograd.noGrad(model(Variable(sample),time,Variable(context),Some(added))).data
      val error=actual.toFloatArray.zip(expected.toFloatArray).map((a,e) => math.abs(a.toDouble-e)).max
      println(s"SDXL CPU batch=$batch maxAbsError=$error")
      assert(error<2e-5)
      intercept[IllegalArgumentException](model(Variable(sample),time,Variable(context)))
      val wrong=added.copy(pooledText=Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))))
      intercept[IllegalArgumentException](model(Variable(sample),time,Variable(context),Some(wrong)))
      val badIds=ids.toFloatArray;badIds(0)=Float.NaN
      intercept[IllegalArgumentException](model(Variable(sample),time,Variable(context),Some(added.copy(timeIds=Tensor.fromFloats(badIds,ids.shape)))))
      val badText=pooled.toFloatArray;badText(0)=Float.PositiveInfinity
      intercept[IllegalArgumentException](model(Variable(sample),time,Variable(context),Some(added.copy(pooledText=Tensor.fromFloats(badText,pooled.shape)))))
      val wrongBatch=Tensor.fromFloats(Array.fill((batch.toInt+1)*6)(0f),Shape.from(Seq(batch+1,6L)))
      intercept[IllegalArgumentException](model(Variable(sample),time,Variable(context),Some(added.copy(timeIds=wrongBatch))))
  }
  private val base="""{"block_out_channels":[32,64],"down_block_types":["DownBlock2D","CrossAttnDownBlock2D"],"up_block_types":["CrossAttnUpBlock2D","UpBlock2D"],"cross_attention_dim":32,"attention_head_dim":[2,4],"transformer_layers_per_block":[1,2],"use_linear_projection":true}"""
  test("stage heads and transformer depths validate each corresponding width") {
    val c=UNetConfig.fromJson(base)
    assertEquals(c.headsAt(0),2);assertEquals(c.headsAt(1),4)
    assertEquals(c.transformerLayersAt(0),1);assertEquals(c.transformerLayersAt(1),2)
    assert(c.linearProjection)
    val scalar=UNetConfig.fromJson(base.replace("[2,4]","8").replace("[1,2]","1"))
    assertEquals(scalar.headsAt(1),8)
    for bad <- Seq(base.replace("[2,4]","[2]"),base.replace("[2,4]","[2,3]"),
        base.replace("[1,2]","[1,0]"),base.replace("[1,2]","[1,2,3]")) do
      intercept[IllegalArgumentException](UNetConfig.fromJson(bad))
    intercept[IllegalArgumentException](UNetConfig.fromJson(base.dropRight(1)+",\"addition_embed_type\":\"text_time\"}"))
  }

  test("stage loader reverses up-block indexing and linear archive projections preserve inference") {
    val directory=sys.props.get("scalatorch.unet.dir")
    assume(directory.nonEmpty,"tiny U-Net required")
    val root=Path.of(directory.get)
    val temp=Files.createTempDirectory("stage-unet")
    try
      val original=scala.util.Using.resource(SafeTensors.open(root.resolve("diffusion_pytorch_model.safetensors"))) { archive =>
        archive.tensors.keys.toVector.sorted.map(name => name->archive.readFloat(name))
      }
      // Every attention is at stage1 in this two-stage checkpoint; duplicate its
      // first transformer layer to exercise depth2 without changing archive provenance.
      val entries=original++original.collect { case (name,value) if name.contains("transformer_blocks.0.") =>
        name.replace("transformer_blocks.0.","transformer_blocks.1.")->value }
      def write(rank: Int): Unit =
        Files.writeString(temp.resolve("config.json"),if rank==4 then base.replace("true","false") else base)
        var offset=0L
        val header=entries.map { (name,value) =>
          val spatial=(name.endsWith("proj_in.weight") || name.endsWith("proj_out.weight")) && name.contains("attentions")
          val dims=if spatial then value.shape.dims.take(rank) else value.shape.dims
          val begin=offset;offset+=value.shape.numel*4
          s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
        }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
        val data=ByteBuffer.allocate(offset.toInt).order(ByteOrder.LITTLE_ENDIAN)
        entries.foreach((_,value) => value.toFloatArray.foreach(data.putFloat))
        Files.write(temp.resolve("diffusion_pytorch_model.safetensors"),ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
          .putLong(header.length).array()++header++data.array())
      write(4)
      val conv=PretrainedUNet.load(temp)
      write(2)
      val linear=PretrainedUNet.load(temp)
      val attention=linear.downBlocks.flatMap(_.layers.flatMap(_.attention))++Vector(linear.midBlock.attention)++
        linear.upBlocks.flatMap(_.layers.flatMap(_.attention))
      assert(attention.nonEmpty)
      attention.foreach { a =>
        assertEquals(a.blocks.size,2)
        a.blocks.foreach(b => assertEquals(b.attn1.heads,4))
      }
      val input=Variable(Tensor.fromFloats(Array.tabulate(4*4*4)(i => math.sin(i*.13).toFloat),Shape.from(Seq(1L,4L,4L,4L))))
      val context=Variable(Tensor.fromFloats(Array.tabulate(3*32)(i => math.cos(i*.07).toFloat),Shape.from(Seq(1L,3L,32L))))
      val time=Tensor.fromFloats(Array(500f),Shape.from(Seq(1L)))
      val expected=Autograd.noGrad(conv(input,time,context)).data
      val unsupported=DiffusionAddedConditioning(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))),
        Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))))
      intercept[IllegalArgumentException](conv(input,time,context,Some(unsupported)))
      val actual=Autograd.noGrad(linear(input,time,context)).data
      assertEquals(actual.toFloatArray.toVector,expected.toFloatArray.toVector)
      if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
        scalatorch.ffm.CudaSession.withDevice() { session =>
          scala.util.Using.Manager { use =>
            val gpu=use(CudaUNet.upload(session,linear,offloadWeights=true))
            val x=use(session.upload(input.data));val c=use(session.upload(context.data))
            val y=use(gpu(x,time,c))
            y.toCpu.toFloatArray.zip(expected.toFloatArray).foreach((a,e) => assertEqualsDouble(a,e,2e-5))
          }.get
          assertEquals(session.allocatedBytes,0L)
        }
      write(3)
      intercept[IllegalArgumentException](PretrainedUNet.load(temp))
    finally
      val files=Files.list(temp)
      try files.forEach(p => {Files.delete(p);()}) finally files.close()
      Files.delete(temp)
  }
