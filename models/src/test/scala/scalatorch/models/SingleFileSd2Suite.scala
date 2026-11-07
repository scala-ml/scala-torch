package scalatorch.models

import scalatorch.Shape
import scalatorch.io.{SafeDType,TensorInfo}
import scalatorch.io.PyTorchTensorArchive.*

class SingleFileSd2Suite extends munit.FunSuite:
  private def info(shape: Long*): TensorInfo =
    TensorInfo(SafeDType.F32,Shape.from(shape),0L,shape.product*4L)

  private def sd2Tensors(depths: Seq[Int]=0 to 23,crossWidth: Int=1024,clipWidth: Int=1024,
      sd1Clip: Boolean=false): Map[String,TensorInfo] =
    val tensors=scala.collection.mutable.Map.empty[String,TensorInfo]
    tensors("model.diffusion_model.input_blocks.0.0.weight")=info(320,4,3,3)
    tensors("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight")=info(320,crossWidth)
    tensors("model.diffusion_model.out.2.weight")=info(4,320,3,3)
    tensors("cond_stage_model.model.token_embedding.weight")=info(49408,clipWidth)
    tensors("cond_stage_model.model.positional_embedding")=info(77,clipWidth)
    tensors("cond_stage_model.model.ln_final.weight")=info(clipWidth)
    tensors("cond_stage_model.model.ln_final.bias")=info(clipWidth)
    tensors("cond_stage_model.model.text_projection")=info(clipWidth,clipWidth)
    tensors("cond_stage_model.model.logit_scale")=info()
    depths.foreach(i => tensors(s"cond_stage_model.model.transformer.resblocks.$i.attn.in_proj_weight")=info(3072,clipWidth))
    tensors("first_stage_model.encoder.conv_in.weight")=info(128,3,3,3)
    tensors("first_stage_model.decoder.conv_out.weight")=info(3,128,3,3)
    if sd1Clip then
      tensors("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")=info(49408,768)
    tensors.toMap

  test("OpenCLIP single-file fused QKV tensors are exposed as separate CLIP projections") {
    val weight=TensorValue(StoredTensor(StorageDType.Float32,Vector(6,2),Array.tabulate(12)(_.toFloat)))
    val bias=TensorValue(StoredTensor(StorageDType.Float32,Vector(6),Array.tabulate(6)(_.toFloat)))
    val values=Map[String,Value](
      "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight"->weight,
      "cond_stage_model.model.transformer.resblocks.0.attn.in_proj_bias"->bias)

    val view=SingleFileTensorView.pytorchSd2Clip(values)
    val prefix="text_model.encoder.layers.0.self_attn."
    val query=view.readFloat(prefix+"q_proj.weight")
    val key=view.readFloat(prefix+"k_proj.weight")
    val value=view.readFloat(prefix+"v_proj.weight")

    assertEquals(query.shape.dims.toVector,Vector(2L,2L))
    assertEquals(query.toFloatArray.toVector,Vector(0f,1f,2f,3f))
    assertEquals(key.toFloatArray.toVector,Vector(4f,5f,6f,7f))
    assertEquals(value.toFloatArray.toVector,Vector(8f,9f,10f,11f))
    assertEquals(view.readFloat(prefix+"q_proj.bias").toFloatArray.toVector,Vector(0f,1f))
    assertEquals(view.readFloat(prefix+"k_proj.bias").toFloatArray.toVector,Vector(2f,3f))
    assertEquals(view.readFloat(prefix+"v_proj.bias").toFloatArray.toVector,Vector(4f,5f))
  }

  test("the SD2 view ignores the dead 24th OpenCLIP layer from released checkpoints") {
    val weight=TensorValue(StoredTensor(StorageDType.Float32,Vector(6,2),Array.tabulate(12)(_.toFloat)))
    val values=Map[String,Value](
      "cond_stage_model.model.transformer.resblocks.22.attn.in_proj_weight"->weight,
      "cond_stage_model.model.transformer.resblocks.23.attn.in_proj_weight"->weight)
    val view=SingleFileTensorView.pytorchSd2Clip(values)
    assert(view.tensors.contains("text_model.encoder.layers.22.self_attn.q_proj.weight"))
    assert(!view.tensors.keySet.exists(_.startsWith("text_model.encoder.layers.23.")),
      "resblock 23 must stay unmapped, matching the 23-layer diffusers text encoder")
  }

  test("detects the SD2 OpenCLIP checkpoint layout") {
    assert(SingleFileSd2Config.isSd2(sd2Tensors()))
    assert(!SingleFileSd2Config.isSd2(Map(
      "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"->info(49408,768))))
  }

  test("accepts the released SD2 layout with the ignored 24th layer") {
    SingleFileSd2Config.inspectTensors(sd2Tensors())
    SingleFileSd2Config.inspectTensors(sd2Tensors(depths=0 to 22))
  }

  test("rejects SD1-layout text encoders") {
    intercept[IllegalArgumentException](SingleFileSd2Config.inspectTensors(sd2Tensors(sd1Clip=true)))
  }

  test("rejects missing OpenCLIP layers before weight allocation") {
    val missing=sd2Tensors(depths=(0 to 23).filter(_!=5))
    interceptMessage[IllegalArgumentException]("requirement failed: missing SD2 OpenCLIP transformer layer: 5") {
      SingleFileSd2Config.inspectTensors(missing)
    }
  }

  test("rejects OpenCLIP stacks deeper than the released 24-layer layout") {
    intercept[IllegalArgumentException](SingleFileSd2Config.inspectTensors(sd2Tensors(depths=0 to 24)))
  }

  test("rejects SD1-width cross attention") {
    intercept[IllegalArgumentException](SingleFileSd2Config.inspectTensors(sd2Tensors(crossWidth=768)))
  }

  test("derives an SD2 U-Net config with 1024 context width") {
    val config=SingleFileSd2Config.unetConfigForTensors(sd2Tensors())
    assertEquals(config.contextWidth,1024)
    assertEquals(config.inChannels,4)
    assertEquals(config.channels,Vector(320,640,1280,1280))
    assertEquals(config.stageHeads,Vector(5,10,20,20))
    assert(config.linearProjection)
  }

  test("single-file VAE config uses a 512px sample tile so 64px and 512px decodes stay whole-frame") {
    val vae=SingleFileSd1Config.vaeConfig
    assertEquals((vae.sampleHeight,vae.sampleWidth),(512,512))
    assert(!EulerImagePipeline.shouldTileDecode(8,8,vae.sampleHeight,vae.sampleWidth,8,configured=false))
    assert(!EulerImagePipeline.shouldTileDecode(64,64,vae.sampleHeight,vae.sampleWidth,8,configured=false))
    assert(EulerImagePipeline.shouldTileDecode(96,96,vae.sampleHeight,vae.sampleWidth,8,configured=false))
  }

