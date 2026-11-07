package scalatorch.models

import scalatorch.Shape
import scalatorch.io.{SafeDType,TensorInfo}
import scalatorch.io.PyTorchTensorArchive.*

class SingleFileSdxlSuite extends munit.FunSuite:
  private def info(shape: Long*): TensorInfo =
    TensorInfo(SafeDType.F32,Shape.from(shape),0L,shape.product*4L)

  private def sdxlTensors(
      clipLDepths: Seq[Int]=0 to 11,
      clipGDepths: Seq[Int]=0 to 31,
      crossWidth: Int=2048,
      admWidth: Int=2816,
      refinerOnly: Boolean=false,
      extraStage: Boolean=false): Map[String,TensorInfo] =
    val tensors=scala.collection.mutable.Map.empty[String,TensorInfo]
    tensors("model.diffusion_model.input_blocks.0.0.weight")=info(320,4,3,3)
    tensors("model.diffusion_model.label_emb.0.0.weight")=info(1280,admWidth)
    tensors("model.diffusion_model.label_emb.0.0.bias")=info(1280)
    tensors("model.diffusion_model.label_emb.0.2.weight")=info(1280,1280)
    tensors("model.diffusion_model.label_emb.0.2.bias")=info(1280)
    tensors("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight")=info(640,crossWidth)
    tensors("model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight")=info(1280,crossWidth)
    tensors("model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_k.weight")=info(1280,crossWidth)
    tensors("model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_k.weight")=info(1280,crossWidth)
    tensors("model.diffusion_model.out.2.weight")=info(4,320,3,3)
    if extraStage then
      tensors("model.diffusion_model.input_blocks.9.0.in_layers.0.weight")=info(1280)
    if refinerOnly then
      tensors("conditioner.embedders.0.model.token_embedding.weight")=info(49408,1280)
      tensors("conditioner.embedders.0.model.positional_embedding")=info(77,1280)
      tensors("conditioner.embedders.0.model.text_projection")=info(1280,1280)
    else
      tensors("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight")=info(49408,768)
      tensors("conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight")=info(77,768)
      tensors("conditioner.embedders.0.transformer.text_model.final_layer_norm.weight")=info(768)
      tensors("conditioner.embedders.0.transformer.text_model.final_layer_norm.bias")=info(768)
      clipLDepths.foreach { i =>
        tensors(s"conditioner.embedders.0.transformer.text_model.encoder.layers.$i.self_attn.q_proj.weight")=info(768,768)
      }
      tensors("conditioner.embedders.1.model.token_embedding.weight")=info(49408,1280)
      tensors("conditioner.embedders.1.model.positional_embedding")=info(77,1280)
      tensors("conditioner.embedders.1.model.ln_final.weight")=info(1280)
      tensors("conditioner.embedders.1.model.ln_final.bias")=info(1280)
      tensors("conditioner.embedders.1.model.text_projection")=info(1280,1280)
      tensors("conditioner.embedders.1.model.logit_scale")=info()
      clipGDepths.foreach { i =>
        tensors(s"conditioner.embedders.1.model.transformer.resblocks.$i.attn.in_proj_weight")=info(3840,1280)
      }
    tensors("first_stage_model.encoder.conv_in.weight")=info(128,3,3,3)
    tensors("first_stage_model.decoder.conv_out.weight")=info(3,128,3,3)
    tensors.toMap

  test("detects the SDXL dual-conditioner checkpoint layout") {
    assert(SingleFileSdxlConfig.isSdxl(sdxlTensors()))
    assert(!SingleFileSdxlConfig.isSdxl(Map(
      "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"->info(49408,768))))
    assert(!SingleFileSdxlConfig.isSdxl(Map(
      "cond_stage_model.model.token_embedding.weight"->info(49408,1024))))
  }

  test("accepts the released SDXL base single-file layout") {
    SingleFileSdxlConfig.inspectTensors(sdxlTensors())
  }

  test("rejects SDXL refiner single-encoder checkpoints") {
    intercept[IllegalArgumentException](SingleFileSdxlConfig.inspectTensors(sdxlTensors(refinerOnly=true)))
  }

  test("rejects missing CLIP-L or CLIP-G transformer layers") {
    val missingL=sdxlTensors(clipLDepths=(0 to 11).filter(_!=4))
    interceptMessage[IllegalArgumentException]("requirement failed: missing SDXL CLIP-L transformer layer: 4") {
      SingleFileSdxlConfig.inspectTensors(missingL)
    }
    val missingG=sdxlTensors(clipGDepths=(0 to 31).filter(_!=17))
    interceptMessage[IllegalArgumentException]("requirement failed: missing SDXL CLIP-G transformer layer: 17") {
      SingleFileSdxlConfig.inspectTensors(missingG)
    }
  }

  test("rejects non-SDXL cross-attention or ADM dimensions and four-stage U-Nets") {
    intercept[IllegalArgumentException](SingleFileSdxlConfig.inspectTensors(sdxlTensors(crossWidth=1024)))
    intercept[IllegalArgumentException](SingleFileSdxlConfig.inspectTensors(sdxlTensors(admWidth=2560)))
    intercept[IllegalArgumentException](SingleFileSdxlConfig.inspectTensors(sdxlTensors(extraStage=true)))
  }

  test("maps CLIP-L and CLIP-G tensors with transposed OpenCLIP text_projection") {
    val clipLWeight=TensorValue(StoredTensor(StorageDType.Float32,Vector(2,2),Array(1f,2f,3f,4f)))
    val clipLValues=Map[String,Value](
      "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"->clipLWeight)
    val clipLView=SingleFileTensorView.pytorchSdxlClip(clipLValues)
    assertEquals(clipLView.readFloat("text_model.embeddings.token_embedding.weight").toFloatArray.toVector,
      Vector(1f,2f,3f,4f))

    val qkvWeight=TensorValue(StoredTensor(StorageDType.Float32,Vector(6,2),Array.tabulate(12)(_.toFloat)))
    val projWeight=TensorValue(StoredTensor(StorageDType.Float32,Vector(2,3),Array(1f,2f,3f,4f,5f,6f)))
    val clipGValues=Map[String,Value](
      "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight"->qkvWeight,
      "conditioner.embedders.1.model.text_projection"->projWeight,
      "conditioner.embedders.1.model.logit_scale"->TensorValue(StoredTensor(StorageDType.Float32,Vector.empty,Array(4.6f))))
    val clipGView=SingleFileTensorView.pytorchSdxlClip2(clipGValues)
    val transposed=clipGView.readFloat("text_projection.weight")
    assertEquals(transposed.shape.dims.toVector,Vector(3L,2L))
    assertEquals(transposed.toFloatArray.toVector,Vector(1f,4f,2f,5f,3f,6f))
    assertEquals(clipGView.readFloat("text_model.encoder.layers.0.self_attn.k_proj.weight").toFloatArray.toVector,
      Vector(4f,5f,6f,7f))
  }

  test("maps SDXL U-Net label_emb to add_embedding and derives 3-stage SDXL U-Net and VAE configs") {
    val w1=TensorValue(StoredTensor(StorageDType.Float32,Vector(2,3),Array.fill(6)(1f)))
    val b1=TensorValue(StoredTensor(StorageDType.Float32,Vector(2),Array.fill(2)(2f)))
    val unetView=SingleFileTensorView.pytorchUnet(Map(
      "model.diffusion_model.label_emb.0.0.weight"->w1,
      "model.diffusion_model.label_emb.0.0.bias"->b1,
      "model.diffusion_model.label_emb.0.2.weight"->w1,
      "model.diffusion_model.label_emb.0.2.bias"->b1))
    assertEquals(unetView.readFloat("add_embedding.linear_1.weight").shape.dims.toVector,Vector(2L,3L))
    assertEquals(unetView.readFloat("add_embedding.linear_2.bias").toFloatArray.toVector,Vector(2f,2f))

    val config=SingleFileSdxlConfig.unetConfigForTensors(sdxlTensors())
    assertEquals(config.channels,Vector(320,640,1280))
    assertEquals(config.downAttention,Vector(false,true,true))
    assertEquals(config.upAttention,Vector(true,true,false))
    assertEquals(config.stageHeads,Vector(5,10,20))
    assertEquals(config.stageTransformerLayers,Vector(1,2,10))
    assertEquals(config.contextWidth,2048)
    assertEquals(config.additionTimeWidth,Some(256))
    assertEquals(config.additionInputWidth,Some(2816))
    assert(config.linearProjection)

    val vae=SingleFileSdxlConfig.vaeConfig
    assertEquals(vae.scalingFactor,0.13025)
    assertEquals((vae.sampleHeight,vae.sampleWidth),(1024,1024))
  }
