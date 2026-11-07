package scalatorch.models

import scalatorch.io.PyTorchTensorArchive.*

class SingleFileSd1PytorchSuite extends munit.FunSuite:
  test("SD1 PyTorch component views preserve names and integer positions") {
    val weight=TensorValue(StoredTensor(StorageDType.Float32,Vector(1),Array(2f)))
    val position=IntegerTensorValue(StoredIntegerTensor(StorageDType.Int64,Vector(2),Array(0L,9007199254740993L)))
    val values=Map[String,Value](
      "cond_stage_model.transformer.text_model.embeddings.position_ids"->position,
      "model.diffusion_model.input_blocks.0.0.weight"->weight,
      "first_stage_model.encoder.conv_in.weight"->weight)
    val clip=SingleFileTensorView.pytorchClip(values)
    val unet=SingleFileTensorView.pytorchUnet(values)
    val vae=SingleFileTensorView.pytorchVae(values)
    assertEquals(clip.readLong("text_model.embeddings.position_ids").toLongArray.toVector,Vector(0L,9007199254740993L))
    assertEquals(unet.readFloat("conv_in.weight").toFloatArray.toVector,Vector(2f))
    assertEquals(vae.readFloat("encoder.conv_in.weight").toFloatArray.toVector,Vector(2f))
  }
