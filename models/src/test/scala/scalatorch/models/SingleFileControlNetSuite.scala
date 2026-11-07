package scalatorch.models

class SingleFileControlNetSuite extends munit.FunSuite:
  test("consuming a PyTorch tensor preserves defensive copy isolation") {
    import scalatorch.io.PyTorchTensorArchive.*
    val raw=Array(2f)
    val value=TensorValue(StoredTensor(StorageDType.Float32,Vector(1),raw))
    val view=SingleFileControlNet.pytorchView(DictValue(Map("control_model.input_blocks.0.0.weight"->value)))
    val tensor=view.readFloat("conv_in.weight")
    raw(0)=9f
    assertEquals(tensor.toFloatArray.toVector,Vector(2f))
    intercept[IllegalArgumentException](view.readFloat("conv_in.weight"))
  }
  test("maps restricted PyTorch state dictionaries without Python") {
    import scalatorch.io.PyTorchTensorArchive.*
    val tensor=TensorValue(StoredTensor(StorageDType.Float16,Vector(1),Array(2f)))
    val values=Map("control_model.input_blocks.0.0.weight"->tensor,
      "control_model.time_embed.0.weight"->tensor)
    for root <- Vector(DictValue(values),DictValue(Map("state_dict"->DictValue(values)))) do
      val view=SingleFileControlNet.pytorchView(root)
      assertEquals(view.tensors.keySet,Set("conv_in.weight","time_embedding.linear_1.weight"))
      assertEquals(view.readFloat("conv_in.weight").toFloatArray.toVector,Vector(2f))
      intercept[IllegalArgumentException](view.readFloat("conv_in.weight"))
      assertEquals(view.readFloat("time_embedding.linear_1.weight").toFloatArray.toVector,Vector(2f))
    intercept[IllegalArgumentException](SingleFileControlNet.pytorchView(DictValue(Map("state_dict"->StringValue("invalid")))))
    intercept[IllegalArgumentException](SingleFileControlNet.pytorchView(DictValue(values.updated("not_a_tensor",StringValue("invalid")))))
  }
  test("recognizes Diffusers single-file tensor names and rejects mixed formats") {
    import scalatorch.{Shape,Tensor}
    import scalatorch.io.{SafeTensorSet,SafeTensorWrite,SafeTensorsWriter}
    import java.nio.file.Files
    import scala.util.Using
    val path=Files.createTempFile("diffusers-controlnet-", ".safetensors")
    val value=SafeTensorWrite.F32(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L))))
    try
      val diffusers=Map("time_embedding.linear_1.weight"->value,"conv_in.weight"->value)
      SafeTensorsWriter.write(path,diffusers,overwrite=true)
      Using.resource(SafeTensorSet.openFile(path)) { archive =>
        val view=SingleFileControlNet.view(archive)
        assertEquals(view.tensors.keySet,diffusers.keySet)
        assertEquals(view.readFloat("conv_in.weight").toFloatArray.toVector,Vector(1f))
      }
      SafeTensorsWriter.write(path,diffusers.updated("control_model.input_blocks.0.0.weight",value),overwrite=true)
      Using.resource(SafeTensorSet.openFile(path)) { archive =>
        intercept[IllegalArgumentException](SingleFileControlNet.view(archive))
      }
    finally Files.deleteIfExists(path)
  }
  test("rejects archives that mix prefixed and unprefixed ControlNet tensors") {
    import scalatorch.{Shape,Tensor}
    import scalatorch.io.{SafeTensorSet,SafeTensorWrite,SafeTensorsWriter}
    import java.nio.file.Files
    import scala.util.Using
    val path=Files.createTempFile("mixed-controlnet-", ".safetensors")
    val value=SafeTensorWrite.F32(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L))))
    try
      SafeTensorsWriter.write(path,Map("control_model.input_blocks.0.0.weight"->value,
        "input_blocks.0.0.bias"->value),overwrite=true)
      Using.resource(SafeTensorSet.openFile(path)) { archive =>
        intercept[IllegalArgumentException](SingleFileControlNet.view(archive))
      }
    finally Files.deleteIfExists(path)
  }
  test("all 340 real checkpoint names match the Diffusers reference converter") {
    import scalatorch.models.worker.NativeJson
    import scalatorch.models.worker.NativeJson.*
    val stream=this.getClass.getResourceAsStream("/controlnet-original-sd15-names-reference.json")
    val reference=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields
      finally stream.close()
    assertEquals(reference.size,340)
    reference.foreach { (target,source) =>
      assertEquals(SingleFileControlNet.mapName(source.asInstanceOf[Str].value),target)
      assertEquals(SingleFileControlNet.mapName(source.asInstanceOf[Str].value.stripPrefix("control_model.")),target)
    }
  }
  test("maps original SD1 ControlNet names into Diffusers names") {
    val pairs=Vector(
      "control_model.input_blocks.0.0.weight" -> "conv_in.weight",
      "control_model.input_blocks.2.0.in_layers.2.weight" -> "down_blocks.0.resnets.1.conv1.weight",
      "control_model.input_blocks.3.0.op.bias" -> "down_blocks.0.downsamplers.0.conv.bias",
      "control_model.input_blocks.5.1.norm.weight" -> "down_blocks.1.attentions.1.norm.weight",
      "control_model.middle_block.2.out_layers.3.weight" -> "mid_block.resnets.1.conv2.weight",
      "control_model.input_hint_block.0.weight" -> "controlnet_cond_embedding.conv_in.weight",
      "control_model.input_hint_block.4.bias" -> "controlnet_cond_embedding.blocks.1.bias",
      "control_model.input_hint_block.14.weight" -> "controlnet_cond_embedding.conv_out.weight",
      "control_model.zero_convs.11.0.weight" -> "controlnet_down_blocks.11.weight",
      "control_model.middle_block_out.0.bias" -> "controlnet_mid_block.bias",
      "control_model.time_embed.2.weight" -> "time_embedding.linear_2.weight")
    pairs.foreach((source,target) => assertEquals(SingleFileControlNet.mapName(source),target))
    intercept[IllegalArgumentException](SingleFileControlNet.mapName("control_model.unknown.weight"))
    intercept[IllegalArgumentException](SingleFileControlNet.mapName("control_model.input_hint_block.3.weight"))
  }
