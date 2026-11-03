package scalatorch.models

import munit.FunSuite
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.ffm.CudaSession
import scalatorch.nn.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}

class DiffusionLoRASuite extends FunSuite:
  private def write(dir: Path,entries: Vector[(String,Vector[Long],Array[Float])]): Path =
    var offset=0
    val header=entries.map { (name,dims,values) =>
      val begin=offset;offset+=values.length*4
      s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
    entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    val path=dir.resolve("adapter.safetensors")
    Files.write(path,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())
    path

  test("loads canonical and Kohya linear factors with independent text and U-Net scales") {
    val dir=Files.createTempDirectory("diffusion-lora-")
    val unetWeight=Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2L,2L))),false)
    val textWeight=Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2L,2L))),false)
    val path=write(dir,Vector(
      ("unet.down_blocks.0.attentions.0.to_q.lora_down.weight",Vector(1L,2L),Array(1f,0f)),
      ("unet.down_blocks.0.attentions.0.to_q.lora_up.weight",Vector(2L,1L),Array(2f,3f)),
      ("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",Vector(1L,2L),Array(1f,0f)),
      ("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight",Vector(2L,1L),Array(2f,3f)),
      ("unet.down_blocks.0.attentions.0.to_q.alpha",Vector.empty,Array(2f))))
    val loaded=DiffusionLoRA.load(path,Map(
      "unet.down_blocks.0.attentions.0.to_q"->unetWeight,
      "text_encoder.text_model.encoder.layers.0.self_attn.q_proj"->textWeight),textEncoderScale=0.5f,unetScale=2f)
    val input=Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1L,2L))))
    val unet=Linear(unetWeight,None)
    val text=Linear(textWeight,None)
    LinearAdapterScope.withAdapters(loaded) {
      assertEquals(unet(input).data.toFloatArray.toVector,Vector(9f,14f))
      assertEquals(text(input).data.toFloatArray.toVector,Vector(2f,3.5f))
    }
    val negative=DiffusionLoRA.load(path,Map(
      "unet.down_blocks.0.attentions.0.to_q"->unetWeight,
      "text_encoder.text_model.encoder.layers.0.self_attn.q_proj"->textWeight),textEncoderScale=0f,unetScale= -0.5f)
    assertEquals(LinearAdapterScope.withAdapters(negative)(unet(input).data.toFloatArray.toVector),Vector(-1f,-1f))
    assertEquals(unet(input).data.toFloatArray.toVector,Vector(1f,2f))
    assertEquals(unetWeight.data.toFloatArray.toVector,Vector(1f,0f,0f,1f))
  }

  test("rejects unsupported, incomplete and shape-mismatched adapters") {
    val dir=Files.createTempDirectory("diffusion-lora-invalid-")
    val weight=Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2L,2L))),false)
    val targets=Map("unet.test"->weight)
    def run(entries: Vector[(String,Vector[Long],Array[Float])]) =
      DiffusionLoRA.load(write(dir,entries),targets)
    intercept[IllegalArgumentException] { run(Vector(("unet.test.lora_down.weight",Vector(1L,2L),Array(1f,0f)))) }
    intercept[IllegalArgumentException] { run(Vector(("unet.test.lora_down.weight",Vector(1L,3L),Array(1f,0f,1f)),
      ("unet.test.lora_up.weight",Vector(2L,1L),Array(1f,1f)))) }
    intercept[IllegalArgumentException] { run(Vector(("unet.test.lora_mid.weight",Vector(1L,1L),Array(1f)))) }
    val peft=DiffusionLoRA.load(write(dir,Vector(
      ("base_model.model.unet.test.lora_A.default.weight",Vector(1L,2L),Array(1f,0f)),
      ("base_model.model.unet.test.lora_B.default.weight",Vector(2L,1L),Array(2f,3f)))),targets,
      textEncoderScale=0f,unetScale=2f)
    val input=Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1L,2L))))
    assertEquals(LinearAdapterScope.withAdapters(peft)(Linear(weight,None)(input).data.toFloatArray.toVector),Vector(5f,8f))
  }

  test("CUDA Linear consumes scoped factors with resident and host-offloaded weights") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val weight=Parameter(Tensor.fromFloats(Array(1f,0f,0f,1f),Shape.from(Seq(2L,2L))),false)
    val linear=Linear(weight,None)
    val input=Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(1L,2L)))
    val adapter=LowRankAdapter(
      Parameter(Tensor.fromFloats(Array(1f,0f),Shape.from(Seq(1L,2L))),false),
      Parameter(Tensor.fromFloats(Array(2f,3f),Shape.from(Seq(2L,1L))),false),1.0)
    val active=Map(weight->Vector(adapter->0.5f))
    val expected=LinearAdapterScope.withAdapters(active)(linear(Variable(input)).data)
    for offload <- Seq(false,true) do
      CudaSession.withDevice() { session =>
        LinearAdapterScope.withAdapters(active) {
          val runtime=CudaDiffusionRuntime.upload(session,linear,offloadWeights=offload)
          val deviceInput=session.upload(input)
          val output=runtime.linear(deviceInput,linear)
          assertEquals(output.toCpu.toFloatArray.toVector,expected.toFloatArray.toVector)
          output.close();deviceInput.close();runtime.close()
          assertEquals(session.allocatedBytes,0L)
        }
      }
  }
