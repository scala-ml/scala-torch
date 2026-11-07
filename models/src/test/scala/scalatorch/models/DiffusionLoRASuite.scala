package scalatorch.models

import munit.FunSuite
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.ffm.CudaSession
import scalatorch.nn.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}

class DiffusionLoRASuite extends FunSuite:
  test("loads safetensors LoRA with uppercase filename extension") {
    val dir=Files.createTempDirectory("uppercase-lora")
    val source=write(dir,Vector(
      ("unet.test.lora_down.weight",Vector(1L,1L),Array(2f)),
      ("unet.test.lora_up.weight",Vector(1L,1L),Array(3f))))
    val path=dir.resolve("adapter.SAFETENSORS")
    Files.move(source,path)
    val weight=Parameter(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))),false)
    val input=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L))))
    val adapters=DiffusionLoRA.load(path,Map("unet.test"->weight))
    assertEquals(LinearAdapterScope.withAdapters(adapters)(Linear(weight,None)(input).data.toFloatArray.head),6f)
  }
  test("restricted PyTorch LoRA archives load with the same scale and dynamic rank") {
    import scalatorch.io.{PyTorchTensorArchive as A,PyTorchTensorArchiveWriter as Writer}
    def tensor(shape: Vector[Long],values: Float*)=A.TensorValue(A.StoredTensor(A.StorageDType.Float32,shape,values.toArray))
    val path=Files.createTempDirectory("pytorch-lora").resolve("adapter.pt")
    Writer.write(path,A.DictValue(Map(
      "unet.test.lora_down.weight"->tensor(Vector(2L,1L),1f,2f),
      "unet.test.lora_up.weight"->tensor(Vector(1L,2L),3f,4f),
      "unet.test.alpha"->tensor(Vector.empty,4f))))
    val weight=Parameter(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))),false)
    val input=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L))))
    val adapters=DiffusionLoRA.load(path,Map("unet.test"->weight),dynamicRank=Some(1))
    assertEquals(LinearAdapterScope.withAdapters(adapters)(Linear(weight,None)(input).data.toFloatArray.head),6f)
    val reference=path.resolveSibling("pytorch-reference.pt")
    val resource=getClass.getResourceAsStream("/pytorch-lora-reference.pt")
    try Files.copy(resource,reference) finally resource.close()
    val fromPyTorch=DiffusionLoRA.load(reference,Map("unet.test"->weight),dynamicRank=Some(1))
    assertEquals(LinearAdapterScope.withAdapters(fromPyTorch)(Linear(weight,None)(input).data.toFloatArray.head),6f)
    val wrapped=path.resolveSibling("wrapped.ckpt")
    Writer.write(wrapped,A.DictValue(Map("epoch"->A.IntegerValue(12),"name"->A.StringValue("adapter"),
      "state_dict"->A.DictValue(Map(
        "unet.test.lora_down.weight"->tensor(Vector(2L,1L),1f,2f),
        "unet.test.lora_up.weight"->tensor(Vector(1L,2L),3f,4f),
        "unet.test.alpha"->tensor(Vector.empty,4f))))))
    val fromWrapped=DiffusionLoRA.load(wrapped,Map("unet.test"->weight),dynamicRank=Some(1))
    assertEquals(LinearAdapterScope.withAdapters(fromWrapped)(Linear(weight,None)(input).data.toFloatArray.head),6f)
    intercept[IllegalArgumentException](DiffusionLoRA.load(path,Map("unet.test"->weight),elementBudget=2))
  }
  test("PyTorch wrapped state dictionary permits numeric training metadata") {
    val path=Files.createTempDirectory("pytorch-wrapped").resolve("adapter.ckpt")
    val resource=getClass.getResourceAsStream("/pytorch-lora-wrapped-reference.ckpt")
    try Files.copy(resource,path) finally resource.close()
    val weight=Parameter(Tensor.fromFloats(Array.fill(128*32)(0f),Shape.from(Seq(128L,32L))),false)
    val adapters=DiffusionLoRA.load(path,Map("unet.time_embedding.linear_1"->weight))
    assert(adapters.contains(weight))
  }
  test("original LDM aliases resolve to SDXL U-Net convolution and attention targets") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"tiny SDXL checkpoint required")
    val model=PretrainedUNet.loadWithLinearTargets(Path.of(configured.get).resolve("unet"))
    val conv=model.linearTargets("unet.down_blocks.0.resnets.0.conv1")
    val attention=model.linearTargets("unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q")
    val in=conv.data.shape.dims(1).toInt;val out=conv.data.shape.dims(0).toInt
    val ain=attention.data.shape.dims(1).toInt;val aout=attention.data.shape.dims(0).toInt
    val path=write(Files.createTempDirectory("ldm-sdxl-lora"),Vector(
      ("lora_unet_input_blocks_1_0_in_layers_2.lora_down.weight",Vector(1L,in.toLong,3L,3L),Array.fill(in*9)(0.01f)),
      ("lora_unet_input_blocks_1_0_in_layers_2.lora_up.weight",Vector(out.toLong,1L,1L,1L),Array.fill(out)(1f)),
      ("lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_q.lora_down.weight",Vector(1L,ain.toLong),Array.fill(ain)(0.01f)),
      ("lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_q.lora_up.weight",Vector(aout.toLong,1L),Array.fill(aout)(1f))))
    val loaded=DiffusionLoRA.load(path,model.linearTargets)
    assert(loaded.contains(conv) && loaded.contains(attention))
  }
  test("SDXL linear projections accept matrix LoRA factors on their pointwise convolution") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty,"tiny SDXL checkpoint required")
    val model=PretrainedUNet.loadWithLinearTargets(Path.of(configured.get).resolve("unet"))
    val name="unet.down_blocks.1.attentions.0.proj_in"
    val target=model.linearTargets(name)
    assertEquals(target.data.shape.rank,4)
    val in=target.data.shape.dims(1).toInt;val out=target.data.shape.dims(0).toInt
    val path=write(Files.createTempDirectory("sdxl-projection-lora"),Vector(
      (s"$name.lora_down.weight",Vector(1L,in.toLong),Array.fill(in)(1f/in)),
      (s"$name.lora_up.weight",Vector(out.toLong,1L),Array.fill(out)(1f))))
    val adapters=DiffusionLoRA.load(path,model.linearTargets)
    assert(adapters.contains(target))
    assertEquals(adapters(target).head._1.a.data.shape.dims,Vector(1L,in.toLong,1L,1L))
    assertEquals(adapters(target).head._1.b.data.shape.dims,Vector(out.toLong,1L,1L,1L))
    val projection=model.model.downBlocks(1).layers.head.attention.get.projIn
    assert(projection.weight eq target,"LoRA target must be the weight used in inference")
    val input=Tensor.fromFloats(Array.fill(in*4)(0.25f),Shape.from(Seq(1L,in.toLong,2L,2L)))
    val baseline=projection(Variable(input)).data.toFloatArray
    val expected=LinearAdapterScope.withAdapters(adapters)(projection(Variable(input)).data.toFloatArray)
    assert(expected.zip(baseline).exists((a,b) => math.abs(a-b)>1e-5f))
    if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then
      CudaSession.withDevice() { session =>
        LinearAdapterScope.withAdapters(adapters) {
          val runtime=CudaDiffusionRuntime.upload(session,projection,offloadWeights=true)
          val x=session.upload(input)
          val output=runtime.conv(x,projection)
          val actual=output.toCpu.toFloatArray
          assert(actual.zip(expected).forall((a,b) => math.abs(a-b)<1e-4f))
          output.close();x.close();runtime.close()
          assertEquals(session.allocatedBytes,0L)
        }
      }
  }
  test("F64 LoRA factors and alpha convert to Float32") {
    val dir=Files.createTempDirectory("double-lora")
    val header="""{"unet.test.lora_down.weight":{"dtype":"F64","shape":[1,1],"data_offsets":[0,8]},"unet.test.lora_up.weight":{"dtype":"F64","shape":[1,1],"data_offsets":[8,16]},"unet.test.alpha":{"dtype":"F64","shape":[],"data_offsets":[16,24]}}""".getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val bytes=ByteBuffer.allocate(8+header.length+24).order(ByteOrder.LITTLE_ENDIAN)
    bytes.putLong(header.length.toLong).put(header).putDouble(2.0).putDouble(3.0).putDouble(0.5)
    val path=dir.resolve("adapter.safetensors")
    Files.write(path,bytes.array())
    val weight=Parameter(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L))),false)
    val adapters=DiffusionLoRA.load(path,Map("unet.test"->weight))
    val input=Variable(Tensor.fromFloats(Array(2f),Shape.from(Seq(1L,1L))))
    val result=LinearAdapterScope.withAdapters(adapters)(Linear(weight,None)(input).data.toFloatArray.toVector)
    assertEquals(result,Vector(8f))
  }
  test("CUDA convolution LoRA matches CPU with resident and offloaded weights") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"real GPU test disabled")
    val dir=Files.createTempDirectory("conv-lora")
    val weight=Parameter(Tensor.fromFloats(Array.fill(18)(0.1f),Shape.from(Seq(2L,1L,3L,3L))),false)
    val layer=Conv2d(weight,None,Conv2dConfig(stride=(2,2),padding=(1,1)))
    val path=write(dir,Vector(
      ("lora_unet_conv.lora_down.weight",Vector(1L,1L,3L,3L),Array.fill(9)(0.25f)),
      ("lora_unet_conv.lora_up.weight",Vector(2L,1L,1L,1L),Array(2f,-1f)),
      ("lora_unet_conv.alpha",Vector.empty,Array(1f))))
    val active=DiffusionLoRA.load(path,Map("unet.conv"->weight),unetScale=0.5f)
    val input=Tensor.fromFloats(Array.tabulate(25)(_.toFloat/25),Shape.from(Seq(1L,1L,5L,5L)))
    val expected=LinearAdapterScope.withAdapters(active)(layer(Variable(input)).data.toFloatArray)
    for offload <- Seq(false,true) do
      CudaSession.withDevice() { session =>
        LinearAdapterScope.withAdapters(active) {
          val runtime=CudaDiffusionRuntime.upload(session,layer,offloadWeights=offload)
          val deviceInput=session.upload(input)
          val result=runtime.conv(deviceInput,layer)
          val actual=result.toCpu.toFloatArray
          assertEquals(actual.length,expected.length)
          assert(actual.zip(expected).forall((a,b) => math.abs(a-b)<1e-5f))
          result.close();deviceInput.close();runtime.close()
          assertEquals(session.allocatedBytes,0L)
        }
      }
  }
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

  test("dynamic rank truncates LoRA factors and preserves alpha per original rank") {
    val dir=Files.createTempDirectory("dynamic-rank-lora")
    val weight=Parameter(Tensor.fromFloats(Array(0f),Shape.from(Seq(1L,1L))),false)
    val path=write(dir,Vector(
      ("unet.test.lora_down.weight",Vector(3L,1L),Array(1f,2f,3f)),
      ("unet.test.lora_up.weight",Vector(1L,3L),Array(4f,5f,6f)),
      ("unet.test.alpha",Vector.empty,Array(6f))))
    val input=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L))))
    def run(rank: Option[Int]) =
      val adapters=DiffusionLoRA.load(path,Map("unet.test"->weight),dynamicRank=rank)
      LinearAdapterScope.withAdapters(adapters)(Linear(weight,None)(input).data.toFloatArray.head)
    assertEquals(run(None),64f)
    assertEquals(run(Some(1)),8f)
    assertEquals(run(Some(-1)),28f)
    assertEquals(run(Some(0)),0f)
    assertEquals(run(Some(-4)),0f)
    assertEquals(run(Some(99)),64f)
  }

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
