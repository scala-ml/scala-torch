package scalatorch.models

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.Files
import scalatorch.{Shape,Tensor}
import scalatorch.nn.LinearAdapterScope
import scalatorch.models.worker.NativeJson
import NativeJson.*

class LoraCombinedReferenceSuite extends munit.FunSuite:
  override val munitTimeout: scala.concurrent.duration.Duration=scala.concurrent.duration.Duration(3,"minutes")
  test("block weights dynamic rank and step windows match Python trajectories") {
    val configured=sys.props.get("scalatorch.clip.dir");assume(configured.nonEmpty,"tiny SD fixture required")
    val root=java.nio.file.Path.of(configured.get).getParent
    val loaded=PretrainedUNet.loadWithLinearTargets(root.resolve("unet"))
    val name="unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q"
    val weight=loaded.linearTargets(name);val in=weight.data.shape.dims(1).toInt;val out=weight.data.shape.dims(0).toInt
    val down=Array.tabulate(2*in)(i=>if i<in then (math.sin(i*.31)*.2).toFloat else (math.cos((i-in)*.17)*.1).toFloat)
    val up=Array.tabulate(out*2)(i=>if i%2==0 then (math.cos((i/2)*.23)*30).toFloat else (math.sin((i/2)*.11)*15).toFloat)
    val entries=Vector((s"$name.lora_down.weight",Vector(2,in),down),(s"$name.lora_up.weight",Vector(out,2),up),
      (s"$name.alpha",Vector.empty,Array(2f)))
    var size=0
    val header=entries.map { (key,dims,values)=>val start=size;size+=values.length*4
      s"\"$key\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$start,$size]}"
    }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
    val data=ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);entries.foreach((_,_,values)=>values.foreach(data.putFloat))
    val file=Files.createTempFile("combined-lora-",".safetensors")
    try
      Files.write(file,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())
      val input=getClass.getResourceAsStream("/lora-combined-reference.json")
      val refs=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally input.close()
      val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
      val clip=PretrainedClip.load(root.resolve("text_encoder"));val vae=PretrainedVae.load(root.resolve("vae"))
      val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
      val cases=Vector(("ALL",2,None,None,"ALL|2|None|None"),("NONE",2,None,None,"NONE|2|None|None"),
        ("MIDD",1,Some(2.0),None,"MIDD|1|2|None"),("ALL",1,None,Some(2.0),"ALL|1|None|2"),
        ("ALL",0,Some(2.0),None,"ALL|0|2|None"))
      val placements=Vector(UNetExecution.Cpu)++Option.when(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"))(UNetExecution.Cuda(offloadWeights=true))
      for placement <- placements; (spec,rank,start,stop,label) <- cases do
        val loadedAdapter=DiffusionLoRA.load(file,Map(name->weight),dynamicRank=Some(rank))(weight).head._1
        val plan=LoraBlockWeights.plan(LoraBlockWeights.parse(spec,false),false)
        val scale=LoraBlockWeights.targetScale(plan,false,name).get.toFloat
        val window=LoraStepWindow(start,stop)
        val request=DiffusionImageRequest("a red cat","",16,16,42,guidanceScale=7.5f,torchCpuNoise=true,
          loraStepFilter=Some((step,total,_)=>window.active(step,total)))
        val pipe=new EulerImagePipeline(tokenizer,clip,loaded.model,vae,unetExecution=placement)
        val actual=scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
        LinearAdapterScope.withAdapters(Map(weight->Vector(loadedAdapter->scale))) {
          pipe.generateLatents(request,noise.euler(4),onProgress=p=>actual+=p.latents.toFloatArray)
        }
        val expected=refs(label).asInstanceOf[Arr].values
        assertEquals(actual.size,expected.size)
        for i <- actual.indices do
          val wanted=expected(i).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
          val error=actual(i).zip(wanted).map((a,b)=>math.abs(a.toDouble-b)).max
          assert(error<.002,s"$label ${placement.device} step=$i error=$error")
    finally Files.deleteIfExists(file)
  }
