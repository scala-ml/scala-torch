package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scalatorch.models.{PretrainedUNet,UNetExecution,VaeExecution}
import NativeJson.*

class NativeDynamicLoraSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("worker applies dynamic LoRA ranks in CUDA image generation") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny SD and CUDA required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-dynamic-lora")
    try
      val target=PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets("unet.time_embedding.linear_1")
      val in=target.data.shape.dims(1).toInt;val out=target.data.shape.dims(0).toInt
      val down=Array.tabulate(2*in)(i=>if i<in then 1f/in else -2f/in)
      val up=Array.tabulate(out*2)(i=>if i%2==0 then 8f else 5f)
      val entries=Vector(("unet.time_embedding.linear_1.lora_down.weight",Vector(2,in),down),
        ("unet.time_embedding.linear_1.lora_up.weight",Vector(out,2),up),
        ("unet.time_embedding.linear_1.alpha",Vector.empty,Array(2f)))
      var offset=0
      val header=entries.map { (name,dims,values) =>
        val start=offset;offset+=values.length*4
        s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$start,$offset]}"
      }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
      val data=ByteBuffer.allocate(offset).order(ByteOrder.LITTLE_ENDIAN)
      entries.foreach((_,_,values)=>values.foreach(data.putFloat))
      val adapter=dir.resolve("rank2.safetensors")
      Files.write(adapter,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())
      import scalatorch.io.{PyTorchTensorArchive as A,PyTorchTensorArchiveWriter as Writer}
      val torchAdapter=dir.resolve("rank2.pt")
      Writer.write(torchAdapter,A.DictValue(entries.map { (name,dims,values) =>
        name->A.TensorValue(A.StoredTensor(A.StorageDType.Float32,dims.map(_.toLong),values))
      }.toMap))
      val scheduler=dir.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      def generate(rank: Option[Int],useAdapter: Boolean=true,adapterPath: Path=adapter): Vector[Int] =
        val lora=Map[String,NativeJson]("name"->Str("rank2"),"weight"->Number(1))++rank.map(r=>"dyn"->Number(r))
        val params=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
          "steps"->Number(4),"seed"->Number(42),"loras"->Arr(if useAdapter then Vector(Obj(lora)) else Vector.empty))
        val response=backend.execute(NativeRequest("generate","txt2img",Map("params"->Obj(params),"output_dir"->Str(dir.toString),
          "lora_paths"->Obj(Map("rank2"->Str(adapterPath.toString)))),None),context)
        val path=Path.of(response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,16,16,null,0,16).toVector
      val plain=generate(None,false)
      assertEquals(generate(Some(0)),plain)
      val rank1=generate(Some(1));val rank2=generate(Some(2))
      assertNotEquals(rank1,plain);assertNotEquals(rank2,plain);assertNotEquals(rank1,rank2)
      assertEquals(generate(Some(-1)),rank1)
      assertEquals(generate(None),rank2)
      assertEquals(generate(Some(1),adapterPath=torchAdapter),rank1)
      assertEquals(generate(None,adapterPath=torchAdapter),rank2)
      def tagged(tag: String): Vector[Int] =
        val params=Obj(Map("prompt"->Str(s"a red cat $tag"),"width"->Number(16),"height"->Number(16),
          "steps"->Number(4),"seed"->Number(42)))
        val result=backend.execute(NativeRequest("tag","txt2img",Map("params"->params,"output_dir"->Str(dir.toString),
          "lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,16,16,null,0,16).toVector
      assertEquals(tagged("<lora:rank2:1:dyn=1>"),rank1)
      assertEquals(tagged("<lyco:rank2:1:1:-1>"),rank1)
      assertEquals(tagged("<lora:rank2:1:start=5>"),plain)
      assertEquals(tagged("<lora:rank2:1:stop=1>"),plain)
      assertEquals(tagged("<lora:rank2:1:start=1:stop=5>"),rank2)
      val secondOnly=tagged("<lora:rank2:1:start=2>")
      assertNotEquals(secondOnly,plain)
      assertNotEquals(secondOnly,rank2)
      assertEquals(tagged("<lora:rank2:1:step=2-5>"),secondOnly)
      val scheduled=Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(2)))
      val scheduledParams=Obj(Map("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(4),"seed"->Number(42),"loras"->Arr(Vector(scheduled))))
      val scheduledResult=backend.execute(NativeRequest("scheduled","txt2img",Map("params"->scheduledParams,
        "output_dir"->Str(dir.toString),"lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
      val scheduledPath=Path.of(scheduledResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assertEquals(ImageIO.read(scheduledPath.toFile).getRGB(0,0,16,16,null,0,16).toVector,secondOnly)
      val mixed=Vector(0,1).map(rank=>Obj(Map("name"->Str("rank2"),"weight"->Number(1),"dyn"->Number(rank))))
      val mixParams=Obj(Map("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
        "steps"->Number(4),"seed"->Number(42),"loras"->Arr(mixed)))
      val mixedResult=backend.execute(NativeRequest("mixed","txt2img",Map("params"->mixParams,
        "output_dir"->Str(dir.toString),"lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
      val mixedPath=Path.of(mixedResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assertEquals(ImageIO.read(mixedPath.toFile).getRGB(0,0,16,16,null,0,16).toVector,rank1,
        "different ranks of one file must have independent cache entries")
      val windowMix=Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(5))),
        Obj(Map("name"->Str("rank2"),"weight"->Number(1),"dyn"->Number(1))))
      val windowMixParams=Obj(mixParams.asInstanceOf[Obj].fields.updated("loras",Arr(windowMix)))
      val windowMixResult=backend.execute(NativeRequest("mixed-window","txt2img",Map("params"->windowMixParams,
        "output_dir"->Str(dir.toString),"lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
      val windowMixPath=Path.of(windowMixResult("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assertEquals(ImageIO.read(windowMixPath.toFile).getRGB(0,0,16,16,null,0,16).toVector,rank1,
        "an inactive window must not disable another adapter from the same file")
      val source=new java.awt.image.BufferedImage(16,16,java.awt.image.BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 16; x <- 0 until 16 do source.setRGB(x,y,((x*16)<<16)|((y*16)<<8)|64)
      val sourcePath=dir.resolve("source.png");ImageIO.write(source,"png",sourcePath.toFile)
      def imageRun(refs: Vector[NativeJson]): Vector[Int] =
        val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(16),"height"->Number(16),
          "steps"->Number(4),"seed"->Number(42),"loras"->Arr(refs),
          "img2img"->Obj(Map("init_image"->Str(sourcePath.toString),"denoising_strength"->Number(BigDecimal("0.5")),"resize_mode"->Number(0)))))
        val result=backend.execute(NativeRequest("img-window","img2img",Map("params"->params,"output_dir"->Str(dir.toString),
          "lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,16,16,null,0,16).toVector
      val imagePlain=imageRun(Vector.empty)
      val imageFull=imageRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1)))))
      val imageOff=imageRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(3)))))
      val imageOn=imageRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"stop"->Number(3)))))
      assertNotEquals(imageFull,imagePlain)
      assertEquals(imageOff,imagePlain)
      assertEquals(imageOn,imageFull)
      def hiresRun(refs: Vector[NativeJson]): Vector[Int] =
        val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(64),"height"->Number(64),
          "steps"->Number(2),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),
          "hires"->Obj(Map("enabled"->Bool(true),"scale"->Number(1),"upscaler"->Str("Latent (nearest)"),
            "second_pass_steps"->Number(4),"denoising_strength"->Number(BigDecimal("0.5"))))))
        val result=backend.execute(NativeRequest("hires-window","txt2img",Map("params"->params,"output_dir"->Str(dir.toString),
          "hires_loras"->Arr(refs),"lora_paths"->Obj(Map("rank2"->Str(adapter.toString)))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,64,64,null,0,64).toVector
      val hiresPlain=hiresRun(Vector.empty)
      val hiresFull=hiresRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1)))))
      val hiresOff=hiresRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(5)))))
      val hiresAll=hiresRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"stop"->Number(5)))))
      assertNotEquals(hiresFull,hiresPlain)
      assertEquals(hiresOff,hiresPlain)
      assertEquals(hiresAll,hiresFull)
      val hiresPartial=hiresRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(2)))))
      val hiresFraction=hiresRun(Vector(Obj(Map("name"->Str("rank2"),"weight"->Number(1),"start"->Number(BigDecimal("0.5"))))))
      assertNotEquals(hiresPartial,hiresFull)
      assertNotEquals(hiresPartial,hiresPlain)
      assertEquals(hiresFraction,hiresPartial,"four Hires steps map 0.5 to step 2")
      val attnKey="unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q"
      val attn=PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets(attnKey)
      val attnIn=attn.data.shape.dims(1).toInt;val attnOut=attn.data.shape.dims(0).toInt
      val parts=Vector((s"$attnKey.lora_down.weight",Vector(1,attnIn),Array.tabulate(attnIn)(i=>(math.sin(i*.31)*.2).toFloat)),
        (s"$attnKey.lora_up.weight",Vector(attnOut,1),Array.tabulate(attnOut)(i=>(math.cos(i*.23)*30).toFloat)))
      var size=0
      val attnHeader=parts.map { (name,dims,values) =>
        val begin=size;size+=values.length*4
        s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$begin,$size]}"
      }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
      val attnData=ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
      parts.foreach((_,_,values)=>values.foreach(attnData.putFloat))
      val attnFile=dir.resolve("attention.safetensors")
      Files.write(attnFile,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(attnHeader.length.toLong).array()++attnHeader++attnData.array())
      def blockRun(spec: Option[String],tagged: Boolean=false,rank: Option[Int]=None,start: Option[Int]=None): Vector[Int] =
        val lora=Obj(Map[String,NativeJson]("name"->Str("attention"),"weight"->Number(1))++spec.map(value=>"lbw"->Str(value))++
          rank.map(value=>"dyn"->Number(value))++start.map(value=>"start"->Number(value)))
        val prompt=if tagged then s"a red cat <lora:attention:1:lbw=${spec.get}${rank.fold("")(v=>s":dyn=$v")}${start.fold("")(v=>s":start=$v")}>" else "a red cat"
        val params=Obj(Map("prompt"->Str(prompt),"width"->Number(16),"height"->Number(16),"steps"->Number(4),"seed"->Number(42),
          "loras"->Arr(if tagged then Vector.empty else Vector(lora))))
        val result=backend.execute(NativeRequest("block","txt2img",Map("params"->params,"output_dir"->Str(dir.toString),
          "lora_paths"->Obj(Map("attention"->Str(attnFile.toString))),"lbw_presets"->Obj(Map("CUSTOM"->Str("ALL")))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,16,16,null,0,16).toVector
      val attentionFull=blockRun(None)
      assertNotEquals(attentionFull,plain)
      assertEquals(blockRun(Some("ALL")),attentionFull)
      assertEquals(blockRun(Some("NONE")),plain)
      assertEquals(blockRun(Some("INS")),plain)
      assertEquals(blockRun(Some("CUSTOM")),attentionFull)
      assertEquals(blockRun(Some("ALL"),tagged=true),attentionFull)
      assertEquals(blockRun(Some("MIDD"),rank=Some(0),start=Some(2)),plain)
      assertEquals(blockRun(Some("MIDD"),rank=Some(1),start=Some(5)),plain)
      val combined=blockRun(Some("MIDD"),rank=Some(1),start=Some(2))
      assertNotEquals(combined,plain)
      assertNotEquals(combined,attentionFull)
      assertEquals(blockRun(Some("MIDD"),tagged=true,rank=Some(1),start=Some(2)),combined)
    finally
      val paths=Files.walk(dir)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
