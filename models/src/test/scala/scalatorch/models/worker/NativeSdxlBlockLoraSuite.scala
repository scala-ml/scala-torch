package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scalatorch.models.{PretrainedUNet,UNetExecution,VaeExecution}
import NativeJson.*

class NativeSdxlBlockLoraSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("SDXL block-weight LoRA selects the requested attention layer on CUDA") {
    val configured=sys.props.get("scalatorch.sdxl.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny SDXL and CUDA required")
    val root=Path.of(configured.get);val dir=Files.createTempDirectory("native-sdxl-block-lora")
    try
      val targets=PretrainedUNet.loadWithLinearTargets(root.resolve("unet")).linearTargets
      val name="unet.mid_block.attentions.0.transformer_blocks.0.attn1.to_q"
      val target=targets(name);val in=target.data.shape.dims(1).toInt;val out=target.data.shape.dims(0).toInt
      val entries=Vector((s"$name.lora_down.weight",Vector(1,in),Array.tabulate(in)(i=>(math.sin(i*.31)*.2).toFloat)),
        (s"$name.lora_up.weight",Vector(out,1),Array.tabulate(out)(i=>(math.cos(i*.23)*30).toFloat)))
      var size=0
      val header=entries.map { (key,dims,values)=>
        val start=size;size+=values.length*4
        s"\"$key\":{\"dtype\":\"F32\",\"shape\":[${dims.mkString(",")}],\"data_offsets\":[$start,$size]}"
      }.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
      val data=ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
      entries.foreach((_,_,values)=>values.foreach(data.putFloat))
      val lora=dir.resolve("mid.safetensors")
      Files.write(lora,ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()++header++data.array())
      val backend=new NativeDiffusionBackend(root,root.resolve("scheduler/scheduler_config.json"),
        vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      def generate(spec: Option[String],enabled: Boolean=true): Vector[Int] =
        val ref=Obj(Map[String,NativeJson]("name"->Str("mid"),"weight"->Number(1))++spec.map(x=>"lbw"->Str(x)))
        val params=Obj(Map("prompt"->Str("a red cat"),"width"->Number(8),"height"->Number(8),"steps"->Number(4),
          "seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),
          "loras"->Arr(if enabled then Vector(ref) else Vector.empty)))
        val result=backend.execute(NativeRequest("generate","txt2img",Map("params"->params,"output_dir"->Str(dir.toString),
          "lora_paths"->Obj(Map("mid"->Str(lora.toString)))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        ImageIO.read(path.toFile).getRGB(0,0,8,8,null,0,8).toVector
      val plain=generate(None,false);val full=generate(None)
      assertNotEquals(full,plain)
      assertEquals(generate(Some("ALL")),full)
      assertEquals(generate(Some("MIDD")),full)
      assertEquals(generate(Some("NONE")),plain)
      assertEquals(generate(Some("INS")),plain)
    finally
      val paths=Files.walk(dir)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
