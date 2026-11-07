package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import scalatorch.{Shape,Tensor}
import scalatorch.autograd.Variable
import scalatorch.models.{DiffusionImageIO,PretrainedVae}
import scala.concurrent.duration.*
import NativeJson.*

class NativeNineChannelUpscaleSuite extends munit.FunSuite:
  override val munitTimeout: Duration=180.seconds
  test("worker nine-channel unmasked latent upscale matches PyTorch images") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"public tiny model required")
    val sourceRoot=Path.of(configured.get).getParent
    val model=Files.createTempDirectory("nine-upscale-model")
    val output=Files.createTempDirectory("nine-upscale-output")
    try
      new NativeDiffusionBackendSuite().makeNineChannelModel(sourceRoot,model)
      val scheduler=output.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(model,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val original=new BufferedImage(16,16,BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 16; x <- 0 until 16 do
        val gray=y*16+x
        original.setRGB(x,y,(gray<<16)|(gray<<8)|gray)
      val input=output.resolve("source.png");ImageIO.write(original,"png",input.toFile)
      val stream=this.getClass.getResourceAsStream("/unmasked-inpaint-latent-upscale-reference.json")
      val fixture=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
      val vae=PretrainedVae.load(model.resolve("vae"))
      for sampler <- Vector("Euler","Euler a","DPM++ 2M"); strength <- Vector(.75,1.0) do
        val fields=Map[String,NativeJson]("prompt"->Str("a red cat"),"width"->Number(24),"height"->Number(24),
          "steps"->Number(if strength==1.0 then 4 else 3),"seed"->Number(42),"cfg_scale"->Number(1),"sampler_name"->Str(sampler),
          "extra"->Obj(Map("rng"->Str("cpu"))),"img2img"->Obj(Map("init_image"->Str(input.toString),
            "denoising_strength"->Number(BigDecimal(strength)))),"hooks"->Obj(Map("latent_upscale"->Obj(Map(
              "method"->Str("nearest-exact"),"scheduler"->Str("normal"))))))
        val response=backend.execute(NativeRequest("upscale","img2img",Map("params"->Obj(fields),"output_dir"->Str(output.toString)),None),context)
        val file=response("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value
        val actual=ImageIO.read(Path.of(file).toFile)
        val values=fixture(s"$sampler:$strength").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
        val expected=DiffusionImageIO.toBufferedImage(vae.decodeDiffusionLatents(Variable(Tensor.fromFloats(values,Shape.from(Seq(1L,4L,12L,12L))))).data)
        val error=(for y <- 0 until 24; x <- 0 until 24; shift <- Vector(0,8,16) yield
          math.abs(((actual.getRGB(x,y)>>>shift)&255)-((expected.getRGB(x,y)>>>shift)&255))).max
        assert(error<=1,s"$sampler at $strength differs from reference by $error")
    finally
      for dir <- Vector(model,output) do
        val paths=Files.walk(dir)
        try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
