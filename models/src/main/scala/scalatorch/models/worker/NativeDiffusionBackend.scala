package scalatorch.models.worker

import scalatorch.models.*
import java.nio.file.{Files,Path}
import java.util.UUID

final class NativeBackendError(val code: String,message: String) extends RuntimeException(message)

/** Explicit CPU Float32 checkpoint adapter. Unsupported settings fail before inference. */
final class NativeDiffusionBackend(modelRoot: Path,schedulerConfig: Path) extends NativeWorkerBackend:
  import NativeJson.*
  private val root=modelRoot.toRealPath()
  private val config=EulerConfig.read(schedulerConfig)
  private var loaded: Option[EulerImagePipeline]=None
  private def checkpointHash(context: NativeWorkContext): String=
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val weights=Vector("text_encoder"->"model.safetensors","unet"->"diffusion_pytorch_model.safetensors",
      "vae"->"diffusion_pytorch_model.safetensors").flatMap { (part,name) =>
      scala.util.Using.resource(scalatorch.io.SafeTensorSet.open(root.resolve(part),filename=name)) { set =>
        Vector(s"$part/config.json")++set.sourceFiles.map(p => root.relativize(p).toString.replace('\\','/'))
      }
    }
    val names=weights++Vector("tokenizer/vocab.json","tokenizer/merges.txt")
    names.foreach { name =>
      digest.update(name.getBytes(java.nio.charset.StandardCharsets.UTF_8))
      digest.update(0.toByte)
      val input=Files.newInputStream(root.resolve(name))
      try
        val buffer=new Array[Byte](65536)
        var count=input.read(buffer)
        while count != -1 do
          context.checkCancelled()
          digest.update(buffer,0,count)
          count=input.read(buffer)
      finally input.close()
    }
    java.util.HexFormat.of().formatHex(digest.digest())
  val supported=Set("load_model","unload","txt2img","list_models","list_samplers")
  private def unsupported(message: String): Nothing=throw new NativeBackendError("unsupported",message)
  private def keys(p: Map[String,NativeJson],allowed: Set[String]): Unit=
    val unknown=p.keySet--allowed
    if unknown.nonEmpty then unsupported(s"unsupported parameters: ${unknown.toVector.sorted.mkString(", ")}")
  private def str(p: Map[String,NativeJson],key: String,default: String): String=p.get(key) match
    case None => default
    case Some(Str(s)) => s
    case _ => throw IllegalArgumentException(s"$key must be a string")
  private def number(p: Map[String,NativeJson],key: String,default: BigDecimal): BigDecimal=p.get(key) match
    case None => default
    case Some(Number(n)) => n
    case _ => throw IllegalArgumentException(s"$key must be numeric")
  private def integer(p: Map[String,NativeJson],key: String,default: Int): Int=
    val n=number(p,key,BigDecimal(default))
    require(n.isValidInt,s"$key must be an integer")
    n.toInt

  def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson]=
    val p=request.params
    context.checkCancelled()
    request.kind match
      case "list_models" =>
        keys(p,Set("dirs"))
        val directories=p.get("dirs") match
          case Some(Arr(values)) => values.map {
            case Str(path) => Path.of(path).toAbsolutePath.normalize()
            case _ => throw IllegalArgumentException("dirs must contain strings")
          }
          case _ => throw IllegalArgumentException("dirs must be an array")
        // Advertise only the configured checkpoint that this worker can load.
        val visible=directories.exists(path => Files.isDirectory(path) && root.startsWith(path.toRealPath()))
        val entries=if !visible then Vector.empty else
          val files=Files.walk(root)
          val bytes=try
            val iterator=files.iterator()
            var total=BigInt(0)
            while iterator.hasNext do
              context.checkCancelled()
              val file=iterator.next()
              if Files.isRegularFile(file,java.nio.file.LinkOption.NOFOLLOW_LINKS) then total+=Files.size(file)
            total
          finally files.close()
          Vector(Obj(Map("name"->Str(root.getFileName.toString),"path"->Str(root.toString),
            "kind"->Str("diffusers"),"size_mb"->Number(BigDecimal(bytes)/BigDecimal(1048576)))))
        Map("models"->Arr(entries))
      case "list_samplers" =>
        keys(p,Set.empty)
        def sampler(name: String,schedule: String)=Obj(Map("name"->Str(name),"supported"->Bool(true),"default_scheduler"->Str(schedule)))
        Map("samplers"->Arr(Vector(sampler("Euler","Uniform"),sampler("DPM++ 2M","Karras"))),
          "schedulers"->Arr(Vector("Automatic","Uniform","Karras").map(Str.apply)),
          "upscalers"->Arr(Vector.empty),"preprocessors"->Arr(Vector.empty))
      case "load_model" =>
        keys(p,Set("path","dtype","vram_mode","vae_path","vae_tiling"))
        require(Path.of(str(p,"path",root.toString)).toRealPath()==root,"model path differs from configured checkpoint")
        if str(p,"dtype","fp32")!="fp32" || str(p,"vram_mode","cpu")!="cpu" then
          unsupported("native adapter currently requires fp32 and cpu")
        if p.get("vae_path").exists(_!=Null) || p.get("vae_tiling").exists(_!=Bool(false)) then
          unsupported("external VAE and tiling are not implemented")
        val tokenizer=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
        context.checkCancelled()
        val clip=PretrainedClip.load(root.resolve("text_encoder"))
        context.checkCancelled()
        val unet=PretrainedUNet.load(root.resolve("unet"))
        context.checkCancelled()
        val vae=PretrainedVae.load(root.resolve("vae"))
        val pipeline=new EulerImagePipeline(tokenizer,clip,unet,vae)
        val hash=checkpointHash(context)
        context.beginPublication()
        loaded=Some(pipeline)
        Map("name"->Str(root.getFileName.toString),"path"->Str(root.toString),"kind"->Str("diffusers"),
          "hash"->Str(hash),"model_token"->Str(UUID.randomUUID().toString))
      case "unload" =>
        keys(p,Set.empty)
        context.beginPublication()
        loaded=None
        Map.empty
      case "txt2img" =>
        val pipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a model first"))
        keys(NativeGenerationDefaults.strip(p,NativeGenerationDefaults.envelope),Set("params","output_dir","preview_every","preview_method"))
        val previewEvery=integer(p,"preview_every",2)
        require(previewEvery>=0,"preview_every must be nonnegative")
        val previewMethod=str(p,"preview_method","approx")
        if !Set("none","approx").contains(previewMethod) then unsupported(s"unsupported preview method: $previewMethod")
        val g=p.get("params") match
          case Some(Obj(fields)) => fields
          case _ => throw IllegalArgumentException("params must be an object")
        keys(NativeGenerationDefaults.strip(g,NativeGenerationDefaults.generation),Set("prompt","negative_prompt","width","height","steps","seed","cfg_scale","sampler_name","scheduler","clip_skip"))
        val seedNumber=number(g,"seed",42)
        require(seedNumber.isValidLong,"seed must be a signed 64-bit integer")
        val seed=if seedNumber.toLong == -1 then new java.security.SecureRandom().nextLong() & Long.MaxValue else seedNumber.toLong
        val steps=integer(g,"steps",20)
        require(steps>0 && steps<=1000,"steps must be between 1 and 1000")
        val imageRequest=DiffusionImageRequest(str(g,"prompt",""),str(g,"negative_prompt",""),
          integer(g,"width",512),integer(g,"height",512),seed,number(g,"cfg_scale",7).toFloat,clipSkip=integer(g,"clip_skip",1))
        val sampler=str(g,"sampler_name","DPM++ 2M")
        val requestedSchedule=str(g,"scheduler","Automatic")
        val schedule=if requestedSchedule=="Automatic" then (if sampler=="DPM++ 2M" then "Karras" else "Uniform") else requestedSchedule
        if !(sampler=="Euler" && Set("Uniform","Karras").contains(schedule)) &&
            !(sampler=="DPM++ 2M" && Set("Uniform","Karras").contains(schedule)) then unsupported(s"unsupported sampler/schedule: $sampler/$schedule")
        val outputName=str(p,"output_dir","")
        require(outputName.nonEmpty,"output_dir is required")
        val output=Path.of(outputName).toAbsolutePath.normalize()
        require(Files.isDirectory(output),"output_dir must be an existing directory")
        val started=System.nanoTime()
        val progress: DiffusionImageProgress => Unit = value =>
          context.event("progress",Map(
            "step"->Number(value.completedSteps),"total"->Number(value.totalSteps),"pass"->Str("base"),
            "eta_s"->Number(0),"image"->Number(0),"images"->Number(1),"batch_size"->Number(1)))
          if previewEvery>0 && previewMethod!="none" && value.completedSteps%previewEvery==0 && value.completedSteps<value.totalSteps then
            context.checkCancelled()
            context.event("preview",Map("step"->Number(value.completedSteps),"jpeg_b64"->Str(NativePreview.jpegBase64(value.latents))))
        val image=if sampler=="DPM++ 2M" then
          val layout=if schedule=="Karras" then DpmSchedule.karras(config.noise,steps) else DpmSchedule.uniform(config.noise,steps)
          pipeline.generateDpm(imageRequest,layout,onProgress=progress,
            cancelled=context.cancelled,prediction=config.prediction)
        else
          val layout=if schedule=="Karras" then config.noise.karrasEuler(steps) else config.noise.euler(steps,config.spacing,config.stepsOffset)
          pipeline.generate(imageRequest,layout,onProgress=progress,cancelled=context.cancelled,prediction=config.prediction)
        context.beginPublication()
        val path=output.resolve(s"native-${UUID.randomUUID()}.png")
        DiffusionImageIO.writePng(image,path)
        Map("images"->Arr(Vector(Obj(Map("path"->Str(path.toString),"seed"->Number(BigDecimal(seed)),
          "width"->Number(imageRequest.width),"height"->Number(imageRequest.height))))),
          "timing_s"->Number(BigDecimal((System.nanoTime()-started)/1e9)),"model_name"->Str(root.getFileName.toString),
          "scheduler"->Str(schedule))
      case _ => unsupported(s"unsupported request: ${request.kind}")

object NativeWorkerMain:
  def main(args: Array[String]): Unit=
    require(args.length==2,"usage: NativeWorkerMain <diffusers-root> <explicit-euler-config.json>")
    val backend=new NativeDiffusionBackend(Path.of(args(0)),Path.of(args(1)))
    val worker=new NativeWorker(System.out,backend)
    worker.serve(new java.io.InputStreamReader(System.in,java.nio.charset.StandardCharsets.UTF_8))
