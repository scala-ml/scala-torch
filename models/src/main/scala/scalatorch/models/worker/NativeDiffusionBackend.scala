package scalatorch.models.worker

import scalatorch.models.*
import scalatorch.nn.LinearAdapterScope
import java.nio.file.{Files,Path}
import java.util.UUID

final class NativeBackendError(val code: String,message: String) extends RuntimeException(message)

/** Host-loaded Float32 models with explicit U-Net/VAE execution placement. */
final class NativeDiffusionBackend(modelRoot: Path,schedulerConfig: Path,
    vaeExecution: VaeExecution=VaeExecution.Cpu,unetExecution: UNetExecution=UNetExecution.Cpu,
    maxAttentionScoreElements: Long=16777216L) extends NativeWorkerBackend:
  require(maxAttentionScoreElements>=0 && maxAttentionScoreElements<=Int.MaxValue,"invalid attention work limit")
  import NativeJson.*
  private val root=modelRoot.toRealPath()
  private val config=EulerConfig.read(schedulerConfig)
  private var loaded: Option[EulerImagePipeline]=None
  private var linearTargets=Map.empty[String,scalatorch.nn.Parameter]
  private var keywordState=ModelKeyword.State()
  @volatile private var modelSummary=Map.empty[String,NativeJson]
  override def device: String=unetExecution.device
  override def usesCuda: Boolean=vaeExecution.device!="cpu" || unetExecution.device!="cpu"
  override def systemStatus: Map[String,NativeJson]=NativeSystemStatus.snapshot(modelSummary,vaeExecution.device,unetExecution.device,
    vaeExecution.offloadWeights,unetExecution.offloadWeights,maxAttentionScoreElements)
  private def sdxlProfile(): Boolean =
    val addition=UNetConfig.read(root.resolve("unet/config.json")).additionTimeWidth.nonEmpty
    val index=root.resolve("model_index.json")
    if Files.exists(index) then
      val name=TokenizerJson.read(index).obj("_class_name").str
      val xl=Set("StableDiffusionXLPipeline","StableDiffusionXLImg2ImgPipeline").contains(name)
      require(xl || Set("StableDiffusionPipeline","StableDiffusionImg2ImgPipeline").contains(name),"unsupported diffusion model family")
      require(xl==addition,"model index and U-Net conditioning disagree")
    require(addition || (!Files.exists(root.resolve("text_encoder_2")) && !Files.exists(root.resolve("tokenizer_2"))),
      "second text encoder requires SDXL text-time conditioning")
    addition
  private def checkpointHash(context: NativeWorkContext): String=
    val sdxl=sdxlProfile()
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    val parts=Vector("text_encoder"->"model.safetensors","unet"->"diffusion_pytorch_model.safetensors",
      "vae"->"diffusion_pytorch_model.safetensors")++(if sdxl then Vector("text_encoder_2"->"model.safetensors") else Vector.empty)
    val weights=parts.flatMap { (part,name) =>
      scala.util.Using.resource(scalatorch.io.SafeTensorSet.open(root.resolve(part),filename=name)) { set =>
        Vector(s"$part/config.json")++set.sourceFiles.map(p => root.relativize(p).toString.replace('\\','/'))
      }
    }
    val textFiles=Vector("tokenizer/vocab.json","tokenizer/merges.txt")++(if sdxl then
      Vector("tokenizer_2/vocab.json","tokenizer_2/merges.txt")++
        Vector("model_index.json","tokenizer/tokenizer_config.json","tokenizer_2/tokenizer_config.json").filter(n => Files.exists(root.resolve(n)))
      else Vector.empty)
    val names=weights++textFiles
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
  val supported=Set("load_model","set_vae","unload","txt2img","img2img","list_models","list_samplers","list_files","model_keyword")
  private def vaeChoice(params: Map[String,NativeJson],key: String): Option[Path]=params.get(key) match
    case None | Some(Null) | Some(Str("")) => None
    case Some(Str(value)) => Some(Path.of(value).toRealPath())
    case _ => throw IllegalArgumentException(s"$key must be a path or null")
  private def readVae(path: Option[Path],context: NativeWorkContext): (VaeModel,Map[String,NativeJson])=
    val selected=path.getOrElse(root.resolve("vae"))
    val directory=Files.isDirectory(selected)
    val folder=if directory then selected else selected.getParent
    val filename=if directory then "diffusion_pytorch_model.safetensors" else selected.getFileName.toString
    val vae=PretrainedVae.load(folder,maxScoreElements=maxAttentionScoreElements,
      configOverride=if directory then None else Some(VaeConfig.read(root.resolve("vae/config.json"))),filename=filename)
    context.checkCancelled()
    val digest=java.security.MessageDigest.getInstance("SHA-256")
    scala.util.Using.resource(scalatorch.io.SafeTensorSet.open(folder,filename=filename)) { set =>
      set.sourceFiles.foreach { file =>
        val input=Files.newInputStream(file)
        try
          val bytes=new Array[Byte](65536)
          var n=input.read(bytes)
          while n != -1 do
            context.checkCancelled()
            digest.update(bytes,0,n)
            n=input.read(bytes)
        finally input.close()
      }
    }
    val info=if path.isEmpty then Map("vae_name"->Null,"vae_hash"->Null) else
      Map("vae_name"->Str(selected.getFileName.toString),"vae_hash"->Str(java.util.HexFormat.of().formatHex(digest.digest())))
    (vae,info)
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
  private def bool(p: Map[String,NativeJson],key: String,default: Boolean): Boolean=p.get(key) match
    case None => default
    case Some(Bool(value)) => value
    case _ => throw IllegalArgumentException(s"$key must be boolean")

  def execute(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson]=
    val p=request.params
    context.checkCancelled()
    request.kind match
      case "model_keyword" =>
        NativeKeywordMappings.execute(p,root,() => modelSummary.get("hash") match
          case Some(Str(hash)) => hash
          case _ => checkpointHash(context)
        ,context)
      case "list_files" =>
        keys(p,Set("kind","dirs"))
        val dirs=p.get("dirs") match
          case Some(Arr(values)) => values.map {
            case Str(path) => path
            case _ => throw IllegalArgumentException("dirs must contain strings")
          }
          case _ => throw IllegalArgumentException("dirs must be an array")
        Map("files"->Arr(NativeFileCatalog.scan(str(p,"kind",""),dirs,() => context.checkCancelled())))
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
          unsupported("native parameter loading requires fp32 and cpu; U-Net/VAE execution placement is configured separately")
        if p.get("vae_tiling").exists(_!=Bool(false)) then unsupported("VAE tiling is not implemented")
        val selectedVae=vaeChoice(p,"vae_path")
        val sdxl=sdxlProfile()
        val tokenizer=if sdxl then ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")) else
          ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))
        context.checkCancelled()
        val clipLoaded=PretrainedClip.loadWithLinearTargets(root.resolve("text_encoder"))
        val clip=clipLoaded.model
        context.checkCancelled()
        val (sdxlText,clip2Targets)=if !sdxl then (None,Map.empty[String,scalatorch.nn.Parameter]) else
          val tokenizer2=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2"))
          val clip2Loaded=PretrainedClip.loadWithLinearTargets(root.resolve("text_encoder_2"),aliasPrefix="text_encoder_2")
          val clip2=clip2Loaded.model
          context.checkCancelled()
          (Some(new SdxlTextConditioning(tokenizer,clip,tokenizer2,clip2)),clip2Loaded.linearTargets)
        val unetLoaded=PretrainedUNet.loadWithLinearTargets(root.resolve("unet"),maxScoreElements=maxAttentionScoreElements,
          maxParameters=if sdxl then 3000000000L else 1000000000L)
        val unet=unetLoaded.model
        val nextLinearTargets=clipLoaded.linearTargets++clip2Targets++unetLoaded.linearTargets
        context.checkCancelled()
        val (vae,vaeInfo)=readVae(selectedVae,context)
        val pipeline=new EulerImagePipeline(tokenizer,clip,unet,vae,vaeExecution=vaeExecution,unetExecution=unetExecution,sdxlConditioning=sdxlText)
        val hash=checkpointHash(context)
        context.beginPublication()
        loaded=Some(pipeline)
        linearTargets=nextLinearTargets
        modelSummary=Map("name"->Str(root.getFileName.toString),"path"->Str(root.toString),"kind"->Str("diffusers"),
          "hash"->Str(hash),"family"->Str(if sdxl then "sdxl" else "sd"),"model_token"->Str(UUID.randomUUID().toString))++vaeInfo
        modelSummary
      case "set_vae" =>
        keys(p,Set("path"))
        val previous=loaded.getOrElse(throw new NativeBackendError("no_model","load a model first"))
        val (vae,info)=readVae(vaeChoice(p,"path"),context)
        require(vae.config.latentChannels==previous.vae.config.latentChannels &&
          vae.config.channels.size==previous.vae.config.channels.size,"replacement VAE latent geometry differs")
        val replacement=new EulerImagePipeline(previous.tokenizer,previous.textEncoder,previous.unet,vae,
          vaeExecution=previous.vaeExecution,unetExecution=previous.unetExecution,sdxlConditioning=previous.sdxlConditioning)
        context.beginPublication()
        loaded=Some(replacement)
        modelSummary=modelSummary++info++Map("model_token"->Str(UUID.randomUUID().toString))
        modelSummary
      case "unload" =>
        keys(p,Set.empty)
        context.beginPublication()
        loaded=None
        linearTargets=Map.empty
        modelSummary=Map.empty
        Map.empty
      case "txt2img" | "img2img" =>
        val pipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a model first"))
        val loraPaths=p.get("lora_paths") match
          case None | Some(Null) => Map.empty[String,String]
          case Some(Obj(paths)) =>
            require(paths.size<=32,"too many resolved LoRA paths")
            paths.map { (name,value) =>
            require(name.nonEmpty && name.length<=256,"invalid resolved LoRA name")
            val path=value match
              case Str(text) if text.nonEmpty && text.length<=4096 => text
              case _ => throw IllegalArgumentException(s"lora_paths.$name must be a path")
            name->path
          }
          case _ => throw IllegalArgumentException("lora_paths must be an object")
        keys(NativeGenerationDefaults.strip(p--Set("lora_paths"),NativeGenerationDefaults.envelope),Set("params","output_dir","preview_every","preview_method","vae_recovery_dir"))
        val previewEvery=integer(p,"preview_every",2)
        require(previewEvery>=0,"preview_every must be nonnegative")
        val previewMethod=str(p,"preview_method","approx")
        if !Set("none","approx").contains(previewMethod) then unsupported(s"unsupported preview method: $previewMethod")
        val g=p.get("params") match
          case Some(Obj(fields)) => fields
          case _ => throw IllegalArgumentException("params must be an object")
        val img2img=request.kind=="img2img"
        val defaults=if img2img then NativeGenerationDefaults.generation.removed("img2img") else NativeGenerationDefaults.generation
        val allowed=Set("prompt","negative_prompt","width","height","steps","seed","subseed","cfg_scale","sampler_name","scheduler","clip_skip","batch_size","n_iter","checkpoint","vae","extra","hooks")
        val requestedLoras=g.get("loras") match
          case None => Vector.empty
          case Some(Arr(values)) if values.isEmpty => Vector.empty
          case Some(Arr(values)) =>
            require(values.size<=16,"at most 16 LoRAs may be applied per request")
            values.map {
            case Obj(fields) =>
              require(fields.keySet.subsetOf(Set("name","weight","unet","lbw","start","stop","dyn")),"unsupported LoRA fields")
              if Set("lbw","start","stop","dyn").exists(key=>fields.get(key).exists(_!=Null)) then
                unsupported("LoRA block weights, schedules and dynamic rank are not implemented")
              val name=fields.get("name") match
                case Some(Str(value)) if value.nonEmpty => value
                case _ => throw IllegalArgumentException("LoRA name is required")
              def scale(key: String,default: Double): Float = fields.get(key) match
                case None | Some(Null) => default.toFloat
                case Some(Number(value)) =>
                  val converted=value.toFloat
                  require(value.toDouble.isFinite && converted.isFinite,s"LoRA $key scale must be finite")
                  converted
                case _ => throw IllegalArgumentException(s"LoRA $key scale must be numeric")
              val textScale=scale("weight",1.0)
              (name,textScale,scale("unet",textScale.toDouble))
            case _ => throw IllegalArgumentException("loras must contain objects")
          }
          case _ => throw IllegalArgumentException("loras must be an array")
        keys(NativeGenerationDefaults.strip(g--Set("loras"),defaults),if img2img then allowed+"img2img"+"loras" else allowed+"loras")
        var remainingAdapterElements=50000000L
        val adapters=requestedLoras.flatMap { (name,textScale,unetScale) =>
          val path=loraPaths.getOrElse(name,throw IllegalArgumentException(s"missing resolved path for LoRA: $name"))
          val file=Path.of(path).toAbsolutePath.normalize()
          require(file.getFileName.toString.endsWith(".safetensors") && Files.isRegularFile(file),s"LoRA safetensors file is missing: $name")
          val loadedAdapter=DiffusionLoRA.load(file,linearTargets,textScale,unetScale,remainingAdapterElements)
          val used=loadedAdapter.valuesIterator.flatten.map { case (adapter,_) =>
            adapter.a.data.shape.numel+adapter.b.data.shape.numel
          }.sum
          remainingAdapterElements-=used
          loadedAdapter.toVector.flatMap { (parameter,items) =>
            items.map(item => parameter->item)
          }
        }.groupMap(_._1)(_._2).view.mapValues(_.toVector).toMap
        val prediction=NativeGenerationMetadata.prediction(g.get("extra"),config.prediction)
        val sdxlSize=NativeGenerationMetadata.sdxlSize(g.get("extra"),pipeline.sdxlConditioning.isDefined)
        g.get("checkpoint") match
          case None | Some(Null) => ()
          case Some(Str(name)) =>
            require(name==root.getFileName.toString || Path.of(name).toAbsolutePath.normalize()==root,"requested checkpoint is not loaded")
          case _ => throw IllegalArgumentException("checkpoint must be a string or null")
        g.get("vae") match
          case None | Some(Null) => ()
          case Some(Str(name)) =>
            require(modelSummary.get("vae_name").contains(Str(name)) ||
              (name=="Automatic" && modelSummary.get("vae_name").contains(Null)),"requested VAE is not loaded")
          case _ => throw IllegalArgumentException("vae must be a string or null")
        val seedNumber=number(g,"seed",42)
        require(seedNumber.isValidLong,"seed must be a signed 64-bit integer")
        val seed=if seedNumber.toLong == -1 then new java.security.SecureRandom().nextLong() & Long.MaxValue else seedNumber.toLong
        val batchSize=integer(g,"batch_size",1)
        val nIter=integer(g,"n_iter",1)
        val iterations=batchSize*nIter
        val subseed=number(g,"subseed",-1)
        require(subseed.isValidLong,"subseed must be a signed 64-bit integer")
        require(batchSize>=1 && batchSize<=16,"batch_size must be between 1 and 16")
        require(nIter>=1 && nIter<=100,"n_iter must be between 1 and 100")
        require(iterations<=1600,"batch_size*n_iter must be at most 1600")
        require(BigInt(seed)+iterations-1<=Long.MaxValue,"iteration seeds exceed signed 64-bit range")
        require(subseed<0 || subseed+iterations-1<=BigDecimal(Long.MaxValue),"iteration subseeds exceed signed 64-bit range")
        val steps=integer(g,"steps",20)
        require(steps>0 && steps<=1000,"steps must be between 1 and 1000")
        val imageRequest=DiffusionImageRequest(str(g,"prompt",""),str(g,"negative_prompt",""),
          integer(g,"width",512),integer(g,"height",512),seed,number(g,"cfg_scale",7).toFloat,clipSkip=integer(g,"clip_skip",1),sdxlSize=sdxlSize)
        var preparedInpaint: Option[NativeInpaint.Prepared]=None
        val source=if !img2img then None else
          val fields=g.get("img2img") match
            case Some(Obj(fields)) => fields
            case _ => throw IllegalArgumentException("img2img must be an object")
          val inpaintKeys=Set("mask","mask_blur","inpainting_fill","inpaint_full_res","inpaint_full_res_padding","inpainting_mask_invert")
          keys(NativeGenerationDefaults.strip(fields--inpaintKeys,NativeGenerationDefaults.inpainting),Set("init_image","denoising_strength","resize_mode"))
          val strength=number(fields,"denoising_strength",BigDecimal("0.75")).toDouble
          require(strength.isFinite && strength>=0 && strength<=1,"denoising_strength must be between 0 and 1")
          val mode=integer(fields,"resize_mode",0)
          require(mode>=0 && mode<=2,"resize_mode must be 0, 1 or 2")
          val maskName=fields.get("mask") match
            case None | Some(Null) | Some(Str("")) => None
            case Some(Str(value)) => Some(value)
            case _ => throw IllegalArgumentException("mask must be a path or null")
          val maskBlur=integer(fields,"mask_blur",4)
          require(maskBlur>=0 && maskBlur<=64,"mask_blur must be between 0 and 64")
          val fill=integer(fields,"inpainting_fill",1)
          require(fill>=0 && fill<=3,"inpainting_fill must be between 0 and 3")
          val fullRes=bool(fields,"inpaint_full_res",false)
          val fullResPadding=integer(fields,"inpaint_full_res_padding",32)
          require(fullResPadding>=0 && fullResPadding<=256,"inpaint_full_res_padding must be between 0 and 256")
          val invert=bool(fields,"inpainting_mask_invert",false)
          if maskName.nonEmpty then
            if fill==0 then unsupported("blurred surrounding-color inpainting fill is not implemented")
          val name=str(fields,"init_image","")
          require(name.nonEmpty,"init_image is required")
          val width=imageRequest.width;val height=imageRequest.height
          require(width>0 && height>0 && width.toLong*height<=16777216L,"invalid img2img output dimensions")
          context.checkCancelled()
          val input=DiffusionImageInput.readRgb(Path.of(name))
          context.checkCancelled()
          val source=maskName match
            case Some(maskPath) =>
              if pipeline.unet.convIn.weight.data.shape.dims(1)!=pipeline.vae.config.latentChannels then
                unsupported("nine-channel inpainting U-Nets are not implemented")
              val mask=DiffusionImageInput.readRgb(Path.of(maskPath))
              val prepared=NativeInpaint.prepare(input,mask,width,height,mode,maskBlur,invert,context.cancelled,
                fullRes=fullRes,fullResPadding=fullResPadding,fill=fill,seed=seed)
              preparedInpaint=Some(prepared)
              Some(DiffusionImageSource(prepared.image,strength,mask=Some(prepared.mask),originalImage=Some(prepared.originalImage),
                finalOverlay=prepared.pasteBox.isEmpty))
            case None =>
              val resized=DiffusionImageResize(input,width,height,mode)
              val plane=width*height
              val values=new Array[Float](plane*3)
              for y <- 0 until height do
                context.checkCancelled()
                for x <- 0 until width do
                  val rgb=resized.getRGB(x,y)
                  for channel <- 0 until 3 do
                    values(channel*plane+y*width+x)=((rgb >>> (16-8*channel))&255)/255f*2f-1f
              Some(DiffusionImageSource(scalatorch.Tensor.fromFloats(values,
                scalatorch.Shape.from(Seq(1L,3L,height.toLong,width.toLong))),strength))
          context.checkCancelled()
          source
        val sampler=str(g,"sampler_name","DPM++ 2M")
        val requestedSchedule=str(g,"scheduler","Automatic")
        val schedule=if requestedSchedule=="Automatic" then (if sampler=="DPM++ 2M" then "Karras" else "Uniform") else requestedSchedule
        if !(sampler=="Euler" && Set("Uniform","Karras").contains(schedule)) &&
            !(sampler=="DPM++ 2M" && Set("Uniform","Karras").contains(schedule)) then unsupported(s"unsupported sampler/schedule: $sampler/$schedule")
        val outputName=str(p,"output_dir","")
        require(outputName.nonEmpty,"output_dir is required")
        val output=Path.of(outputName).toAbsolutePath.normalize()
        require(Files.isDirectory(output),"output_dir must be an existing directory")
        val recoveryName=str(p,"vae_recovery_dir","")
        val recovery=if recoveryName.isEmpty then output else Path.of(recoveryName).toAbsolutePath.normalize()
        val keywordPlan=NativeKeywordHook.plan(g.get("hooks"),imageRequest.prompt,
          modelSummary("hash").asInstanceOf[Str].value,root.getFileName.toString,iterations,seed,keywordState,() => context.checkCancelled())
        val started=System.nanoTime()
        val staging=Files.createTempDirectory(output,".native-staging-")
        val staged=scala.collection.mutable.ArrayBuffer.empty[Path]
        val published=scala.collection.mutable.ArrayBuffer.empty[Path]
        try
          val images=Vector.tabulate(iterations) { index =>
            context.checkCancelled()
            val current=imageRequest.copy(seed=seed+index,prompt=keywordPlan.fold(imageRequest.prompt)(_.prompts(index)))
            if Seq(current.prompt,current.negativePrompt).exists(s => "<(?:lora|lyco|hypernet|cdt|cdtc):[^>]+>".r.findFirstIn(s).nonEmpty) then
              unsupported("prompt extension directives require native extension execution, which is not implemented")
            val progress: DiffusionImageProgress => Unit = value =>
              context.event("progress",Map(
                "step"->Number(value.completedSteps),"total"->Number(value.totalSteps),"pass"->Str("base"),
                "eta_s"->Number(0),"image"->Number(index),"images"->Number(iterations),"batch_size"->Number(batchSize)))
              if previewEvery>0 && previewMethod!="none" && value.completedSteps%previewEvery==0 && value.completedSteps<value.totalSteps then
                context.checkCancelled()
                context.event("preview",Map("step"->Number(value.completedSteps),"jpeg_b64"->Str(NativePreview.jpegBase64(value.latents))))
            val image=try LinearAdapterScope.withAdapters(adapters) {
              if sampler=="DPM++ 2M" then
                val layout=if schedule=="Karras" then DpmSchedule.karras(config.noise,steps) else DpmSchedule.uniform(config.noise,steps)
                pipeline.generateDpm(current,layout,onProgress=progress,cancelled=context.cancelled,prediction=prediction,source=source)
              else
                val layout=if schedule=="Karras" then config.noise.karrasEuler(steps) else config.noise.euler(steps,config.spacing,config.stepsOffset)
                pipeline.generate(current,layout,onProgress=progress,cancelled=context.cancelled,prediction=prediction,source=source)
            }
            catch
              case failure: VaeDecodeFailure =>
                context.checkCancelled()
                val artifact=NativeVaeRecovery.save(failure.decoderInput,recovery)
                throw new NativeBackendError("vae_decode_failed",s"VAE decode failed; exact decoder input saved to $artifact")
            context.checkCancelled()
            val fullImage=preparedInpaint.filter(_.pasteBox.nonEmpty)
              .map(_.pasteBack(DiffusionImageIO.toBufferedImage(image),context.cancelled))
            val name=s"native-${UUID.randomUUID()}.png"
            val temporary=staging.resolve(name)
            fullImage match
              case Some(value) => DiffusionImageIO.writePngImage(value,temporary)
              case None => DiffusionImageIO.writePng(image,temporary)
            staged+=temporary
            val resultWidth=fullImage.fold(current.width)(_.getWidth)
            val resultHeight=fullImage.fold(current.height)(_.getHeight)
            Obj(Map("path"->Str(output.resolve(name).toString),"seed"->Number(BigDecimal(current.seed)),
              "subseed"->Number(if subseed<0 then subseed else subseed+index),
              "width"->Number(resultWidth),"height"->Number(resultHeight))++keywordPlan.fold(Map.empty[String,NativeJson]) { _ =>
                Map("effective_prompt"->Str(current.prompt),"model_keyword_source_prompt"->Str(current.prompt))
              })
          }
          context.beginPublication()
          staged.foreach { temporary =>
            val destination=output.resolve(temporary.getFileName)
            Files.move(temporary,destination)
            published+=destination
          }
          Files.delete(staging)
          keywordPlan.foreach(plan => keywordState=plan.nextState)
          Map("images"->Arr(images),"timing_s"->Number(BigDecimal((System.nanoTime()-started)/1e9)),
            "model_name"->Str(root.getFileName.toString),"scheduler"->Str(schedule))
        catch
          case error: Throwable =>
            (staged.toVector++published.toVector :+ staging).foreach { path =>
              try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
            }
            throw error
      case _ => unsupported(s"unsupported request: ${request.kind}")

object NativeWorkerMain:
  def main(args: Array[String]): Unit=
    require(args.length==2,"usage: NativeWorkerMain <diffusers-root> <explicit-euler-config.json>")
    val backend=new NativeDiffusionBackend(Path.of(args(0)),Path.of(args(1)),VaeExecution.fromEnvironment(sys.env),
      UNetExecution.fromEnvironment(sys.env),sys.env.get("SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS").fold(16777216L)(_.toLong))
    val worker=new NativeWorker(System.out,backend)
    worker.serve(new java.io.InputStreamReader(System.in,java.nio.charset.StandardCharsets.UTF_8))
