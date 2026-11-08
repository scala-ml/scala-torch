package scalatorch.models.worker

import scalatorch.models.*
import scalatorch.nn.LinearAdapterScope
import java.nio.file.{Files,Path}
import java.util.UUID

final class NativeBackendError(val code: String,message: String,cause: Throwable=null) extends RuntimeException(message,cause)

/** Host-loaded Float32 models with explicit U-Net/VAE execution placement. */
final class NativeDiffusionBackend(modelRoot: Path,schedulerConfig: Path,
    vaeExecution: VaeExecution=VaeExecution.Cpu,unetExecution: UNetExecution=UNetExecution.Cpu,
    maxAttentionScoreElements: Long=16777216L,maxVaeAttentionScoreElements: Long= -1L,
    tinyPreviewEnvironment: Map[String,String]=sys.env) extends NativeWorkerBackend:
  require(maxAttentionScoreElements>=0 && maxAttentionScoreElements<=(1L<<36),"invalid attention work limit")
  private val vaeAttentionLimit=if maxVaeAttentionScoreElements<0 then math.min(maxAttentionScoreElements,Int.MaxValue.toLong) else maxVaeAttentionScoreElements
  require(vaeAttentionLimit<=Int.MaxValue,"invalid VAE attention work limit")
  import NativeJson.*
  private val root=modelRoot.toRealPath()
  private val tinyPreview=NativeTinyPreview.fromEnvironment(tinyPreviewEnvironment,vaeExecution)
  private def singleFileKind: String = if SingleFileSd1Model.isPyTorch(root) then "ckpt" else "safetensors"
  private val config=if schedulerConfig.toString=="-" && Files.isRegularFile(root) then
    EulerConfig.fromJson("""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","prediction_type":"epsilon"}""")
  else EulerConfig.read(schedulerConfig)
  private var loaded: Option[EulerImagePipeline]=None
  private final case class ControlNetFileStamp(path: Path,size: Long,modified: java.nio.file.attribute.FileTime,key: Option[String])
  private var loadedControlNet: Option[(Path,Vector[ControlNetFileStamp],ConditionalControlNet2d)]=None
  private var loadedEmbeddingTokens=Set.empty[String]
  private var loadedEmbeddingInfo=Vector.empty[NativeEmbeddingInspector.LoadedEmbedding]
  private var linearTargets=Map.empty[String,scalatorch.nn.Parameter]
  private var keywordState=ModelKeyword.State()
  @volatile private var pendingSystemSubmission: Option[(Long,Map[String,NativeJson])]=None
  @volatile private var vramLastProfile: Option[NativeJson]=None
  @volatile private var modelSummary=Map.empty[String,NativeJson]
  @volatile private var vaeTilingEnabled=false
  private var singleFileFamily="sd1"
  override def device: String=unetExecution.device
  // Text encoders follow the U-Net onto CUDA unless SCALATORCH_CLIP_DEVICE says otherwise.
  if !sys.env.contains("SCALATORCH_CLIP_DEVICE") then unetExecution.device match
    case value if value.startsWith("cuda") => ClipCuda.enable(value.drop(5).toIntOption.getOrElse(0))
    case _ => ()
  override def features: Vector[String]=Vector("img2img_hires_fix","loopback_scaler","tokenizer","latent_upscale","inpaint_difference","gpu_temperature_guard","rescale_cfg","cfg_schedule","resharpen","latent_mirroring","color_grading","anti_burn","vectorscope_cc","detail_daemon","vae_tools","model_keyword","freeu","dynamic_thresholding","tiling","cads","hypertile","metadata_utils","model_sampling","apg","mahiro","pag","weight_gradient","negpip","kohya_hrfix","token_downsampling","conditioning_highres_fix","multidiffusion","latent_modifier","soft_inpainting","neveroom","semantic_guidance","sonar","hires_fix_tweaks","pnginfo_injection","prevent_interrupt","controlnet","controlnet_preprocess_extras","system_info","save_intermediate","steps_animation","img2imgalt","cd_tuner","llul","agent_attention","incantations","characteristic_guidance","clip_io","prompt_fusion","neutral_prompt","composable_lora","embedding_merge","aesthetic_gradients","sag","reference","revision","ip_adapter","controllllite","fooocus_inpaint","compile","taesd","spandrel","vram_estimator","adetailer","batch_face_swap","face_restore","nsfw_checker","aesthetic_score","animatediff","cn_animation","controlnet_fastload_data","photomaker","caption","img2vid","novel_view","custom_code","ldsr","marigold","controlnet_aux","temporal_lab","clip_interrogator","chara_searcher","batch_interrogate_wd_v3_partial")
  override def usesCuda: Boolean=vaeExecution.device!="cpu" || unetExecution.device!="cpu"
  override def systemStatus: Map[String,NativeJson]=NativeSystemStatus.snapshot(modelSummary,vaeExecution.device,unetExecution.device,
    vaeExecution.offloadWeights,unetExecution.offloadWeights,maxAttentionScoreElements,vaeTilingEnabled,vaeAttentionLimit)
  private def sdxlProfile(): Boolean =
    if Files.isRegularFile(root) then return false
    val addition=UNetConfig.read(root.resolve("unet/config.json")).additionTimeWidth.nonEmpty
    val index=root.resolve("model_index.json")
    if Files.exists(index) then
      val name=TokenizerJson.read(index).obj("_class_name").str
      val xl=Set("StableDiffusionXLPipeline","StableDiffusionXLImg2ImgPipeline","StableDiffusionXLInpaintPipeline").contains(name)
      require(xl || Set("StableDiffusionPipeline","StableDiffusionImg2ImgPipeline","StableDiffusionInpaintPipeline").contains(name),"unsupported diffusion model family")
      require(xl==addition,"model index and U-Net conditioning disagree")
    require(addition || (!Files.exists(root.resolve("text_encoder_2")) && !Files.exists(root.resolve("tokenizer_2"))),
      "second text encoder requires SDXL text-time conditioning")
    addition
  private def checkpointHash(context: NativeWorkContext): String=
    if Files.isRegularFile(root) then
      val digest=java.security.MessageDigest.getInstance("SHA-256")
      val input=Files.newInputStream(root)
      try
        val bytes=new Array[Byte](65536)
        var n=input.read(bytes)
        while n != -1 do
          context.checkCancelled()
          digest.update(bytes,0,n)
          n=input.read(bytes)
      finally input.close()
      return java.util.HexFormat.of().formatHex(digest.digest())
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
  val supported=Set("load_model","set_vae","unload","txt2img","img2img","upscale","image_tool","vae_tools","preprocess","inpaint_difference_preview","tokenize","embedding_inspector","embedding_editor","list_models","list_samplers","list_files","model_keyword","model_keyword_plan","metadata_utils","vram_estimator","merge_models","convert_model","memory","system_info","system_benchmark","system_submission_preview","system_submit","embedding_merge","aesthetic_embedding","controlnet_fastload_data","temporal_lab","cn_animation","animatediff_encode","restore_faces","caption","clip_interrogator","chara_searcher","img2vid","novel_view","run_code","batch_interrogate_prepare","batch_interrogate_execute")
  private def vaeChoice(params: Map[String,NativeJson],key: String): Option[Path]=params.get(key) match
    case None | Some(Null) | Some(Str("")) => None
    case Some(Str(value)) => Some(Path.of(value).toRealPath())
    case _ => throw IllegalArgumentException(s"$key must be a path or null")
  private def readVae(path: Option[Path],context: NativeWorkContext,singleFileVaeOverride: Option[VaeConfig]=None): (VaeModel,Map[String,NativeJson])=
    if Files.isRegularFile(root) && path.isEmpty then
      val vae=if SingleFileSd1Model.isPyTorch(root) then
        SingleFileSd1Model.loadVae(root,vaeAttentionLimit,()=>context.checkCancelled())
      else scala.util.Using.resource(scalatorch.io.SafeTensorSet.openFile(root)) { archive =>
          if SingleFileSdxlConfig.isSdxl(archive.tensors) then
            PretrainedVae.loadSdxlSingleFile(archive,maxScoreElements=vaeAttentionLimit)
          else if SingleFileSd2Config.isSd2(archive.tensors) then
            PretrainedVae.loadSd2SingleFile(archive,maxScoreElements=vaeAttentionLimit)
          else PretrainedVae.loadSingleFile(archive,maxScoreElements=vaeAttentionLimit)
        }
      context.checkCancelled()
      return (vae,Map("vae_name"->Null,"vae_hash"->Null))
    val selected=path.getOrElse(root.resolve("vae"))
    val directory=Files.isDirectory(selected)
    val folder=if directory then selected else selected.getParent
    val filename=if directory then "diffusion_pytorch_model.safetensors" else selected.getFileName.toString
    val vae=PretrainedVae.load(folder,maxFeatureElements=Int.MaxValue.toLong,maxScoreElements=vaeAttentionLimit,
      configOverride=if directory then None else Some(if Files.isRegularFile(root) then singleFileVaeOverride.getOrElse(if singleFileFamily=="sdxl" then SingleFileSdxlConfig.vaeConfig else SingleFileSd1Config.vaeConfig)
        else VaeConfig.read(root.resolve("vae/config.json"))),filename=filename)
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
  private def controlNetStamp(directory: Path,context: NativeWorkContext): Vector[ControlNetFileStamp] =
    if Files.isRegularFile(directory) then
      context.checkCancelled()
      val attrs=Files.readAttributes(directory,classOf[java.nio.file.attribute.BasicFileAttributes])
      return Vector(ControlNetFileStamp(directory,attrs.size(),attrs.lastModifiedTime(),Option(attrs.fileKey()).map(_.toString)))
    val stream=Files.list(directory)
    try
      import scala.jdk.CollectionConverters.*
      stream.iterator().asScala.filter { path =>
        val name=path.getFileName.toString
        name=="config.json" || name.endsWith(".safetensors") || name.endsWith(".safetensors.index.json")
      }.toVector.sortBy(_.getFileName.toString).map { path =>
        context.checkCancelled()
        val attrs=Files.readAttributes(path,classOf[java.nio.file.attribute.BasicFileAttributes])
        require(attrs.isRegularFile,s"ControlNet component is not a regular file: $path")
        ControlNetFileStamp(path,attrs.size(),attrs.lastModifiedTime(),Option(attrs.fileKey()).map(_.toString))
      }
    finally stream.close()
  private def controlNetModel(path: Path,context: NativeWorkContext): ConditionalControlNet2d =
    val real=path.toRealPath()
    val stamp=controlNetStamp(real,context)
    loadedControlNet match
      case Some((cachedPath,cachedStamp,model)) if cachedPath==real && cachedStamp==stamp => model
      case _ =>
        context.checkCancelled()
        val controlConfig=if Files.isRegularFile(real) then SingleFileControlNet.config
          else ControlNetConfig.read(real.resolve("config.json"))
        controlConfig.requireCompatible(if Files.isRegularFile(root) && SingleFileSd1Model.isPyTorch(root) then
          (if singleFileFamily=="sdxl" then SingleFileSdxlConfig.unetConfig else if singleFileFamily=="sd2" then SingleFileSd2Config.unetConfig else SingleFileSd1Config.unetConfig)
            .copy(inChannels=loaded.getOrElse(
              throw IllegalStateException("base model is not loaded")).unet.convIn.weight.data.shape.dims(1).toInt)
          else if Files.isRegularFile(root) then
            scala.util.Using.resource(scalatorch.io.SafeTensorSet.openFile(root)) { archive =>
              if SingleFileSdxlConfig.isSdxl(archive.tensors) then SingleFileSdxlConfig.unetConfigFor(archive)
              else if SingleFileSd2Config.isSd2(archive.tensors) then SingleFileSd2Config.unetConfigFor(archive)
              else SingleFileSd1Config.unetConfigFor(archive)
            }
          else UNetConfig.read(root.resolve("unet/config.json")))
        val model=PretrainedControlNet.loadWithProgress(real,500000000L,maxAttentionScoreElements,context.cancelled,() =>
          context.event("log",Map("level"->Str("info"),"message"->Str("Reading ControlNet tensor weights"))))
        context.checkCancelled()
        require(controlNetStamp(real,context)==stamp,"ControlNet files changed during loading")
        loadedControlNet=Some((real,stamp,model))
        model
  private def withEmbeddings(base: EulerImagePipeline,paths: Vector[String],prompt: String,negativePrompt: String,
      alreadyLoaded: Set[String],context: NativeWorkContext): (EulerImagePipeline,Set[String],Vector[String],Vector[NativeEmbeddingInspector.LoadedEmbedding]) =
    require(paths.size<=512,"too many textual-inversion paths")
    val active=paths.flatMap { name =>
      val path=Path.of(name).toAbsolutePath.normalize()
      val token=Option(path.getFileName).map(_.toString).getOrElse("").replaceFirst("\\.[^.]*$","")
      if !Files.isRegularFile(path) then None
      else if token.isEmpty || alreadyLoaded(token) || !Seq(prompt,negativePrompt).exists(text =>
          (s"(?<![\\w-])${java.util.regex.Pattern.quote(token)}(?![\\w-])").r.findFirstIn(text).nonEmpty) then None
      else Some(path->token)
    }.distinctBy(_._2)
    var tokenizer=base.tokenizer
    var encoder=base.textEncoder
    var conditioning=base.sdxlConditioning
    var attempted=alreadyLoaded
    val warnings=scala.collection.mutable.ArrayBuffer.empty[String]
    val added=scala.collection.mutable.ArrayBuffer.empty[NativeEmbeddingInspector.LoadedEmbedding]
    active.foreach { (path,token) =>
      context.checkCancelled()
      try
        val sdxl=base.sdxlConditioning.nonEmpty
        val clipGWidth=conditioning.map(_.encoder2.finalNorm.width).getOrElse(0)
        val lowerName=path.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
        val vectors=if lowerName.endsWith(".safetensors") then
          ClipTextualInversion.readSafetensors(path,sdxl,encoder.finalNorm.width,clipGWidth)
        else if lowerName.endsWith(".pt") || lowerName.endsWith(".pth") || lowerName.endsWith(".bin") then
          ClipTextualInversion.readPyTorch(path,sdxl,encoder.finalNorm.width,clipGWidth)
        else throw IllegalArgumentException("native textual inversion supports safetensors and restricted PyTorch tensor archives only")
        val nextTokenizer=tokenizer.withAddedTokens(Vector(token->vectors.clipL.size))
        val nextEncoder=encoder.withAddedTokenEmbeddings(vectors.clipL)
        val nextConditioning=conditioning.map { text =>
          val rows=vectors.clipG.getOrElse(throw IllegalArgumentException("missing CLIP-G vectors"))
          val nextTokenizer2=text.tokenizer2.withAddedTokens(Vector(token->rows.size))
          val nextEncoder2=text.encoder2.withAddedTokenEmbeddings(rows)
          new SdxlTextConditioning(nextTokenizer,nextEncoder,nextTokenizer2,nextEncoder2,text.maxElements)
        }
        tokenizer=nextTokenizer
        encoder=nextEncoder
        conditioning=nextConditioning
        added+=NativeEmbeddingInspector.LoadedEmbedding(token,path,vectors.clipL.size,encoder.finalNorm.width,
          conditioning.map(_.encoder2.finalNorm.width))
        attempted+=token
      catch
        case scala.util.control.NonFatal(error) =>
          warnings+=s"Textual inversion '$token' skipped: ${Option(error.getMessage).getOrElse("invalid embedding")}"
          attempted+=token
    }
    val next=if tokenizer eq base.tokenizer then base else
      new EulerImagePipeline(tokenizer,encoder,base.unet,base.vae,vaeExecution=base.vaeExecution,
        unetExecution=base.unetExecution,sdxlConditioning=conditioning,vaeTiling=base.vaeTiling)
    (next,attempted,warnings.toVector,added.toVector)
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
    context.checkCancelled()
    def refreshCudaTelemetry(): Unit =
      if usesCuda then
        val index=Seq(unetExecution.device,vaeExecution.device).collectFirst {
          case value if value.startsWith("cuda:") => value.drop(5).toInt
        }.getOrElse(0)
        NativeSystemStatus.captureCudaMemory(index)
    if request.kind=="merge_models" then NativeModelMerge.execute(request.params,context)
    else if request.kind=="convert_model" then NativeModelConvert.execute(request.params,context)
    else if request.kind=="metadata_utils" then NativeMetadataUtils.execute(request.params,context)
    else if request.kind=="system_info" then
      refreshCudaTelemetry()
      val samplerInfo=executeNativeRequest(NativeRequest(request.id,"list_samplers",Map.empty,request.owner),context)
      NativeSystemInfo.execute(request.params,systemStatus,features,samplerInfo,context)
    else if request.kind=="system_submission_preview" then
      val preview=NativeSystemInfo.submissionPreview(request.params)
      pendingSystemSubmission=Some((System.nanoTime(),preview))
      preview
    else if request.kind=="system_submit" then
      val current=pendingSystemSubmission
      pendingSystemSubmission=None
      NativeSystemInfo.submit(request.params,current)
    else if request.kind=="system_benchmark" then
      if loaded.isEmpty then throw new NativeBackendError("no_model","load a checkpoint before benchmarking")
      refreshCudaTelemetry()
      val samplerInfo=executeNativeRequest(NativeRequest(request.id,"list_samplers",Map.empty,request.owner),context)
      val info=NativeSystemInfo.execute(Map.empty,systemStatus,features,samplerInfo,context)
      NativeSystemBenchmark.run(request.params,info,context) { (batch,width,height,steps,output,owner) =>
        val generation=NativeSystemBenchmark.generationParams(batch,width,height,steps)
        val params=Map[String,NativeJson]("params"->Obj(generation),"output_dir"->Str(output.toString),
          "vae_recovery_dir"->Str(output.toString),"preview_every"->Number(0),"preview_method"->Str("none"))
        val nested=new NativeWorkContext(owner.cancelled,()=>true,(_,_)=>())
        val result=execute(NativeRequest(UUID.randomUUID().toString,"txt2img",params,None),nested)
        result("images").asInstanceOf[Arr].values.size
      }
    else if request.kind=="memory" then
      require(request.params.isEmpty,"memory takes no parameters")
      refreshCudaTelemetry()
      val details=systemStatus("memory").asInstanceOf[Obj].fields
      Map("allocated_mb"->Number(0),"reserved_mb"->Number(0),"ram_used_mb"->details.getOrElse("process_rss_mb",Number(0)),
        "total_mb"->details.getOrElse("cuda_total_mb",Number(0)),
        "device_used_mb"->details.getOrElse("cuda_used_mb",Null))
    else if request.kind=="vram_estimator" then executeVramEstimator(request.params,context)
    else if request.kind=="embedding_merge" then NativeBatch18Services.executeEmbeddingMerge(request.params,context,loaded)
    else if request.kind=="aesthetic_embedding" then NativeBatch18Services.executeAestheticEmbedding(request.params,context)
    else if request.kind=="controlnet_fastload_data" then NativeBatch18Services.executeControlNetFastloadData(request.params,context)
    else if request.kind=="temporal_lab" then NativeBatch18Services.executeTemporalLab(request.params,context)
    else if request.kind=="cn_animation" then NativeBatch18Services.executeCnAnimation(request.params,context)
    else if request.kind=="animatediff_encode" then NativeBatch18Services.executeAnimateDiffEncode(request.params,context)
    else if request.kind=="restore_faces" then NativeBatch18Services.executeRestoreFaces(request.params,context)
    else if request.kind=="caption" then NativeBatch18Services.executeCaption(request.params,context)
    else if request.kind=="clip_interrogator" then NativeBatch18Services.executeClipInterrogator(request.params,context)
    else if request.kind=="chara_searcher" then NativeBatch18Services.executeCharaSearcher(request.params,context)
    else if request.kind=="img2vid" then NativeBatch18Services.executeImg2Vid(request.params,context)
    else if request.kind=="novel_view" then NativeBatch18Services.executeNovelView(request.params,context)
    else if request.kind=="run_code" then NativeBatch18Services.executeRunCode(request.params,context)
    else if request.kind=="batch_interrogate_prepare" then
      NativeBatch18Services.executeBatchInterrogatePrepare(request.params,request.owner,modelSummary,loaded.nonEmpty,context)
    else if request.kind=="batch_interrogate_execute" then
      NativeBatch18Services.executeBatchInterrogateExecute(request.params,request.owner,modelSummary,loaded.nonEmpty,context) {
        (img2imgParams,ownerCtx) =>
          val nested=new NativeWorkContext(ownerCtx.cancelled,()=>true,(_,_)=>())
          execute(NativeRequest(UUID.randomUUID().toString,"img2img",img2imgParams,request.owner),nested)
      }
    else if request.kind=="vae_tools" then
      val pipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a checkpoint before using VAE Tools"))
      val approximation=str(request.params,"approximation","Full")
      val full=new NativeFullVaeTools(pipeline.vae,vaeExecution,bool(request.params,"tiled",false))
      val tool=if approximation=="Full" then full
      else if approximation=="Approx cheap" then new NativeCheapVaeTools(full,pipeline.sdxlConditioning.nonEmpty)
      else if approximation=="Approx NN" then
        val explicit=str(request.params,"approx_nn_path","")
        val cache=str(request.params,"cache_dir","")
        val name=if pipeline.sdxlConditioning.nonEmpty then "vaeapprox-sdxl.pt" else "model.pt"
        val file=if explicit.nonEmpty then Path.of(explicit) else Path.of(cache).resolve("vae-approx").resolve(name)
        if explicit.isEmpty && !Files.exists(file) && bool(request.params,"allow_download",false) then
          NativeApproxDownload.official(file,name,()=>context.checkCancelled())
        if !Files.isRegularFile(file) then throw new NativeBackendError("unsupported","Approx NN requires a local A1111 weights file")
        new NativeApproxNnVaeTools(full,NativeApproxNn.load(file))
      else if approximation!="TAESD" then throw new NativeBackendError("unsupported",s"VAE approximation '$approximation' is not implemented")
      else
        val cache=str(request.params,"cache_dir","")
        val cachePath=if cache.nonEmpty then Path.of(cache) else Path.of(str(request.params,"output_dir",".")).resolve(".vae-tools-models")
        val directory=NativeTinyWeights.resolve(tinyPreviewEnvironment,cachePath,bool(request.params,"allow_download",false),
          pipeline.sdxlConditioning.nonEmpty,()=>context.checkCancelled())
        context.checkCancelled()
        val encoder=TinyAutoencoderEncoder.load(directory)
        context.checkCancelled()
        val decoder=TinyAutoencoderDecoder.load(directory)
        new NativeTinyVaeTools(encoder,decoder,vaeExecution)
      val provenance=Map[String,NativeJson]("model"->Obj(modelSummary),
        "vae"->Obj(modelSummary.filter((key,_)=>Set("vae_name","vae_hash")(key))),
        "family"->Str(if pipeline.sdxlConditioning.nonEmpty then "sdxl" else "sd15"),
        "scaling_factor"->Number(BigDecimal(pipeline.vae.config.scalingFactor)),
        "shift_factor"->Null,"latents_mean"->Null,"latents_std"->Null)
      NativeVaeTools.execute(request.params,tool,context,provenance,request.imagePassword)
    else if request.kind=="image_tool" then NativeCasImageTool.execute(request.params,context)
    else if request.kind=="preprocess" then NativePreprocess.execute(request.params,context)
    else if request.kind=="inpaint_difference_preview" then NativeInpaintDifferencePreview.execute(request.params,context)
    else if request.kind=="embedding_editor" then executeEmbeddingEditor(request.params,context)
    else if request.kind=="embedding_inspector" then
      val pipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a checkpoint before inspecting embeddings"))
      if request.params.get("operation").contains(Str("list")) then
        NativeEmbeddingInspector.list(request.params,loadedEmbeddingInfo,context)
      else if request.params.get("operation").contains(Str("save")) then
        if bool(request.params,"batch_presets",false) then saveMixedBatch(request.params,pipeline,context)
        else saveMixedEmbedding(request.params,pipeline,context)
      else if Set(Str("inspect"),Str("export")).contains(request.params.getOrElse("operation",Null)) then
        val streams=Vector(("clip_l",pipeline.tokenizer,pipeline.textEncoder.tokenEmbedding.weight.data))++
          pipeline.sdxlConditioning.toVector.map(c=>("clip_g",c.tokenizer2,c.encoder2.tokenEmbedding.weight.data))
        if request.params("operation")==Str("export") then NativeEmbeddingInspector.exportVectors(request.params,streams,context)
        else NativeEmbeddingInspector.inspect(request.params,streams,context)
      else
        val tokenizers=Vector("clip_l"->pipeline.tokenizer)++pipeline.sdxlConditioning.toVector.map(c=>"clip_g"->c.tokenizer2)
        NativeEmbeddingInspector.execute(request.params,tokenizers,context)
    else if request.kind=="tokenize" then
      val pipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a checkpoint before inspecting its tokenizer"))
      val first=if Files.isRegularFile(root) then sys.env.get("SDFX_SCALATORCH_TOKENIZER").map(Path.of(_)).getOrElse(root.getParent.resolve("tokenizer"))
        else root.resolve("tokenizer")
      val tokenizers=Vector(("tokenizer",first.toString,pipeline.tokenizer))++pipeline.sdxlConditioning.toVector.map(c =>
        ("tokenizer_2",(if Files.isRegularFile(root) then { val cand=sys.env.get("SDFX_SCALATORCH_TOKENIZER_2").map(Path.of(_)).getOrElse(first.resolveSibling("tokenizer_2")); if Files.isDirectory(cand) then cand else first } else root.resolve("tokenizer_2")).toString,c.tokenizer2))
      NativeTokenizerInspection.execute(request.params,tokenizers,context)
    else if Set("txt2img","img2img")(request.kind) && request.params.contains("embedding_editor_preview") then
      executeEditorPreview(request,context)
    else executeNativeRequest(request,context)

  private def executeEditorPreview(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson] =
    val draft=request.params("embedding_editor_preview") match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("invalid embedding editor preview")
    val name=draft.get("name") match
      case Some(Str(value)) if value.nonEmpty => value
      case _ => throw IllegalArgumentException("invalid embedding editor preview")
    executeEmbeddingEditor(Map("operation"->Str("guidance"),"name"->Str(name),"draft"->Obj(draft)),context)
    val original=loaded.getOrElse(throw new NativeBackendError("no_model","load a checkpoint before previewing"))
    val originalTokens=loadedEmbeddingTokens;val originalInfo=loadedEmbeddingInfo
    val vectors=draft("vectors").asInstanceOf[Obj].fields
    def captured(tokenizer: ClipFastTokenizer,encoder: ClipTextModel): Vector[Array[Float]] =
      val ids=tokenizer.encode(name,addSpecialTokens=false).toVector
      val tensor=encoder.tokenEmbedding.weight.data
      val width=tensor.shape.dims(1).toInt;val data=tensor.toFloatArray
      ids.map(id=>java.util.Arrays.copyOfRange(data,id.toInt*width,(id.toInt+1)*width))
    val savedLeft=captured(original.tokenizer,original.textEncoder)
    val savedRight=original.sdxlConditioning.map(value=>captured(value.tokenizer2,value.encoder2))
    def replace(key: String,tokenizer: ClipFastTokenizer,encoder: ClipTextModel): ClipTextModel =
      val ids=tokenizer.encode(name,addSpecialTokens=false).toVector
      val rows=vectors(key).asInstanceOf[Arr].values.map { row =>
        row.asInstanceOf[Arr].values.map { item =>
          val value=item.asInstanceOf[Number].value.toFloat
          require(value.isFinite,"embedding preview value exceeds Float32 range")
          value
        }.toArray
      }
      encoder.withReplacedTokenEmbeddings(ids,rows)
    val encoder=replace("clip_l",original.tokenizer,original.textEncoder)
    val conditioning=original.sdxlConditioning.map { value =>
      new SdxlTextConditioning(original.tokenizer,encoder,value.tokenizer2,
        replace("clip_g",value.tokenizer2,value.encoder2),value.maxElements)
    }
    val temporary=new EulerImagePipeline(original.tokenizer,encoder,original.unet,original.vae,
      vaeExecution=original.vaeExecution,unetExecution=original.unetExecution,
      sdxlConditioning=conditioning,vaeTiling=original.vaeTiling)
    loaded=Some(temporary)
    try executeNativeRequest(request.copy(params=request.params-"embedding_editor_preview"),context)
    finally
      try
        val current=loaded.getOrElse(throw IllegalStateException("preview lost loaded model"))
        if current eq temporary then loaded=Some(original)
        else
          val left=current.textEncoder.withReplacedTokenEmbeddings(current.tokenizer.encode(name,false).toVector,savedLeft)
          val restored=current.sdxlConditioning.map { value =>
            val right=value.encoder2.withReplacedTokenEmbeddings(value.tokenizer2.encode(name,false).toVector,
              savedRight.getOrElse(throw IllegalStateException("preview lost CLIP-G rows")))
            new SdxlTextConditioning(current.tokenizer,left,value.tokenizer2,right,value.maxElements)
          }
          loaded=Some(new EulerImagePipeline(current.tokenizer,left,current.unet,current.vae,vaeExecution=current.vaeExecution,
            unetExecution=current.unetExecution,sdxlConditioning=restored,vaeTiling=current.vaeTiling))
      catch
        case error: Throwable =>
          loaded=Some(original);loadedEmbeddingTokens=originalTokens;loadedEmbeddingInfo=originalInfo
          throw error

  private def executeEmbeddingEditor(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    val operation=str(p,"operation","")
    if !Set("open","guidance","save")(operation) then unsupported(s"native embedding editor does not support $operation yet")
    if operation=="save" then require(bool(p,"overwrite",false),"explicit overwrite consent is required")
    val name=str(p,"name","").trim
    require(name.nonEmpty,"choose an embedding")
    val token=modelSummary.get("model_token") match
      case Some(Str(value)) if value.nonEmpty => value
      case _ => throw new NativeBackendError("no_model","load a checkpoint before using the embedding editor")
    val original=loaded.getOrElse(throw new NativeBackendError("no_model","load a checkpoint before using the embedding editor"))
    val matches=loadedEmbeddingInfo.filter(_.name.equalsIgnoreCase(name))
    require(matches.size<=1,"embedding name is ambiguous")
    val requested=p.get("path") match
      case None | Some(Null) | Some(Str("")) => None
      case Some(Str(value)) => Some(Path.of(value).toRealPath())
      case _ => throw IllegalArgumentException("invalid embedding selection")
    val (pipeline,info,addedTokens)=matches.headOption match
      case Some(existing) =>
        requested.foreach(path=>require(existing.path.toRealPath()==path,"another embedding with this name is already loaded"))
        (original,existing,None)
      case None =>
        require(operation=="open" && requested.nonEmpty,"embedding is not loaded")
        val file=requested.get
        require(Files.isRegularFile(file) && file.getFileName.toString.replaceFirst("\\.[^.]*$","").equalsIgnoreCase(name),
          "selected embedding path and name do not match")
        val (next,tokens,warnings,added)=withEmbeddings(original,Vector(file.toString),name,"",
          loadedEmbeddingTokens.filterNot(_.equalsIgnoreCase(name)),context)
        require(warnings.isEmpty && added.size==1,"embedding could not be activated")
        (next,added.head,Some(tokens))
    val streams=Vector(("clip_l",pipeline.tokenizer,pipeline.textEncoder.tokenEmbedding.weight.data))++
      pipeline.sdxlConditioning.toVector.map(c=>("clip_g",c.tokenizer2,c.encoder2.tokenEmbedding.weight.data))
    val result=NativeEmbeddingEditor.open(info.name,info.path,token,str(p,"guidance",""),streams,context)
    if operation=="guidance" || operation=="save" then
      val draft=p.get("draft") match
        case Some(Obj(fields)) => fields
        case _ => throw IllegalArgumentException("invalid editor draft")
      require(draft.get("name").contains(Str(info.name)),"draft embedding changed; reopen the embedding")
      for key <- Vector("model_token","path","file_identity") do
        require(draft.get(key).contains(result(key)),"embedding changed; reopen the embedding")
      val expected=result("vectors").asInstanceOf[Obj].fields
      val vectors=draft.get("vectors") match
        case Some(Obj(fields)) => fields
        case _ => throw IllegalArgumentException("invalid editor vectors")
      require(vectors.keySet==expected.keySet,"invalid editor encoder set")
      for (key,value) <- vectors do
        val rows=value match
          case Arr(items) => items
          case _ => throw IllegalArgumentException("invalid editor vectors")
        val originalRows=expected(key).asInstanceOf[Arr].values
        require(rows.size==originalRows.size,"embedding vector dimensions changed")
        rows.zip(originalRows).foreach { (row,old) =>
          val values=row match
            case Arr(items) => items
            case _ => throw IllegalArgumentException("invalid editor vector")
          require(values.size==old.asInstanceOf[Arr].values.size && values.forall {
            case Number(n) => n.toDouble.isFinite
            case _ => false
          },"invalid editor vector")
        }
      if operation=="guidance" then Map("ranges"->result("ranges"),"guidance"->result("guidance"))
      else
        def convert(fields: Map[String,NativeJson]): Map[String,Vector[Vector[Double]]] =
          fields.map { (key,value) => key->value.asInstanceOf[Arr].values.map { row =>
            row.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
          }}
        saveEmbeddingEdit(info,pipeline,convert(vectors),convert(expected),result,context)
    else
      addedTokens.foreach { tokens =>
        context.beginPublication()
        NativeBatch18Services.clearPreparedBatchInterrogates()
        loaded=Some(pipeline);loadedEmbeddingTokens=tokens;loadedEmbeddingInfo=loadedEmbeddingInfo :+ info
      }
      result

  private def saveEmbeddingEdit(info: NativeEmbeddingInspector.LoadedEmbedding,pipeline: EulerImagePipeline,
      edited: Map[String,Vector[Vector[Double]]],baseline: Map[String,Vector[Vector[Double]]],
      snapshot: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    val source=info.path.toAbsolutePath.normalize()
    val lower=source.toString.toLowerCase(java.util.Locale.ROOT)
    val safe=lower.endsWith(".safetensors")
    require(safe || lower.endsWith(".pt") || lower.endsWith(".bin"),"unsupported embedding file format")
    val fingerprint=snapshot("file_identity").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
    val suffix=UUID.randomUUID().toString
    val stage=source.resolveSibling(s".${source.getFileName}.editing-$suffix")
    val backup=source.resolveSibling(s"${source.getFileName}.before-edit-$suffix.backup")
    var stageCreated=false
    try
      if safe then ClipTextualInversion.stageSafetensorsEditDraft(source,stage,edited,baseline,fingerprint.last,context.cancelled)
      else ClipTextualInversion.stagePyTorchEditDraft(source,stage,edited,baseline,fingerprint.last,context.cancelled)
      stageCreated=true
      val sdxl=pipeline.sdxlConditioning.nonEmpty
      val rightWidth=pipeline.sdxlConditioning.map(_.encoder2.finalNorm.width).getOrElse(0)
      val saved=if safe then ClipTextualInversion.readSafetensors(stage,sdxl,pipeline.textEncoder.finalNorm.width,rightWidth)
        else ClipTextualInversion.readPyTorch(stage,sdxl,pipeline.textEncoder.finalNorm.width,rightWidth)
      val (baseTokenizer,leftIds)=pipeline.tokenizer.withoutAddedTokenGroup(info.name)
      val tokenizer=baseTokenizer.withAddedTokens(Vector(info.name->saved.clipL.size))
      val encoder=pipeline.textEncoder.withSelectedTokenEmbeddings(leftIds).withAddedTokenEmbeddings(saved.clipL)
      val conditioning=pipeline.sdxlConditioning.map { current =>
        val right=saved.clipG.getOrElse(throw IllegalArgumentException("missing CLIP-G"))
        val (baseTokenizer2,rightIds)=current.tokenizer2.withoutAddedTokenGroup(info.name)
        new SdxlTextConditioning(tokenizer,encoder,baseTokenizer2.withAddedTokens(Vector(info.name->right.size)),
          current.encoder2.withSelectedTokenEmbeddings(rightIds).withAddedTokenEmbeddings(right),current.maxElements)
      }
      val next=new EulerImagePipeline(tokenizer,encoder,pipeline.unet,pipeline.vae,vaeExecution=pipeline.vaeExecution,
        unetExecution=pipeline.unetExecution,sdxlConditioning=conditioning,vaeTiling=pipeline.vaeTiling)
      require(NativeEmbeddingEditor.fingerprint(source,context)==fingerprint,"embedding changed; reopen the embedding")
      context.checkCancelled()
      context.beginPublication()
      Files.move(source,backup)
      try
        require(NativeEmbeddingEditor.fingerprint(backup,context).last==fingerprint.last,"embedding changed before publication")
        Files.createLink(source,stage)
      catch
        case error: Throwable =>
          try Files.createLink(source,backup)
          catch case restore: Throwable => error.addSuppressed(restore)
          throw error
      loaded=Some(next)
      loadedEmbeddingInfo=loadedEmbeddingInfo.filterNot(_.name.equalsIgnoreCase(info.name)) :+
        NativeEmbeddingInspector.LoadedEmbedding(info.name,source,saved.clipL.size,encoder.finalNorm.width,
          conditioning.map(_.encoder2.finalNorm.width))
      Map("path"->Str(source.toString),"backup"->Str(backup.toString),"reloaded"->Bool(true))
    finally
      if stageCreated then Files.deleteIfExists(stage)

  private def saveMixedBatch(p: Map[String,NativeJson],pipeline: EulerImagePipeline,
      context: NativeWorkContext): Map[String,NativeJson] =
    val output=str(p,"output","")
    val extension=if output.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then ".safetensors"
      else if output.toLowerCase(java.util.Locale.ROOT).endsWith(".pt") then ".pt"
      else throw IllegalArgumentException("embedding output must be safetensors or .pt")
    val base=Path.of(output)
    val stem=base.getFileName.toString.dropRight(extension.length)
    require(stem.nonEmpty,"embedding name is empty")
    val presets=Vector("Boost"->"=v*8","Digitize"->"=math.ceil(v*8)/8",
      "Binary"->"=(1*(v>=0)-1*(v<0))/50","Randomize"->"=v*random.random()",
      "Sine"->"=v*math.sin(i/maxi*math.pi)","Comb"->"=v*((i%2)==0)",
      "Crop_high"->"=v*(i<maxi//2)","Crop_low"->"=v*(i>=maxi//2)")
    val jobs=Vector((base,p.getOrElse("eval",Str(""))))++presets.map { (name,expression) =>
      (base.resolveSibling(s"${stem}_$name$extension"),Str(expression))
    }
    val overwrite=bool(p,"overwrite",false)
    val catalog=p.get("embedding_paths") match
      case None => Vector.empty
      case Some(Arr(values)) if values.size<=100000 => values.map {
        case Str(value) => Path.of(value).toAbsolutePath.normalize()
        case _ => throw IllegalArgumentException("invalid embedding paths")
      }
      case _ => throw IllegalArgumentException("invalid embedding paths")
    jobs.foreach { (path,_) =>
      context.checkCancelled()
      val target=path.toAbsolutePath.normalize()
      require(target.getParent!=null && Files.isDirectory(target.getParent),"embedding output directory does not exist")
      val name=target.getFileName.toString.dropRight(extension.length)
      require(!catalog.exists(other=>other.getFileName.toString.replaceFirst("\\.[^.]*$","").equalsIgnoreCase(name) &&
        !other.toString.equalsIgnoreCase(target.toString)),"duplicate embedding name in catalog")
      if !overwrite && Files.exists(target) then throw new java.nio.file.FileAlreadyExistsException(target.toString)
    }
    val streams=Vector(("clip_l",pipeline.tokenizer,pipeline.textEncoder.tokenEmbedding.weight.data))++
      pipeline.sdxlConditioning.toVector.map(c=>("clip_g",c.tokenizer2,c.encoder2.tokenEmbedding.weight.data))
    val inputs=NativeEmbeddingInspector.prepareMixInputs(p,streams,context)
    val warnings=scala.collection.mutable.ArrayBuffer.empty[String]
    val ready=jobs.flatMap { (path,expression) =>
      val params=p.updated("batch_presets",Bool(false)).updated("eval",expression).updated("output",Str(path.toString))
      try Some((path,params,NativeEmbeddingInspector.applyPreparedMix(params,inputs)))
      catch
        case error: IllegalArgumentException if Option(error.getMessage).exists(_.contains("all mixed vectors are zero")) =>
          warnings+=s"${path.getFileName}: all mixed vectors are zero; skipped";None
    }
    require(ready.nonEmpty,"all mixed vectors are zero")
    // Verify every token name against the full set before any file is published.
    var leftTokenizer=pipeline.tokenizer
    var rightTokenizer=pipeline.sdxlConditioning.map(_.tokenizer2)
    ready.foreach { (path,_,values) =>
      val name=path.getFileName.toString.dropRight(extension.length)
      loadedEmbeddingInfo.find(_.name.equalsIgnoreCase(name)).foreach { previous =>
        leftTokenizer=leftTokenizer.withoutAddedTokenGroup(previous.name)._1
        rightTokenizer=rightTokenizer.map(_.withoutAddedTokenGroup(previous.name)._1)
      }
      leftTokenizer=leftTokenizer.withAddedTokens(Vector(name->values("clip_l").size))
      rightTokenizer=rightTokenizer.map(_.withAddedTokens(Vector(name->values("clip_g").size)))
    }
    val completed=scala.collection.mutable.ArrayBuffer.empty[String]
    var last=Map.empty[String,NativeJson]
    try
      ready.foreach { (path,params,values) =>
        val current=loaded.getOrElse(throw IllegalStateException("model was unloaded"))
        last=saveMixedEmbedding(params,current,context,Some(values))
        completed+=path.toString
      }
    catch
      case scala.util.control.NonFatal(error) if completed.nonEmpty =>
        throw new IllegalStateException(s"batch partially saved: ${completed.mkString(", ")}; ${error.getMessage}",error)
    Map("path"->Str(output),"paths"->Arr(completed.toVector.map(Str(_))),"warnings"->Arr(warnings.toVector.map(Str(_))),
      "encoders"->last("encoders"))

  private def saveMixedEmbedding(p: Map[String,NativeJson],pipeline: EulerImagePipeline,
      context: NativeWorkContext,mixedOverride: Option[NativeEmbeddingMixer.Vectors]=None): Map[String,NativeJson] =
    val output=str(p,"output","")
    val extension=if output.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then ".safetensors"
      else if output.toLowerCase(java.util.Locale.ROOT).endsWith(".pt") then ".pt"
      else throw IllegalArgumentException("embedding output must be safetensors or .pt")
    val target=Path.of(output).toAbsolutePath.normalize()
    val name=target.getFileName.toString.dropRight(extension.length)
    require(name.nonEmpty,"embedding name is empty")
    val overwrite=bool(p,"overwrite",false)
    val parent=target.getParent
    require(parent!=null && Files.isDirectory(parent),"embedding output directory does not exist")
    val previous=loadedEmbeddingInfo.filter(_.name.equalsIgnoreCase(name))
    require(previous.size<=1,"ambiguous active embedding name")
    val paths=p.get("embedding_paths") match
      case None => Vector.empty
      case Some(Arr(values)) if values.size<=100000 => values.map {
        case Str(path) => Path.of(path).toAbsolutePath.normalize()
        case _ => throw IllegalArgumentException("invalid embedding paths")
      }
      case _ => throw IllegalArgumentException("invalid embedding paths")
    require(!paths.exists(path=>path.getFileName.toString.replaceFirst("\\.[^.]*$","").equalsIgnoreCase(name) &&
      !path.toString.equalsIgnoreCase(target.toString)),"duplicate embedding name in catalog")
    if !overwrite && Files.exists(target) then throw new java.nio.file.FileAlreadyExistsException(target.toString)
    val streams=Vector(("clip_l",pipeline.tokenizer,pipeline.textEncoder.tokenEmbedding.weight.data))++
      pipeline.sdxlConditioning.toVector.map(c=>("clip_g",c.tokenizer2,c.encoder2.tokenEmbedding.weight.data))
    val mixed=mixedOverride.getOrElse(NativeEmbeddingInspector.prepareMix(p,streams,context))
    val left=mixed("clip_l").map(_.toArray)
    val (baseTokenizer,baseEncoder)=previous.headOption match
      case Some(existing) =>
        val (tokenizer,ids)=pipeline.tokenizer.withoutAddedTokenGroup(existing.name)
        (tokenizer,pipeline.textEncoder.withSelectedTokenEmbeddings(ids))
      case None => (pipeline.tokenizer,pipeline.textEncoder)
    val tokenizer=baseTokenizer.withAddedTokens(Vector(name->left.size))
    val encoder=baseEncoder.withAddedTokenEmbeddings(left)
    val conditioning=pipeline.sdxlConditioning.map { value =>
      val right=mixed("clip_g").map(_.toArray)
      val (baseTokenizer2,baseEncoder2)=previous.headOption match
        case Some(existing) =>
          val (tokenizer2,ids)=value.tokenizer2.withoutAddedTokenGroup(existing.name)
          (tokenizer2,value.encoder2.withSelectedTokenEmbeddings(ids))
        case None => (value.tokenizer2,value.encoder2)
      new SdxlTextConditioning(tokenizer,encoder,baseTokenizer2.withAddedTokens(Vector(name->right.size)),
        baseEncoder2.withAddedTokenEmbeddings(right),value.maxElements)
    }
    val next=new EulerImagePipeline(tokenizer,encoder,pipeline.unet,pipeline.vae,vaeExecution=pipeline.vaeExecution,
      unetExecution=pipeline.unetExecution,sdxlConditioning=conditioning,vaeTiling=pipeline.vaeTiling)
    val step=p.get("step") match
      case None | Some(Null) => None
      case Some(Number(value)) if value.isValidLong => Some(value.toLong)
      case _ => throw IllegalArgumentException("invalid training step")
    val stage=parent.resolve(s".embedding-${UUID.randomUUID()}$extension")
    try
      if extension==".safetensors" then
        ClipTextualInversion.writeSafetensors(stage,ClipTextualInversion(left,mixed.get("clip_g").map(_.map(_.toArray))),
          step=step,cancelled=context.cancelled)
      else
        import scalatorch.io.PyTorchTensorArchive.*
        def tensor(rows: Vector[Array[Float]]): Value = TensorValue(StoredTensor(StorageDType.Float32,
          Vector(rows.size.toLong,rows.head.length.toLong),rows.flatten.toArray))
        val value: Value=mixed.get("clip_g") match
          case Some(right) => DictValue(Map("clip_l"->tensor(left),"clip_g"->tensor(right.map(_.toArray)))++
            step.map(value=>"step"->IntegerValue(value)))
          case None => DictValue(Map("string_to_param"->DictValue(Map("*"->tensor(left))),"name"->StringValue(name),
            "step"->step.map(IntegerValue(_)).getOrElse(NoneValue)))
        scalatorch.io.PyTorchTensorArchiveWriter.write(stage,value,cancelled=context.cancelled)
      context.checkCancelled()
      context.beginPublication()
      if overwrite then Files.move(stage,target,java.nio.file.StandardCopyOption.ATOMIC_MOVE,java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      else Files.createLink(target,stage)
      loaded=Some(next)
      loadedEmbeddingTokens=loadedEmbeddingTokens.filterNot(_.equalsIgnoreCase(name))+name
      loadedEmbeddingInfo=loadedEmbeddingInfo.filterNot(_.name.equalsIgnoreCase(name)) :+ NativeEmbeddingInspector.LoadedEmbedding(name,target,left.size,
        encoder.finalNorm.width,conditioning.map(_.encoder2.finalNorm.width))
      val result=mixed.map { (key,rows) => key->Obj(Map("shape"->Arr(Vector(Number(rows.size),Number(rows.head.size))),
        "vectors"->Arr(rows.map(row=>Arr(row.map(value=>Number(BigDecimal(value.toDouble)))))))) }
      Map("path"->Str(output),"paths"->Arr(Vector(Str(output))),"warnings"->Arr(Vector.empty),"encoders"->Obj(result))
    finally Files.deleteIfExists(stage)

  private def executeVramEstimator(p: Map[String,NativeJson],context: NativeWorkContext): Map[String,NativeJson] =
    keys(p,Set("operation","profile","mode","width","height","batch","max_size","max_batch","steps",
      "session_id","output_dir","save_samples"))
    val operation=str(p,"operation","benchmark")
    require(Set("profile","estimate","benchmark")(operation),"VRAM estimator: unknown operation")
    val requestedProfile=p.get("profile").filter(_!=Null).orElse(vramLastProfile)
    val nativeSupported=usesCuda && loaded.nonEmpty
    val deviceIndex=Seq(vaeExecution.device,unetExecution.device).collectFirst {
      case value if value.startsWith("cuda:") => value.drop(5).toInt
    }.getOrElse(0)
    val memory=if nativeSupported then Some(scalatorch.ffm.CudaProbe.memoryInfo(deviceIndex)) else None
    val profileInfo=memory match
      case Some(info) => NativeVramEstimator.profile(modelSummary,info.deviceName,info.totalBytes,
        vaeExecution.device,unetExecution.device,requestedProfile,true,maxAttentionScoreElements,
        vaeExecution.offloadWeights,unetExecution.offloadWeights,vaeTilingEnabled,vaeAttentionLimit)
      case None => NativeVramEstimator.profile(modelSummary,"",0L,vaeExecution.device,unetExecution.device,requestedProfile,false,
        maxAttentionScoreElements,vaeExecution.offloadWeights,unetExecution.offloadWeights,vaeTilingEnabled,vaeAttentionLimit)
    operation match
      case "profile" => profileInfo
      case "estimate" => requestedProfile match
        case None => Map("available"->Bool(false),"reason"->Str("no_profile"))
        case Some(record) if !nativeSupported => Map("available"->Bool(false),"reason"->Str("unsupported"))
        case Some(record) if !profileInfo("compatible").asInstanceOf[Bool].value =>
          Map("available"->Bool(false),"reason"->Str("profile_identity_mismatch"),
            "current_fingerprint"->profileInfo("current_fingerprint"),
            "profile_fingerprint"->record.asInstanceOf[Obj].fields.getOrElse("fingerprint",Null))
        case Some(record) =>
          val info=memory.get
          val estimated=NativeVramEstimator.estimate(record,str(p,"mode","txt2img"),integer(p,"width",512),
            integer(p,"height",512),integer(p,"batch",1),(info.totalBytes-info.freeBytes).toDouble/1048576.0,
            info.totalBytes.toDouble/1048576.0)
          estimated++Map("fingerprint"->profileInfo("current_fingerprint"),
            "profile_status"->record.asInstanceOf[Obj].fields.getOrElse("status",Null))
      case "benchmark" =>
        if !nativeSupported then
          if !usesCuda then unsupported("VRAM estimator requires an active CUDA device")
          else throw new NativeBackendError("no_model","load a checkpoint before VRAM calibration")
        val identity=profileInfo("current_identity")
        val fingerprint=profileInfo("current_fingerprint").asInstanceOf[Str].value
        NativeVramEstimator.benchmark(p,identity,fingerprint,deviceIndex,context,value=>vramLastProfile=Some(value)) {
          (calibration,owner,inputPath,caseDir) =>
            val steps=integer(p,"steps",1)
            val hires=calibration.upscaler.map(name=>Obj(Map("enabled"->Bool(true),"scale"->Number(2),
              "resize_x"->Number(calibration.target),"resize_y"->Number(calibration.target),"upscaler"->Str(name),
              "second_pass_steps"->Number(steps),"denoising_strength"->Number(BigDecimal("0.7"))))).getOrElse(Null)
            val generation=Map[String,NativeJson]("prompt"->Str("an architectural city scene, daylight, detailed"),
              "negative_prompt"->Str("blurry, monochrome"),"seed"->Number(42),"steps"->Number(steps),
              "sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"),"cfg_scale"->Number(7),
              "width"->Number(calibration.width),"height"->Number(calibration.height),
              "batch_size"->Number(calibration.batch),"n_iter"->Number(1),"hooks"->Obj(Map.empty),
              "loras"->Arr(Vector.empty)) ++ (if hires==Null then Map.empty else Map("hires"->hires)) ++
              (if calibration.mode=="img2img" then Map("img2img"->Obj(Map("init_image"->Str(inputPath.toString),
                "denoising_strength"->Number(BigDecimal("0.75"))))) else Map.empty)
            val requestParams=Map[String,NativeJson]("params"->Obj(generation),"output_dir"->Str(caseDir.toString),
              "vae_recovery_dir"->Str(caseDir.toString),"preview_every"->Number(0),"preview_method"->Str("none"),
              "lora_paths"->Obj(Map.empty),"embedding_paths"->Arr(Vector.empty),"upscaler_paths"->Obj(Map.empty))
            val nested=new NativeWorkContext(owner.cancelled,()=>true,(_,_)=>())
            val result=execute(NativeRequest(UUID.randomUUID().toString,
              if calibration.mode=="img2img" then "img2img" else "txt2img",requestParams,None),nested)
            val images=result("images").asInstanceOf[Arr].values
            require(images.size==calibration.batch,"VRAM calibration returned an unexpected image count")
            images.foreach { image =>
              val fields=image.asInstanceOf[Obj].fields
              val expected=if calibration.upscaler.nonEmpty then calibration.target else calibration.width
              require(fields("width")==Number(expected) && fields("height")==Number(expected),
                "VRAM calibration returned unexpected image dimensions")
            }
        }.asInstanceOf[Obj].fields

  private def executeNativeRequest(request: NativeRequest,context: NativeWorkContext): Map[String,NativeJson] =
    val p=request.params
    context.checkCancelled()
    request.kind match
      case "upscale" => NativeStandaloneUpscale.execute(p,context,unetExecution.device)
      case "model_keyword_plan" =>
        val (res,next)=NativeKeywordMappings.planRequest(p,root,() => modelSummary.get("hash") match
          case Some(Str(hash)) => hash
          case _ => checkpointHash(context)
        ,keywordState,context)
        keywordState=next
        res
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
            "kind"->Str(if Files.isRegularFile(root) then singleFileKind else "diffusers"),"size_mb"->Number(BigDecimal(bytes)/BigDecimal(1048576)))))
        Map("models"->Arr(entries))
      case "list_samplers" =>
        keys(p,Set.empty)
        def sampler(name: String,schedule: String)=Obj(Map("name"->Str(name),"supported"->Bool(true),"default_scheduler"->Str(schedule)))
        Map("samplers"->Arr(Vector(
          sampler("DPM++ 2M","Karras"),sampler("DPM++ SDE","Karras"),sampler("DPM++ 2M SDE","Exponential"),
          sampler("DPM++ 2M SDE Heun","Exponential"),sampler("DPM++ 2S a","Karras"),sampler("DPM++ 3M SDE","Exponential"),
          sampler("Euler a","Uniform"),sampler("Euler","Uniform"),sampler("LMS","Uniform"),sampler("Heun","Uniform"),
          sampler("DPM2","Karras"),sampler("DPM2 a","Karras"),sampler("DDIM","Uniform"),sampler("UniPC","Uniform"),sampler("LCM","Uniform"))),
          "schedulers"->Arr(Vector("Automatic","Uniform","Karras","Exponential","Polyexponential","Normal","Simple","SGM Uniform","Align Your Steps","KL Optimal","DDIM","Beta").map(Str.apply)),
          "upscalers"->Arr((NativeHires.LatentModes++Vector("Nearest","Lanczos","R-ESRGAN 4x+","LDSR","Waifu2x")).map(Str.apply)),"preprocessors"->Arr(Vector("none","invert","inpaint","canny",
            "recolor_luminance","recolor_intensity","tile_resample","tile_colorfix","blur_gaussian",
            "depth","depth_marigold","normal_marigold","openpose","lineart","lineart_anime","softedge_hed","mlsd","normal_bae").map(Str.apply)))
      case "load_model" if Files.isRegularFile(root) =>
        keys(p,Set("path","dtype","vram_mode","vae_path","vae_tiling"))
        require(Path.of(str(p,"path",root.toString)).toRealPath()==root,"model path differs from configured checkpoint")
        if str(p,"dtype","fp32")!="fp32" || str(p,"vram_mode","cpu")!="cpu" then
          unsupported("native parameter loading requires fp32 and cpu; execution placement is configured separately")
        val selectedVae=vaeChoice(p,"vae_path")
        val tokenizerDir=sys.env.get("SDFX_SCALATORCH_TOKENIZER").map(Path.of(_)).getOrElse(root.getParent.resolve("tokenizer"))
        val tokenizer=ClipFastTokenizer.load(tokenizerDir.resolve("vocab.json"),tokenizerDir.resolve("merges.txt")).withNuiFallbackNormalization
        val tiling=bool(p,"vae_tiling",false)
        val before=Files.readAttributes(root,classOf[java.nio.file.attribute.BasicFileAttributes])
        val components=SingleFileSd1Model.load(root,maxAttentionScoreElements,() => context.checkCancelled(),vaeAttentionLimit)
        val sdxlText=components.clip2.map { clip2 =>
          val secondDir=sys.env.get("SDFX_SCALATORCH_TOKENIZER_2").map(Path.of(_)).getOrElse(tokenizerDir.resolveSibling("tokenizer_2"))
          val tokenizer2=(if Files.isRegularFile(secondDir.resolve("vocab.json")) && Files.isRegularFile(secondDir.resolve("merges.txt")) then
            if Files.isRegularFile(secondDir.resolve("tokenizer_config.json")) then ClipFastTokenizer.loadDirectory(secondDir)
            else ClipFastTokenizer.load(secondDir.resolve("vocab.json"),secondDir.resolve("merges.txt"),padToken="!")
          else ClipFastTokenizer.load(tokenizerDir.resolve("vocab.json"),tokenizerDir.resolve("merges.txt"),padToken="!")).withNuiFallbackNormalization
          new SdxlTextConditioning(tokenizer,components.clip,tokenizer2,clip2)
        }
        val (vae,vaeInfo)=selectedVae.fold((components.vae,Map[String,NativeJson]("vae_name"->Null,"vae_hash"->Null)))(p => readVae(Some(p),context,Some(components.vae.config)))
        val pipeline=new EulerImagePipeline(tokenizer,components.clip,components.unet,vae,
          vaeExecution=vaeExecution,unetExecution=unetExecution,sdxlConditioning=sdxlText,vaeTiling=tiling)
        val digest=java.security.MessageDigest.getInstance("SHA-256")
        val stream=Files.newInputStream(root)
        try
          val buffer=new Array[Byte](65536)
          var n=stream.read(buffer)
          while n != -1 do
            context.checkCancelled()
            digest.update(buffer,0,n)
            n=stream.read(buffer)
        finally stream.close()
        val after=Files.readAttributes(root,classOf[java.nio.file.attribute.BasicFileAttributes])
        require(before.size()==after.size() && before.lastModifiedTime()==after.lastModifiedTime() &&
          before.fileKey()==after.fileKey(),"checkpoint changed during loading or hashing")
        context.beginPublication()
        NativeBatch18Services.clearPreparedBatchInterrogates()
        loaded=Some(pipeline)
        vaeTilingEnabled=tiling
        loadedEmbeddingTokens=Set.empty
        loadedEmbeddingInfo=Vector.empty
        linearTargets=components.linearTargets
        singleFileFamily=components.family
        modelSummary=Map("name"->Str(root.getFileName.toString),"path"->Str(root.toString),"kind"->Str(singleFileKind),
          "hash"->Str(java.util.HexFormat.of().formatHex(digest.digest())),"family"->Str(if components.family=="sdxl" then "sdxl" else if components.family=="sd2" then "sd2" else "sd"),
          "model_token"->Str(UUID.randomUUID().toString))++vaeInfo
        modelSummary
      case "load_model" =>
        keys(p,Set("path","dtype","vram_mode","vae_path","vae_tiling"))
        require(Path.of(str(p,"path",root.toString)).toRealPath()==root,"model path differs from configured checkpoint")
        if str(p,"dtype","fp32")!="fp32" || str(p,"vram_mode","cpu")!="cpu" then
          unsupported("native parameter loading requires fp32 and cpu; U-Net/VAE execution placement is configured separately")
        val vaeTiling=bool(p,"vae_tiling",false)
        val selectedVae=vaeChoice(p,"vae_path")
        val sdxl=sdxlProfile()
        val tokenizer=(if sdxl then ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")) else
          ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"))).withNuiFallbackNormalization
        context.checkCancelled()
        val clipLoaded=PretrainedClip.loadWithLinearTargets(root.resolve("text_encoder"))
        val clip=clipLoaded.model
        context.checkCancelled()
        val (sdxlText,clip2Targets)=if !sdxl then (None,Map.empty[String,scalatorch.nn.Parameter]) else
          val tokenizer2=ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2")).withNuiFallbackNormalization
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
        val pipeline=new EulerImagePipeline(tokenizer,clip,unet,vae,vaeExecution=vaeExecution,unetExecution=unetExecution,
          sdxlConditioning=sdxlText,vaeTiling=vaeTiling)
        val hash=checkpointHash(context)
        context.beginPublication()
        NativeBatch18Services.clearPreparedBatchInterrogates()
        loaded=Some(pipeline)
        vaeTilingEnabled=vaeTiling
        loadedEmbeddingTokens=Set.empty
        loadedEmbeddingInfo=Vector.empty
        linearTargets=nextLinearTargets
        modelSummary=Map("name"->Str(root.getFileName.toString),"path"->Str(root.toString),"kind"->Str("diffusers"),
          "hash"->Str(hash),"family"->Str(if sdxl then "sdxl" else "sd"),"model_token"->Str(UUID.randomUUID().toString))++vaeInfo
        if usesCuda then
          val deviceIndex=Seq(vaeExecution.device,unetExecution.device).collectFirst {
            case value if value.startsWith("cuda:") => value.drop(5).toInt
          }.getOrElse(0)
          NativeSystemStatus.captureCudaMemory(deviceIndex)
        modelSummary
      case "set_vae" =>
        keys(p,Set("path"))
        val previous=loaded.getOrElse(throw new NativeBackendError("no_model","load a model first"))
        val (vae,info)=readVae(vaeChoice(p,"path"),context)
        require(vae.config.latentChannels==previous.vae.config.latentChannels &&
          vae.config.channels.size==previous.vae.config.channels.size,"replacement VAE latent geometry differs")
        val replacement=new EulerImagePipeline(previous.tokenizer,previous.textEncoder,previous.unet,vae,
          vaeExecution=previous.vaeExecution,unetExecution=previous.unetExecution,sdxlConditioning=previous.sdxlConditioning,
          vaeTiling=previous.vaeTiling)
        context.beginPublication()
        NativeBatch18Services.clearPreparedBatchInterrogates()
        loaded=Some(replacement)
        modelSummary=modelSummary++info++Map("model_token"->Str(UUID.randomUUID().toString))
        modelSummary
      case "unload" =>
        keys(p,Set.empty)
        context.beginPublication()
        NativeBatch18Services.clearPreparedBatchInterrogates()
        loaded=None
        loadedControlNet=None
        vaeTilingEnabled=false
        loadedEmbeddingTokens=Set.empty
        loadedEmbeddingInfo=Vector.empty
        linearTargets=Map.empty
        NativeSystemStatus.clearCudaMemory()
        modelSummary=Map.empty
        Map.empty
      case "txt2img" | "img2img" =>
        var hardwareStarted=System.nanoTime()
        val basePipeline=loaded.getOrElse(throw new NativeBackendError("no_model","load a model first"))
        val hardware=NativeHardwareInfo.options(p.get("hardware_info"))
        val hardwareProfile=hardware.map(_ => NativeHardwareInfo.profile(unetExecution.device))
        val pipeline=basePipeline
        val embeddingPaths=p.get("embedding_paths") match
          case None | Some(Null) => Vector.empty
          case Some(Arr(values)) => values.map { case Str(value) if value.nonEmpty && value.length<=4096 => value
            case _ => throw IllegalArgumentException("embedding_paths must contain nonempty paths") }
          case _ => throw IllegalArgumentException("embedding_paths must be an array")
        val loraPaths=p.get("lora_paths") match
          case None | Some(Null) => Map.empty[String,String]
          case Some(Obj(paths)) =>
            // NUI sends its catalog (including aliases) for keyword expansion;
            // the separate per-pass limit controls how many adapters are loaded.
            require(paths.size<=8192,"too many resolved LoRA paths")
            paths.map { (name,value) =>
            require(name.nonEmpty && name.length<=256,"invalid resolved LoRA name")
            val path=value match
              case Str(text) if text.nonEmpty && text.length<=4096 => text
              case _ => throw IllegalArgumentException(s"lora_paths.$name must be a path")
            name->path
          }
          case _ => throw IllegalArgumentException("lora_paths must be an object")
        val upscalerPaths=p.get("upscaler_paths") match
          case None | Some(Null) => Map.empty[String,String]
          case Some(Obj(paths)) =>
            require(paths.size<=32,"too many resolved upscaler paths")
            paths.map { (name,value) =>
              require(name.nonEmpty && name.length<=256,"invalid resolved upscaler name")
              val path=value match
                case Str(text) if text.nonEmpty && text.length<=4096 => text
                case _ => throw IllegalArgumentException(s"upscaler_paths.$name must be a path")
              name->path
            }
          case _ => throw IllegalArgumentException("upscaler_paths must be an object")
        val controlnetPaths=p.get("controlnet_paths") match
          case None | Some(Null) => Map.empty[String,String]
          case Some(Obj(paths)) =>
            require(paths.size<=16,"too many resolved ControlNet paths")
            paths.map { (name,value) =>
              require(name.nonEmpty && name.length<=256,"invalid resolved ControlNet name")
              val path=value match
                case Str(text) if text.nonEmpty && text.length<=4096 => text
                case _ => throw IllegalArgumentException(s"controlnet_paths.$name must be a path")
              name->path
            }
          case _ => throw IllegalArgumentException("controlnet_paths must be an object")
        val lbwPresets=p.get("lbw_presets") match
          case None | Some(Null) => Map.empty[String,String]
          case Some(Obj(values)) => values.map { (name,value) => value match
            case Str(text) => name->text
            case _ => throw IllegalArgumentException(s"lbw_presets.$name must be text")
          }
          case _ => throw IllegalArgumentException("lbw_presets must be an object")
        val systemInfoFields=p.get("system_info_fields") match
          case None | Some(Null) => Vector.empty[String]
          case Some(Arr(values)) => values.map {
            case Str(s) => s
            case _ => throw IllegalArgumentException("system_info_fields must contain strings")
          }
          case _ => throw IllegalArgumentException("system_info_fields must be an array")
        keys(NativeGenerationDefaults.strip(p--Set("lora_paths","embedding_paths","controlnet_paths","upscaler_paths","hires_loras","manual_loras","img2img_hires_loras","hardware_info","lbw_presets","system_info_fields"),NativeGenerationDefaults.envelope),
          Set("params","output_dir","preview_every","preview_method","vae_recovery_dir","upscaler_paths"))
        val previewEvery=integer(p,"preview_every",2)
        require(previewEvery>=0,"preview_every must be nonnegative")
        val previewMethod=str(p,"preview_method","approx")
        if !Set("none","approx","taesd").contains(previewMethod) then unsupported(s"unsupported preview method: $previewMethod")
        val g=p.get("params") match
          case Some(Obj(fields)) => fields
          case _ => throw IllegalArgumentException("params must be an object")
        val img2img=request.kind=="img2img"
        val baseDefaults=NativeGenerationDefaults.generation.removed("eta")
        val defaults=if img2img then baseDefaults.removed("img2img") else baseDefaults
        val allowed=Set("prompt","negative_prompt","width","height","steps","seed","subseed","cfg_scale","sampler_name","scheduler","clip_skip","batch_size","n_iter","checkpoint","vae","extra","hooks","hires","controlnet","eta")
        def parseLoras(value: Option[NativeJson]): Vector[(String,Float,Float,Option[Int],Option[Double],Option[Double],Option[String])] = value match
          case None => Vector.empty
          case Some(Arr(values)) if values.isEmpty => Vector.empty
          case Some(Arr(values)) =>
            require(values.size<=16,"at most 16 LoRAs may be applied per request")
            values.map {
            case Obj(fields) =>
              require(fields.keySet.subsetOf(Set("name","weight","unet","lbw","start","stop","dyn")),"unsupported LoRA fields")
              val blockWeight=fields.get("lbw") match
                case None | Some(Null) | Some(Str("")) => None
                case Some(Str(value)) => Some(value)
                case _ => throw IllegalArgumentException("LoRA lbw must be text")
              val dynamicRank=fields.get("dyn") match
                case None | Some(Null) => None
                case Some(Number(value)) if value.isValidInt => Some(value.toInt)
                case _ => throw IllegalArgumentException("LoRA dynamic rank must be an integer")
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
              def bound(key: String): Option[Double]=fields.get(key) match
                case None | Some(Null) => None
                case Some(Number(value)) if value.toDouble.isFinite => Some(value.toDouble)
                case _ => throw IllegalArgumentException(s"LoRA $key must be finite")
              (name,textScale,scale("unet",textScale.toDouble),dynamicRank,bound("start"),bound("stop"),blockWeight)
            case _ => throw IllegalArgumentException("loras must contain objects")
          }
          case _ => throw IllegalArgumentException("loras must be an array")
        val requestedLoras=parseLoras(g.get("loras"))
        val manualLoras=p.get("manual_loras") match
          case None | Some(Null) => requestedLoras
          case value => parseLoras(value)
        g.get("styles") match
          case None | Some(Null) => ()
          case Some(Arr(values)) if values.forall(_.isInstanceOf[Str]) => ()
          case _ => throw IllegalArgumentException("styles must be an array of strings")
        keys(NativeGenerationDefaults.strip(g--Set("loras","subseed_strength","styles"),defaults),if img2img then allowed+"img2img"+"loras" else allowed+"loras")
        // NUI applies the standard Hires pass only to txt2img; img2img uses the
        // separate img2img_hires_fix hook if refinement is requested.
        val hires=if img2img then None else NativeHires.parse(g.get("hires"),upscalerPaths)
        val difference=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("inpaint_difference").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value => Some(NativeInpaintDifference.parse(value))
          }
          case _ => None
        if difference.nonEmpty && !img2img then throw IllegalArgumentException("inpaint difference requires img2img")
        val temperatureGuard=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("gpu_temperature_guard").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case Bool(true) => Some(NativeTemperatureGuard.parse(Obj(Map.empty)))
            case value => Some(NativeTemperatureGuard.parse(value))
          }
          case _ => None
        val guidanceRescale=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("rescale_cfg") match
            case None | Some(Null) | Some(Bool(false)) => 0f
            case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => 0f
            case Some(Bool(true)) => .7f
            case Some(Obj(fields)) =>
              require(fields.keySet.subsetOf(Set("enabled","multiplier")),"unknown rescale_cfg option")
              val value=number(fields,"multiplier",BigDecimal("0.7")).toFloat
              require(value.isFinite && value>=0 && value<=1,"rescale_cfg multiplier must be in 0..1")
              value
            case _ => throw IllegalArgumentException("invalid rescale_cfg hook")
          case _ => 0f
        val cfgSchedule=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("cfg_schedule").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid cfg_schedule hook")
              require(fields.keySet.subsetOf(Set("enabled","start_cfg","end_cfg","curve","start","end")),"unknown cfg_schedule option")
              Some(CfgSchedule(number(fields,"start_cfg",7).toDouble,number(fields,"end_cfg",3).toDouble,
                str(fields,"curve","linear").toLowerCase(java.util.Locale.ROOT),number(fields,"start",0).toDouble,number(fields,"end",1).toDouble))
          }
          case _ => None
        if cfgSchedule.nonEmpty && number(g,"cfg_scale",7)<=1 then
          context.event("log",Map("level"->Str("warn"),"message"->Str("CFG schedule needs CFG scale > 1 (CFG is off); ignored")))
        val resharpen=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("resharpen").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid resharpen hook")
              require(fields.keySet.subsetOf(Set("enabled","sharpness","scaling")),"unknown resharpen option")
              Some(Resharpen(number(fields,"sharpness",BigDecimal("0.5")).toDouble,str(fields,"scaling","Flat")))
          }
          case _ => None
        val latentMirroring=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("latent_mirroring").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid latent_mirroring hook")
              require(fields.keySet.subsetOf(Set("enabled","mode","style","max_step","strength")),"unknown latent_mirroring option")
              Some(LatentMirroring(str(fields,"mode","Alternate Steps"),str(fields,"style","Vertical Mirroring"),
                number(fields,"max_step",BigDecimal("0.25")).toDouble,number(fields,"strength",1).toDouble))
          }
          case _ => None
        val colorGrading=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("color_grading").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid color_grading hook")
              require(fields.keySet.subsetOf(Set("enabled","recenter","recenter_strength","r","g","b","normalize","normalize_strength")),"unknown color_grading option")
              Some(LatentColorGrading(bool(fields,"recenter",true),number(fields,"recenter_strength",1).toDouble,
                number(fields,"r",0).toDouble,number(fields,"g",0).toDouble,number(fields,"b",0).toDouble,
                bool(fields,"normalize",false),number(fields,"normalize_strength",1).toDouble))
          }
          case _ => None
        val antiBurn=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("anti_burn").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid anti_burn hook")
              require(fields.keySet.subsetOf(Set("enabled","skip_last","average")),"unknown anti_burn option")
              Some(AntiBurn(integer(fields,"skip_last",2),integer(fields,"average",3)))
          }
          case _ => None
        val vectorscope=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("vectorscope_cc").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid vectorscope_cc hook")
              require(fields.keySet.subsetOf(Set("enabled","brightness","contrast","saturation","r","g","b","method","scaling","strength")),"unknown vectorscope_cc option")
              Some(LatentVectorscope(number(fields,"brightness",0).toDouble,number(fields,"contrast",0).toDouble,
                number(fields,"saturation",0).toDouble,number(fields,"r",0).toDouble,number(fields,"g",0).toDouble,
                number(fields,"b",0).toDouble,str(fields,"method","Straight"),str(fields,"scaling","Flat"),number(fields,"strength",1).toDouble))
          }
          case _ => None
        val freeU=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("freeu").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid freeu hook")
              require(fields.keySet.subsetOf(Set("enabled","b1","b2","s1","s2")),"unknown freeu option")
              val defaults=FreeU.defaults(basePipeline.sdxlConditioning.nonEmpty)
              Some(FreeUConfig(
                b1=number(fields,"b1",BigDecimal(defaults.b1.toString)).toFloat,
                b2=number(fields,"b2",BigDecimal(defaults.b2.toString)).toFloat,
                s1=number(fields,"s1",BigDecimal(defaults.s1.toString)).toFloat,
                s2=number(fields,"s2",BigDecimal(defaults.s2.toString)).toFloat))
          }
          case _ => None
        val (msPrediction,msZsnr)=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("model_sampling") match
            case None | Some(Null) | Some(Bool(false)) => (None,false)
            case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => (None,false)
            case Some(value) =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid model_sampling hook")
              require(fields.keySet.subsetOf(Set("enabled","sampling","zsnr")),"unknown model_sampling option")
              val sampling=str(fields,"sampling","default").toLowerCase(java.util.Locale.ROOT)
              val predOpt=sampling match
                case "default" | "" => None
                case "eps" | "epsilon" => Some(DiffusionPrediction.Epsilon)
                case "v_prediction" | "v" | "v_prediction_zsnr" => Some(DiffusionPrediction.VPrediction)
                case other => unsupported(s"model sampling '$other' is not supported (use default, eps, v_prediction)")
              val zsnr=bool(fields,"zsnr",false) || sampling=="v_prediction_zsnr"
              (predOpt,zsnr)
          case _ => (None,false)
        val activeNoise=if msZsnr then config.noise.withZeroTerminalSnr else config.noise
        val dynamicThresholding=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("dynamic_thresholding").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid dynamic_thresholding hook")
              require(fields.keySet.subsetOf(Set("enabled","mimic_scale","threshold_percentile","mimic_mode","mimic_scale_min",
                "cfg_mode","cfg_scale_min","sched_val","separate_feature_channels","scaling_startpoint","variability_measure","interpolate_phi")),
                "unknown dynamic_thresholding option")
              Some(DynamicThresholdingConfig(
                mimicScale=number(fields,"mimic_scale",7).toDouble,
                thresholdPercentile=number(fields,"threshold_percentile",1).toDouble,
                mimicMode=str(fields,"mimic_mode","Constant"),
                mimicScaleMin=number(fields,"mimic_scale_min",0).toDouble,
                cfgMode=str(fields,"cfg_mode","Constant"),
                cfgScaleMin=number(fields,"cfg_scale_min",0).toDouble,
                schedVal=number(fields,"sched_val",1).toDouble,
                separateFeatureChannels=bool(fields,"separate_feature_channels",true),
                scalingStartpoint=str(fields,"scaling_startpoint","MEAN"),
                variabilityMeasure=str(fields,"variability_measure","AD"),
                interpolatePhi=number(fields,"interpolate_phi",1).toDouble,
                trainingSigmas=activeNoise.floatTrainingSigmas))
          }
          case _ => None
        val tilingHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("tiling").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid tiling hook")
              require(fields.keySet.subsetOf(Set("enabled","tile_x","tile_y","start_step","stop_step")),"unknown tiling option")
              val tx=bool(fields,"tile_x",true)
              val ty=bool(fields,"tile_y",true)
              if !tx && !ty then None
              else Some(scalatorch.nn.CircularConvConfig(tileX=tx,tileY=ty,startStep=integer(fields,"start_step",0),stopStep=integer(fields,"stop_step",-1)))
          }
          case _ => None
        val cadsHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("cads").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid cads hook")
              require(fields.keySet.subsetOf(Set("enabled","tau1","tau2","noise_scale","mixing_factor","rescale","apply_to_uncond")),"unknown cads option")
              val t1=number(fields,"tau1",BigDecimal("0.6")).toDouble
              val t2=number(fields,"tau2",BigDecimal("0.9")).toDouble
              Some(CadsConfig(tau1=math.min(t1,t2),tau2=math.max(t1,t2),
                noiseScale=number(fields,"noise_scale",BigDecimal("0.25")).toDouble,
                mixingFactor=number(fields,"mixing_factor",1).toDouble,
                rescale=bool(fields,"rescale",true),
                applyToUncond=bool(fields,"apply_to_uncond",false)))
          }
          case _ => None
        val hyperTileHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("hypertile").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid hypertile hook")
              require(fields.keySet.subsetOf(Set("enabled","tile_size","swap_size","max_depth","scale_depth")),"unknown hypertile option")
              Some(HyperTileConfig(
                tileSize=integer(fields,"tile_size",256),
                swapSize=integer(fields,"swap_size",2),
                maxDepth=integer(fields,"max_depth",0),
                scaleDepth=bool(fields,"scale_depth",false)))
          }
          case _ => None
        val apgHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("apg").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid apg hook")
              require(fields.keySet.subsetOf(Set("enabled","momentum","adaptive_momentum","norm_threshold","eta","guidance_limiter","guidance_sigma_start","guidance_sigma_end")),"unknown apg option")
              Some(ApgConfig(
                momentum=number(fields,"momentum",BigDecimal("0.5")).toDouble,
                adaptiveMomentum=number(fields,"adaptive_momentum",BigDecimal("0.18")).toDouble,
                normThreshold=number(fields,"norm_threshold",BigDecimal("15.0")).toDouble,
                eta=number(fields,"eta",1).toDouble,
                guidanceLimiter=bool(fields,"guidance_limiter",false),
                guidanceSigmaStart=number(fields,"guidance_sigma_start",BigDecimal("5.42")).toDouble,
                guidanceSigmaEnd=number(fields,"guidance_sigma_end",BigDecimal("0.28")).toDouble))
          }
          case _ => None
        val mahiroHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("mahiro") match
            case None | Some(Null) | Some(Bool(false)) => false
            case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => false
            case Some(Bool(true)) => true
            case Some(Obj(fields)) =>
              require(fields.keySet.subsetOf(Set("enabled")),"unknown mahiro option")
              bool(fields,"enabled",true)
            case _ => throw IllegalArgumentException("invalid mahiro hook")
          case _ => false
        val pagHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("pag").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid pag hook")
              require(fields.keySet.subsetOf(Set("enabled","scale","adaptive_scale","layers")),"unknown pag option")
              val scale=number(fields,"scale",3).toDouble
              if scale<=0 then None
              else
                val layers=fields.get("layers") match
                  case None | Some(Null) => Vector("mid")
                  case Some(Str(s)) => s.split(",").map(_.trim).filter(_.nonEmpty).toVector
                  case Some(Arr(items)) => items.collect { case Str(s) if s.trim.nonEmpty => s.trim }.toVector
                  case _ => throw IllegalArgumentException("pag layers must be a string or array")
                Some(PagConfig(scale=scale,adaptiveScale=number(fields,"adaptive_scale",0).toDouble,layers=if layers.isEmpty then Vector("mid") else layers))
          }
          case _ => None
        val weightGradientHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("weight_gradient") match
            case None | Some(Null) | Some(Bool(false)) => false
            case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => false
            case Some(Bool(true)) => true
            case Some(Obj(fields)) =>
              require(fields.keySet.subsetOf(Set("enabled")),"unknown weight_gradient option")
              bool(fields,"enabled",true)
            case _ => throw IllegalArgumentException("invalid weight_gradient hook")
          case _ => false
        val negPipHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("negpip") match
            case None | Some(Null) | Some(Bool(false)) => false
            case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => false
            case Some(Bool(true)) => true
            case Some(Obj(fields)) =>
              require(fields.keySet.subsetOf(Set("enabled")),"unknown negpip option")
              bool(fields,"enabled",true)
            case _ => throw IllegalArgumentException("invalid negpip hook")
          case _ => false
        val kohyaHrFixHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("kohya_hrfix").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid kohya_hrfix hook")
              require(fields.keySet.subsetOf(Set("enabled","block_number","downscale_factor","start_percent","end_percent","downscale_after_skip","downscale_method","upscale_method")),"unknown kohya_hrfix option")
              val dm=str(fields,"downscale_method","bicubic")
              val um=str(fields,"upscale_method","bicubic")
              val cfg=KohyaHrFixConfig(
                blockNumber=integer(fields,"block_number",3),
                downscaleFactor=number(fields,"downscale_factor",2).toDouble,
                startPercent=number(fields,"start_percent",0).toDouble,
                endPercent=number(fields,"end_percent",BigDecimal("0.35")).toDouble,
                downscaleAfterSkip=bool(fields,"downscale_after_skip",true),
                downscaleMethod=if KohyaHrFix.InterpMethods.contains(dm) then dm else "bicubic",
                upscaleMethod=if KohyaHrFix.InterpMethods.contains(um) then um else "bicubic")
              cfg.validate(basePipeline.unet.downBlocks.map(_.layers.size))
              Some(cfg)
          }
          case _ => None
        val tokenDownsamplingHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("token_downsampling").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid token_downsampling hook")
              require(fields.keySet.subsetOf(Set("enabled","factor","factor2","disable_after")),"unknown token_downsampling option")
              Some(TokenDownsamplingConfig(
                factor=number(fields,"factor",1).toDouble,
                factor2=number(fields,"factor2",1).toDouble,
                disableAfter=number(fields,"disable_after",1).toDouble))
          }
          case _ => None
        val condHighresMode=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("conditioning_highres_fix").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid conditioning_highres_fix hook")
              require(fields.keySet.subsetOf(Set("enabled","mode")),"unknown conditioning_highres_fix option")
              val m=str(fields,"mode","disabled").trim.toLowerCase(java.util.Locale.ROOT)
              require(ConditioningHighresFix.Modes.contains(m),s"Conditioning Highres Fix: unknown conditioning highres mode '$m'")
              if m=="disabled" then None else Some(m)
          }
          case _ => None
        val latentModifierHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("latent_modifier").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid latent_modifier hook")
              require(fields.keySet.subsetOf(Set("enabled","sharpness_multiplier","sharpness_method","tonemap_multiplier","tonemap_method","tonemap_percentile","contrast_multiplier","combat_method","combat_cfg_drift","extra_noise_type","extra_noise_multiplier","extra_noise_lowpass","divisive_norm_size","divisive_norm_multiplier","spectral_mod_mode","spectral_mod_percentile","spectral_mod_multiplier")),"unknown latent_modifier option")
              Some(LatentModifierConfig(
                sharpnessMultiplier=number(fields,"sharpness_multiplier",0).toDouble,
                sharpnessMethod=str(fields,"sharpness_method","anisotropic").toLowerCase(java.util.Locale.ROOT),
                tonemapMultiplier=number(fields,"tonemap_multiplier",0).toDouble,
                tonemapMethod=str(fields,"tonemap_method","reinhard").toLowerCase(java.util.Locale.ROOT),
                tonemapPercentile=number(fields,"tonemap_percentile",100).toDouble,
                contrastMultiplier=number(fields,"contrast_multiplier",0).toDouble,
                combatMethod=str(fields,"combat_method","subtract").toLowerCase(java.util.Locale.ROOT),
                combatCfgDrift=number(fields,"combat_cfg_drift",0).toDouble,
                extraNoiseType=str(fields,"extra_noise_type","gaussian").toLowerCase(java.util.Locale.ROOT),
                extraNoiseMultiplier=number(fields,"extra_noise_multiplier",0).toDouble,
                extraNoiseLowpass=integer(fields,"extra_noise_lowpass",100),
                divisiveNormSize=integer(fields,"divisive_norm_size",127),
                divisiveNormMultiplier=number(fields,"divisive_norm_multiplier",0).toDouble,
                spectralModMode=str(fields,"spectral_mod_mode","hard_clamp").toLowerCase(java.util.Locale.ROOT),
                spectralModPercentile=number(fields,"spectral_mod_percentile",5).toDouble,
                spectralModMultiplier=number(fields,"spectral_mod_multiplier",0).toDouble))
          }
          case _ => None
        val softInpaintingHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("soft_inpainting").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid soft_inpainting hook")
              require(fields.keySet.subsetOf(Set("enabled","mask_blend_power","mask_blend_scale","inpaint_detail_preservation","composite_mask_influence","composite_difference_threshold","composite_difference_contrast")),"unknown soft_inpainting option")
              Some(SoftInpaintingConfig(
                maskBlendPower=number(fields,"mask_blend_power",1).toDouble,
                maskBlendScale=number(fields,"mask_blend_scale",BigDecimal("0.5")).toDouble,
                inpaintDetailPreservation=number(fields,"inpaint_detail_preservation",4).toDouble,
                compositeMaskInfluence=number(fields,"composite_mask_influence",0).toDouble,
                compositeDifferenceThreshold=number(fields,"composite_difference_threshold",BigDecimal("0.5")).toDouble,
                compositeDifferenceContrast=number(fields,"composite_difference_contrast",2).toDouble))
          }
          case _ => None
        val multiDiffusionHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("multidiffusion").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid multidiffusion hook")
              require(fields.keySet.subsetOf(Set("enabled","method","tile_width","tile_height","tile_overlap","tile_batch_size")),"unknown multidiffusion option")
              Some(MultiDiffusionConfig(
                method=str(fields,"method","MultiDiffusion"),
                tileWidth=integer(fields,"tile_width",768),
                tileHeight=integer(fields,"tile_height",768),
                tileOverlap=integer(fields,"tile_overlap",64),
                tileBatchSize=integer(fields,"tile_batch_size",4)))
          }
          case _ => None
        val neverOomHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("neveroom").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid neveroom hook")
              require(fields.keySet.subsetOf(Set("enabled","unet_offload","vae_tiling","encoder_tile_size","decoder_tile_size")),"unknown neveroom option")
              Some(NeverOomConfig(
                unetOffload=bool(fields,"unet_offload",false),
                vaeTiling=bool(fields,"vae_tiling",false),
                encoderTileSize=integer(fields,"encoder_tile_size",1024),
                decoderTileSize=integer(fields,"decoder_tile_size",128)))
          }
          case _ => None
        val semanticGuidanceHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("semantic_guidance").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid semantic_guidance hook")
              require(fields.keySet.subsetOf(Set("enabled","prompt","negative_prompt","warmup","edit_guidance_scale","tail_percentage_threshold","momentum_scale","momentum_beta","concepts","scale","cutoff","neg_guidance","normalize_threshold","threshold")),"unknown semantic_guidance option")
              val sgPrompt=if fields.contains("prompt") then str(fields,"prompt","") else str(fields,"concepts","")
              val sgNeg=str(fields,"negative_prompt","")
              val sgScale=if fields.contains("edit_guidance_scale") then number(fields,"edit_guidance_scale",1).toDouble else number(fields,"scale",1).toDouble
              val sgTail=if fields.contains("tail_percentage_threshold") then number(fields,"tail_percentage_threshold",BigDecimal("0.05")).toDouble else if fields.contains("cutoff") then number(fields,"cutoff",BigDecimal("0.05")).toDouble else if fields.contains("normalize_threshold") then number(fields,"normalize_threshold",BigDecimal("0.05")).toDouble else number(fields,"threshold",BigDecimal("0.05")).toDouble
              if SemanticGuidance.parseConcepts(sgPrompt,sgNeg).isEmpty then None
              else Some(SemanticGuidanceConfig(
                prompt=sgPrompt,
                negativePrompt=sgNeg,
                warmup=integer(fields,"warmup",10),
                editGuidanceScale=sgScale,
                tailPercentageThreshold=sgTail,
                momentumScale=number(fields,"momentum_scale",BigDecimal("0.3")).toDouble,
                momentumBeta=number(fields,"momentum_beta",BigDecimal("0.6")).toDouble))
          }
          case _ => None
        val sonarHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("sonar").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid sonar hook")
              require(fields.keySet.subsetOf(Set("enabled","sampler","momentum","momentum_hist","momentum_hist_init","momentum_sign","ref_img","ref_meth","ref_hgf","ref_min_step","ref_max_step","upscale","upscaler","upscale_ratio","upscale_width","upscale_height","direction","init","rand_type","power","scaling","cfg_t","linear_T","ref_weight","hide_orig","upscale_scale","upscale_steps","upscale_denoise","upscale_mom","upscale_hist")),"unknown sonar option")
              val histInit=if fields.contains("momentum_hist_init") then str(fields,"momentum_hist_init","zero") else str(fields,"init","zero").toLowerCase(java.util.Locale.ROOT) match { case "zero" => "zero"; case "rand_init" | "initial" => "rand_init"; case _ => "rand_new" }
              val signStr=if fields.contains("momentum_sign") then str(fields,"momentum_sign","pos") else if number(fields,"direction",1).toDouble < 0 then "neg" else "pos"
              val refHgfVal=if fields.contains("ref_hgf") then number(fields,"ref_hgf",BigDecimal("0.2")).toDouble else number(fields,"ref_weight",BigDecimal("0.2")).toDouble
              val upRatio=if fields.contains("upscale_ratio") then number(fields,"upscale_ratio",1).toDouble else number(fields,"upscale_scale",BigDecimal("1.5")).toDouble
              Some(SonarConfig(
                sampler=str(fields,"sampler","Euler a"),
                momentum=number(fields,"momentum",BigDecimal("0.95")).toDouble,
                momentumHist=number(fields,"momentum_hist",BigDecimal("0.75")).toDouble,
                momentumHistInit=histInit,
                momentumSign=signStr,
                refImg=str(fields,"ref_img",""),
                refMeth=str(fields,"ref_meth","linear").toLowerCase(java.util.Locale.ROOT),
                refHgf=refHgfVal,
                refMinStep=number(fields,"ref_min_step",0).toDouble,
                refMaxStep=number(fields,"ref_max_step",BigDecimal("0.75")).toDouble,
                upscale=bool(fields,"upscale",false),
                upscaler=str(fields,"upscaler","Lanczos"),
                upscaleRatio=upRatio,
                upscaleWidth=integer(fields,"upscale_width",0),
                upscaleHeight=integer(fields,"upscale_height",0)))
          }
          case _ => None
        val hiresFixTweaksHook=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("hires_fix_tweaks").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid hires_fix_tweaks hook")
              require(fields.keySet.subsetOf(Set("enabled","batch_count","seed_enabled","seed","extras","subseed","strength","resize_width","resize_height","prompt","negative_prompt","seed_offset","subseed_strength","repeat","seed_resize_from_w","seed_resize_from_h")),"unknown hires_fix_tweaks option")
              val bCount=if fields.contains("batch_count") then integer(fields,"batch_count",1) else integer(fields,"repeat",1)
              val sOffset=integer(fields,"seed_offset",0)
              val sExplicit=number(fields,"seed",0).toLong
              val sEnabled=bool(fields,"seed_enabled",fields.contains("seed") || fields.contains("seed_offset") || fields.contains("subseed") || fields.contains("subseed_strength") || fields.contains("strength") || fields.contains("resize_width") || fields.contains("seed_resize_from_w"))
              val sExtras=bool(fields,"extras",fields.contains("subseed") || fields.contains("subseed_strength") || fields.contains("strength") || fields.contains("resize_width") || fields.contains("seed_resize_from_w"))
              val subVal=number(fields,"subseed",0).toLong
              val strVal=if fields.contains("strength") then number(fields,"strength",0).toDouble else number(fields,"subseed_strength",0).toDouble
              val rw=if fields.contains("resize_width") then integer(fields,"resize_width",0) else integer(fields,"seed_resize_from_w",0)
              val rh=if fields.contains("resize_height") then integer(fields,"resize_height",0) else integer(fields,"seed_resize_from_h",0)
              val plan=HiresFixTweaks.seedPlan(
                batchCount=bCount,
                seedEnabled=sEnabled,
                seed=sExplicit,
                extras=sExtras,
                subseed=subVal,
                strength=strVal,
                resizeWidth=rw,
                resizeHeight=rh,
                baseSeeds=Vector(0L),
                baseSubseeds=Vector(0L),
                baseStrength=0.0)
              Some(plan.copy(seedOffset=sOffset))
          }
          case _ => None
        g.get("hooks") match
          case Some(Obj(hooks)) =>
            hooks.get("pnginfo_injection").foreach {
              case Null | Bool(_) => ()
              case Obj(fields) => require(fields.keySet.subsetOf(Set("enabled","mode","custom_keys")),"unknown pnginfo_injection option")
              case _ => throw IllegalArgumentException("invalid pnginfo_injection hook")
            }
            hooks.get("prevent_interrupt").foreach {
              case Null | Bool(_) => ()
              case Obj(fields) => require(fields.keySet.subsetOf(Set("enabled","mode")),"unknown prevent_interrupt option")
              case _ => throw IllegalArgumentException("invalid prevent_interrupt hook")
            }
          case _ => ()
        val detailDaemon=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("detail_daemon").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value =>
              val fields=value match
                case Bool(true) => Map.empty[String,NativeJson]
                case Obj(v) => v
                case _ => throw IllegalArgumentException("invalid detail_daemon hook")
              require(fields.keySet.subsetOf(Set("enabled","mode","amount","start","end","bias","exponent","start_offset","end_offset","fade","smooth")),"unknown detail_daemon option")
              Some(DetailDaemon(str(fields,"mode","both").toLowerCase(java.util.Locale.ROOT),number(fields,"amount",BigDecimal("0.1")).toDouble,
                number(fields,"start",BigDecimal("0.2")).toDouble,number(fields,"end",BigDecimal("0.8")).toDouble,
                number(fields,"bias",BigDecimal("0.5")).toDouble,number(fields,"exponent",1).toDouble,
                number(fields,"start_offset",0).toDouble,number(fields,"end_offset",0).toDouble,
                number(fields,"fade",0).toDouble,bool(fields,"smooth",true),activeNoise.floatTrainingSigmas.map(_.toDouble)))
          }
          case _ => None
        val latentUpscale=g.get("hooks") match
          case Some(Obj(hooks)) => hooks.get("latent_upscale").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value => Some(NativeLatentUpscale.parse(value))
          }
          case _ => None
        if latentUpscale.nonEmpty && !img2img then unsupported("latent upscale requires img2img")
        val hiresFix=g.get("hooks") match
          case None => None
          case Some(Obj(hooks)) => hooks.get("img2img_hires_fix").flatMap {
            case Null | Bool(false) => None
            case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
            case value => Some(NativeImg2ImgHires.parse(value,upscalerPaths))
          }
          case Some(_) => throw IllegalArgumentException("hooks must be an object")
        val refinementLoras=p.get("img2img_hires_loras") match
          case None | Some(Null) => None
          case value => Some(parseLoras(value))
        val hiresLoras=p.get("hires_loras") match
          case None | Some(Null) => None
          case value => Some(parseLoras(value))
        val nineChannelModel=pipeline.unet.convIn.weight.data.shape.dims(1)==2*pipeline.vae.config.latentChannels+1
        var remainingAdapterElements=1073741824L
        val adapterCache=scala.collection.mutable.Map.empty[(Path,Option[Int]),Map[scalatorch.nn.Parameter,Vector[LinearAdapterScope.Applied]]]
        val adapterWindows=new java.util.IdentityHashMap[scalatorch.nn.LowRankAdapter,LoraStepWindow]()
        val adaptersByName=scala.collection.mutable.HashMap.empty[String,scala.collection.mutable.ArrayBuffer[scalatorch.nn.LowRankAdapter]]
        val unetTargets=linearTargets.iterator.collect { case (name,parameter) if name.startsWith("unet.") => parameter }.toSet
        val namesByParameter=linearTargets.iterator.map((name,parameter)=>parameter->name).toMap
        def loadAdapters(requested: Vector[(String,Float,Float,Option[Int],Option[Double],Option[Double],Option[String])]) = requested.flatMap { (name,textScale,unetScale,dynamicRank,start,stop,blockWeight) =>
          val path=loraPaths.get(name).orElse(loraPaths.iterator.collectFirst {
            case (alias,path) if alias.equalsIgnoreCase(name) => path
          }).getOrElse(throw IllegalArgumentException(s"missing resolved path for LoRA: $name"))
          val file=Path.of(path).toRealPath()
          val lower=file.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
          require(Vector(".safetensors",".pt",".pth",".bin",".ckpt").exists(lower.endsWith) && Files.isRegularFile(file),
            s"LoRA tensor archive is missing or unsupported: $name")
          val loadedAdapter=adapterCache.getOrElseUpdate((file,dynamicRank), {
            val loaded=DiffusionLoRA.load(file,linearTargets,elementBudget=remainingAdapterElements,dynamicRank=dynamicRank)
            val used=loaded.valuesIterator.flatten.map { case (adapter,_) =>
              adapter.a.data.shape.numel+adapter.b.data.shape.numel
            }.sum
            remainingAdapterElements-=used
            loaded
          })
          val blockPlan=blockWeight.map(value=>LoraBlockWeights.plan(LoraBlockWeights.parse(value,pipeline.sdxlConditioning.nonEmpty,lbwPresets),
            pipeline.sdxlConditioning.nonEmpty,unetScale,textScale))
          loadedAdapter.toVector.flatMap { (parameter,items) =>
            val scale=blockPlan match
              case None => if unetTargets(parameter) then unetScale else textScale
              case Some(plan) =>
                val target=namesByParameter(parameter)
                LoraBlockWeights.targetScale(plan,pipeline.sdxlConditioning.nonEmpty,target).getOrElse(
                  unsupported(s"LoRA block weights cannot resolve target: $target")).toFloat
            items.map { case (adapter,multiplier) =>
              val instance=adapter.copy()
              adaptersByName.getOrElseUpdate(name,scala.collection.mutable.ArrayBuffer.empty)+=instance
              if unetTargets(parameter) && (start.nonEmpty || stop.nonEmpty) then
                adapterWindows.put(instance,LoraStepWindow(start,stop))
              parameter -> (instance,multiplier*scale)
            }
          }
        }.groupMap(_._1)(_._2).view.mapValues(_.toVector).toMap
        val refinementAdapters=refinementLoras.map(loadAdapters)
        val hiresAdapters=if hires.nonEmpty then hiresLoras.map(loadAdapters) else None
        val prediction=msPrediction.getOrElse(NativeGenerationMetadata.prediction(g.get("extra"),config.prediction,cudaAvailable=usesCuda))
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
        val subseedNumber=number(g,"subseed",-1)
        require(subseedNumber.isValidLong,"subseed must be a signed 64-bit integer")
        val subseed=if subseedNumber<0 then new java.security.SecureRandom().nextLong(4294967295L) else subseedNumber.toLong
        val variationStrength=number(g,"subseed_strength",0).toDouble
        require(variationStrength.isFinite && variationStrength>=0 && variationStrength<=1,"variation strength must be in 0..1")
        require(batchSize>=1 && batchSize<=16,"batch_size must be between 1 and 16")
        require(nIter>=1 && nIter<=100,"n_iter must be between 1 and 100")
        require(iterations<=1600,"batch_size*n_iter must be at most 1600")
        require(BigInt(seed)+iterations-1<=Long.MaxValue,"iteration seeds exceed signed 64-bit range")
        require(BigInt(subseed)+iterations-1<=Long.MaxValue,"iteration subseeds exceed signed 64-bit range")
        val steps=integer(g,"steps",20)
        require(steps>0 && steps<=1000,"steps must be between 1 and 1000")
        val saveIntermediateHook=NativeIntermediateRecorder.parseSaveIntermediate(g.get("hooks"))
        val stepsAnimationHook=NativeIntermediateRecorder.parseStepsAnimation(g.get("hooks"),batchSize,nIter,saveIntermediateHook.isDefined)
        val img2imgAltHook=NativeIntermediateRecorder.parseImg2ImgAlt(g.get("hooks"))
        val cdTunerHook=NativeIntermediateRecorder.parseCdTuner(g.get("hooks"))
        val llulHook=NativeIntermediateRecorder.parseLlul(g.get("hooks"))
        val agentAttentionHook=NativeIntermediateRecorder.parseAgentAttention(g.get("hooks"))
        val incantationsHook=NativeIntermediateRecorder.parseIncantations(g.get("hooks"))
        val chgHook=NativeIntermediateRecorder.parseCharacteristicGuidance(g.get("hooks"))
        val clipIoHook=NativeIntermediateRecorder.parseClipIo(g.get("hooks"))
        val promptFusionHook=NativeIntermediateRecorder.parsePromptFusion(g.get("hooks"))
        val neutralPromptHook=NativeIntermediateRecorder.parseNeutralPrompt(g.get("hooks"))
        val composableLoraHook=NativeIntermediateRecorder.parseComposableLora(g.get("hooks"))
        val aestheticGradientsHook=NativeIntermediateRecorder.parseAestheticGradients(g.get("hooks"))
        val sagHook=NativeIntermediateRecorder.parseSag(g.get("hooks"))
        val referenceHook=NativeIntermediateRecorder.parseReference(g.get("hooks"))
        val revisionHook=NativeIntermediateRecorder.parseRevision(g.get("hooks"))
        val ipAdapterHook=NativeIntermediateRecorder.parseIpAdapter(g.get("hooks"))
        val controlLLLiteHook=NativeIntermediateRecorder.parseControlLLLite(g.get("hooks"))
        val fooocusInpaintHook=NativeIntermediateRecorder.parseFooocusInpaint(g.get("hooks"))
        val compileHook=NativeIntermediateRecorder.parseCompile(g.get("hooks"))
        val adetailerUnits=NativeBatch18Services.parseAdetailer(g.get("hooks"))
        val faceRestoreHook=NativeBatch18Services.parseFaceRestore(g.get("hooks"))
        val nsfwCheckerHook=NativeBatch18Services.parseNsfwChecker(g.get("hooks"))
        val aestheticScoreHook=NativeBatch18Services.parseAestheticScore(g.get("hooks"))
        val animateDiffHook=NativeBatch18Services.parseAnimateDiff(g.get("hooks"))
        require(steps>0 && steps<=1000,"steps must be between 1 and 1000")
        val baseImageRequest=DiffusionImageRequest(str(g,"prompt",""),str(g,"negative_prompt",""),
          integer(g,"width",512),integer(g,"height",512),seed,number(g,"cfg_scale",7).toFloat,
          clipSkip=integer(g,"clip_skip",1),sdxlSize=sdxlSize,
          torchCpuNoise=NativeGenerationMetadata.torchCpuNoise(g.get("extra"),cudaAvailable=usesCuda),guidanceRescale=guidanceRescale,cfgSchedule=cfgSchedule,resharpen=resharpen,latentMirroring=latentMirroring,colorGrading=colorGrading,antiBurn=antiBurn,vectorscope=vectorscope,detailDaemon=detailDaemon,dynamicThresholding=dynamicThresholding,tiling=tilingHook,cads=cadsHook,hyperTile=hyperTileHook,apg=apgHook,mahiro=mahiroHook,pag=pagHook,trainingSigmas=activeNoise.floatTrainingSigmas,weightGradient=weightGradientHook,negPip=negPipHook,kohyaHrFix=kohyaHrFixHook,tokenDownsampling=tokenDownsamplingHook,latentModifier=latentModifierHook,softInpainting=softInpaintingHook,multiDiffusion=multiDiffusionHook,semanticGuidance=semanticGuidanceHook,sonar=sonarHook,cdTuner=cdTunerHook,llul=llulHook,agentAttention=agentAttentionHook,incantations=incantationsHook,characteristicGuidance=chgHook,clipIo=clipIoHook,promptFusion=promptFusionHook,neutralPrompt=neutralPromptHook,composableLora=composableLoraHook,aestheticGradients=aestheticGradientsHook,sag=sagHook,reference=referenceHook,revision=revisionHook,ipAdapter=ipAdapterHook,controlLLLite=controlLLLiteHook,fooocusInpaint=fooocusInpaintHook,compile=compileHook,controlNetPaths=controlnetPaths)
        val fallbackControlImage=if img2img then g.get("img2img") match
          case Some(Obj(fields)) => fields.get("init_image").collect { case Str(path) if path.nonEmpty => path }
          case _ => None
        else None
        val controlNet=NativeControlNet.prepare(g.get("controlnet"),controlnetPaths,fallbackControlImage,
          baseImageRequest.width,baseImageRequest.height,context.cancelled,path=>controlNetModel(path,context))
        compileHook.filter(_.enabled).foreach { cCfg =>
          val activeHookNames=g.get("hooks") match
            case Some(Obj(hooks)) => hooks.collect {
              case (k,Bool(true)) => k
              case (k,Obj(f)) if !f.get("enabled").contains(Bool(false)) => k
            }.toSet
            case _ => Set.empty[String]
          val (_,promptLorasCheck)=NativeLoraPrompt.parse(baseImageRequest.prompt)
          CompileAccel.validateJob(cCfg,activeHookNames,controlNet.nonEmpty,requestedLoras.nonEmpty || promptLorasCheck.nonEmpty,composableLoraHook.exists(_.enabled))
        }
        val imageRequest=baseImageRequest.copy(controlNets=controlNet)
        val keywordPlan=NativeKeywordHook.plan(g.get("hooks"),imageRequest.prompt,
          modelSummary("hash").asInstanceOf[Str].value,root.getFileName.toString,iterations,seed,keywordState,
          () => context.checkCancelled(),allowActive=Set("img2img_hires_fix","latent_upscale","inpaint_difference","gpu_temperature_guard","rescale_cfg","cfg_schedule","resharpen","latent_mirroring","color_grading","anti_burn","vectorscope_cc","detail_daemon","freeu","dynamic_thresholding","tiling","cads","hypertile","model_sampling","apg","mahiro","pag","weight_gradient","negpip","kohya_hrfix","token_downsampling","conditioning_highres_fix","multidiffusion","latent_modifier","soft_inpainting","neveroom","semantic_guidance","sonar","hires_fix_tweaks","pnginfo_injection","prevent_interrupt","save_intermediate","steps_animation","img2imgalt","cd_tuner","llul","agent_attention","incantations","characteristic_guidance","clip_io","prompt_fusion","neutral_prompt","composable_lora","embedding_merge","aesthetic_gradients","sag","reference","revision","ip_adapter","controllllite","fooocus_inpaint","compile","adetailer","batch_face_swap","face_restore","nsfw_checker","aesthetic_score","animatediff","photomaker"))
        val keywordRefinement=hiresFix.filter(_.prompt.isEmpty).flatMap(_ => keywordPlan.map(_.preview))
        val keywordRefinementAdapters=keywordRefinement.map { text =>
          val refs=manualLoras++NativeLoraPrompt.parse(text)._2
          require(refs.size<=16,"at most 16 LoRAs may be applied per request")
          loadAdapters(refs)
        }
        val hiresPrompt=hires.map(_.prompt).filter(_.nonEmpty).getOrElse("")
        val hiresNegative=hires.map(_.negativePrompt).filter(_.nonEmpty).getOrElse("")
        val embeddingPrompts=Vector(imageRequest.prompt,hiresPrompt,hiresFix.fold("")(_.prompt))++
          keywordPlan.toVector.flatMap(plan => plan.preview+:plan.prompts)
        val embeddingNegatives=Vector(imageRequest.negativePrompt,hiresNegative,hiresFix.fold("")(_.negativePrompt))++img2imgAltHook.toVector.flatMap(a => Vector(a.overridePrompt,a.overrideNegative))
        val (loadedPipeline,nextEmbeddingTokens,embeddingWarnings,addedEmbeddings)=withEmbeddings(basePipeline,embeddingPaths,
          embeddingPrompts.map(NativeLoraPrompt.parse(_)._1).mkString(" "),
          embeddingNegatives.map(NativeLoraPrompt.parse(_)._1).mkString(" "),loadedEmbeddingTokens,context)
        loaded=Some(loadedPipeline)
        val generationPipeline=neverOomHook.fold(loadedPipeline) { noCfg =>
          val uExec=loadedPipeline.unetExecution match
            case c: UNetExecution.Cuda if noCfg.unetOffload => c.copy(offloadWeights=true)
            case u => u
          new EulerImagePipeline(loadedPipeline.tokenizer,loadedPipeline.textEncoder,loadedPipeline.unet,loadedPipeline.vae,
            vaeExecution=loadedPipeline.vaeExecution,unetExecution=uExec,sdxlConditioning=loadedPipeline.sdxlConditioning,
            vaeTiling=loadedPipeline.vaeTiling || noCfg.vaeTiling)
        }
        loadedEmbeddingTokens=nextEmbeddingTokens
        loadedEmbeddingInfo=loadedEmbeddingInfo++addedEmbeddings
        var preparedInpaint: Option[NativeInpaint.Prepared]=None
        var baseSteps=steps
        def fullInpaintSource(image: scalatorch.Tensor[Float],strength: Double): DiffusionImageSource =
          val dims=image.shape.dims
          val pixels=Math.toIntExact(Math.multiplyExact(dims(2),dims(3)))
          val mask=scalatorch.Tensor.fromFloats(Array.fill(pixels)(1f),scalatorch.Shape.from(Seq(1L,1L,dims(2),dims(3))))
          val masked=condHighresMode.flatMap(m => ConditioningHighresFix.resolveWeight(m,Some(strength))).fold(scalatorch.Tensor.fromFloats(new Array[Float](Math.multiplyExact(pixels,3)),image.shape))(w => ConditioningHighresFix.conditionPixels(image,mask,w))
          DiffusionImageSource(image,strength,mask=Some(mask),originalImage=Some(image),finalOverlay=false,maskedImage=Some(masked))
        val source=if !img2img then
          if !nineChannelModel then None else
            require(imageRequest.width>0 && imageRequest.height>0 &&
              imageRequest.width.toLong*imageRequest.height<=16777216L,"invalid nine-channel base image dimensions")
            val pixels=Math.toIntExact(Math.multiplyExact(imageRequest.width.toLong,imageRequest.height.toLong))
            val gray=scalatorch.Tensor.fromFloats(Array.fill(Math.multiplyExact(pixels,3))(127f/127.5f-1f),
              scalatorch.Shape.from(Seq(1L,3L,imageRequest.height.toLong,imageRequest.width.toLong)))
            Some(fullInpaintSource(gray,1.0))
        else
          val fields=g.get("img2img") match
            case Some(Obj(fields)) => fields
            case _ => throw IllegalArgumentException("img2img must be an object")
          val inpaintKeys=Set("mask","mask_blur","inpainting_fill","inpaint_full_res","inpaint_full_res_padding","inpainting_mask_invert")
          keys(NativeGenerationDefaults.strip(fields--inpaintKeys,NativeGenerationDefaults.inpainting),Set("init_image","denoising_strength","resize_mode"))
          val strength=number(fields,"denoising_strength",BigDecimal("0.75")).toDouble
          require(strength.isFinite && strength>=0 && strength<=1,"denoising_strength must be between 0 and 1")
          val effectiveStrength=latentUpscale match
            case Some(_) =>
              val (count,effective,_)=NativeLatentUpscale.fullSteps(steps,strength)
              baseSteps=count
              effective
            case None => strength
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
          val name=str(fields,"init_image","")
          require(name.nonEmpty,"init_image is required")
          val width=imageRequest.width;val height=imageRequest.height
          require(width>0 && height>0 && width.toLong*height<=16777216L,"invalid img2img output dimensions")
          context.checkCancelled()
          val input=if difference.nonEmpty then DiffusionImageInput.readRgb(Path.of(name),maxBytes=200000000L)
            else DiffusionImageInput.readRgb(Path.of(name))
          val differenceMask=difference.map { settings =>
            val base=DiffusionImageInput.readRgb(Path.of(settings.basePath),maxBytes=200000000L)
            val pixels=NativeInpaintDifference.mask(base,input,settings.threshold,settings.dilation,settings.erosion,context.cancelled,settings.contours)
            val image=new java.awt.image.BufferedImage(input.getWidth,input.getHeight,java.awt.image.BufferedImage.TYPE_BYTE_GRAY)
            for y <- 0 until input.getHeight; x <- 0 until input.getWidth do
              image.getRaster.setSample(x,y,0,pixels(y*input.getWidth+x))
            image
          }
          context.checkCancelled()
          val source=differenceMask.orElse(maskName.map(path => DiffusionImageInput.readMask(Path.of(path)))) match
            case Some(mask) =>
              val prepared=NativeInpaint.prepare(input,mask,width,height,if latentUpscale.nonEmpty then 0 else mode,maskBlur,invert,context.cancelled,
                fullRes=fullRes,fullResPadding=fullResPadding,fill=fill,seed=seed)
              preparedInpaint=Some(prepared)
              val unetInputs=generationPipeline.unet.convIn.weight.data.shape.dims(1)
              val sourcePrepared=if latentUpscale.nonEmpty && !fullRes then
                NativeInpaint.prepare(input,mask,input.getWidth,input.getHeight,0,maskBlur,invert,context.cancelled,
                  fullRes=false,fullResPadding=fullResPadding,fill=fill,seed=seed)
              else prepared
              val factor=1 << (generationPipeline.vae.decoder.upBlocks.size-1)
              val (sourceImage,sourceMask)=if latentUpscale.nonEmpty then NativeInpaint.sourceForVae(sourcePrepared,factor,context.cancelled)
                else (sourcePrepared.image,sourcePrepared.mask)
              val maskedImage=if latentUpscale.nonEmpty || unetInputs==2*generationPipeline.vae.config.latentChannels+1 then
                Some(condHighresMode.flatMap(m => ConditioningHighresFix.resolveWeight(m,Some(strength))).fold(NativeInpaint.maskedImage(sourceImage,sourceMask,context.cancelled))(w => ConditioningHighresFix.conditionPixels(sourceImage,sourceMask,w))) else None
              val transform=latentUpscale.map(settings => (latents: scalatorch.Tensor[Float]) =>
                NativeHires.resize(latents,width/factor,height/factor,NativeLatentUpscale.mode(settings.method)))
              Some(DiffusionImageSource(sourceImage,effectiveStrength,mask=Some(prepared.mask),originalImage=Some(prepared.originalImage),
                finalOverlay=prepared.pasteBox.isEmpty,maskedImage=maskedImage,latentTransform=transform))
            case None =>
              val factor=1 << (generationPipeline.vae.decoder.upBlocks.size-1)
              val sourceWidth=if latentUpscale.nonEmpty then input.getWidth/factor*factor else width
              val sourceHeight=if latentUpscale.nonEmpty then input.getHeight/factor*factor else height
              require(sourceWidth>0 && sourceHeight>0,"latent upscale source is smaller than the VAE scale")
              val resized=if sourceWidth==input.getWidth && sourceHeight==input.getHeight then input
                else DiffusionImageResize(input,sourceWidth,sourceHeight,if latentUpscale.nonEmpty then 0 else mode)
              val plane=sourceWidth*sourceHeight
              val values=new Array[Float](plane*3)
              for y <- 0 until sourceHeight do
                context.checkCancelled()
                for x <- 0 until sourceWidth do
                  val rgb=resized.getRGB(x,y)
                  for channel <- 0 until 3 do
                    values(channel*plane+y*sourceWidth+x)=((rgb >>> (16-8*channel))&255)/255f*2f-1f
              val transform=latentUpscale.map(settings => (latents: scalatorch.Tensor[Float]) =>
                NativeHires.resize(latents,width/factor,height/factor,NativeLatentUpscale.mode(settings.method)))
              val image=scalatorch.Tensor.fromFloats(values,
                scalatorch.Shape.from(Seq(1L,3L,sourceHeight.toLong,sourceWidth.toLong)))
              Some(if nineChannelModel && transform.isEmpty then fullInpaintSource(image,effectiveStrength)
                else DiffusionImageSource(image,effectiveStrength,latentTransform=transform))
          context.checkCancelled()
          source
        val allSupportedSamplers=Set("Euler","Euler a","Heun","LMS","DPM2","DPM2 a","DPM++ 2S a","DPM++ SDE","DPM++ 2M","DPM++ 2M SDE","DPM++ 2M SDE Heun","DPM++ 3M SDE","DDIM","UniPC","LCM")
        def canonicalSampler(name: String): String =
          val aliases=Map(
            "k_euler"->"Euler","k_euler_a"->"Euler a","k_dpmpp_2m"->"DPM++ 2M","heun"->"Heun","k_heun"->"Heun","ddim"->"DDIM",
            "k_lms"->"LMS","k_dpm_2"->"DPM2","k_dpm_2_a"->"DPM2 a","k_dpmpp_2s_a"->"DPM++ 2S a","k_dpmpp_sde"->"DPM++ SDE",
            "k_dpmpp_2m_sde"->"DPM++ 2M SDE","k_dpmpp_2m_sde_heun"->"DPM++ 2M SDE Heun","k_dpmpp_3m_sde"->"DPM++ 3M SDE",
            "unipc"->"UniPC","lcm"->"LCM",
            "DPM++ 2M Karras"->"DPM++ 2M","DPM++ SDE Karras"->"DPM++ SDE","DPM++ 2M SDE Karras"->"DPM++ 2M SDE",
            "DPM++ 2M SDE Exponential"->"DPM++ 2M SDE","DPM2 Karras"->"DPM2","DPM2 a Karras"->"DPM2 a",
            "DPM++ 2S a Karras"->"DPM++ 2S a","DPM++ 3M SDE Karras"->"DPM++ 3M SDE","DPM++ 3M SDE Exponential"->"DPM++ 3M SDE"
          )
          aliases.getOrElse(name,allSupportedSamplers.find(_.equalsIgnoreCase(name)).getOrElse(name))
        def oldCanonicalSampler(name: String): String =
          val aliases=Map("k_euler"->"Euler","k_euler_a"->"Euler a","k_dpmpp_2m"->"DPM++ 2M","heun"->"Heun","ddim"->"DDIM",
            "DPM++ 2M Karras"->"DPM++ 2M")
          aliases.getOrElse(name,Vector("Euler","Euler a","DPM++ 2M","DDIM").find(_.equalsIgnoreCase(name)).getOrElse(name))
        def canonicalSchedule(name: String): String =
          val value=name.trim
          if value.equalsIgnoreCase("sgm_uniform") then "SGM Uniform"
          else if value.equalsIgnoreCase("align_your_steps") || value.equalsIgnoreCase("ays") then "Align Your Steps"
          else if value.equalsIgnoreCase("kl_optimal") then "KL Optimal"
          else Vector("Automatic","Uniform","Karras","Exponential","Polyexponential","Normal","Simple","SGM Uniform","Align Your Steps","KL Optimal","DDIM","Beta")
            .find(_.equalsIgnoreCase(value)).getOrElse(value)
        def resolvedSchedule(name: String,samplerName: String,sampler: String): String =
          if name!="Automatic" then name
          else if samplerName.endsWith(" Karras") then "Karras"
          else if samplerName.endsWith(" Exponential") then "Exponential"
          else if Set("DPM++ 2M","DPM++ SDE","DPM++ 2S a","DPM2","DPM2 a")(sampler) then "Karras"
          else if Set("DPM++ 2M SDE","DPM++ 2M SDE Heun","DPM++ 3M SDE")(sampler) then "Exponential"
          else "Uniform"
        def oldResolvedSchedule(name: String,samplerName: String,sampler: String): String =
          if name!="Automatic" then name
          else if samplerName.endsWith(" Karras") || sampler=="DPM++ 2M" then "Karras" else "Uniform"
        val samplerName=sonarHook.fold(str(g,"sampler_name","DPM++ 2M"))(_.effectiveSampler)
        val sampler=canonicalSampler(samplerName)
        val ddimEta=g.get("eta") match
          case None | Some(Null) => 0.0
          case Some(Number(value)) => value.toDouble
          case _ => throw IllegalArgumentException("eta must be a number or null")
        require(ddimEta.isFinite,"eta must be finite")
        val etaProvided=g.get("eta").exists(_!=Null)
        val requestedSchedule=canonicalSchedule(latentUpscale.map(_.scheduler).filterNot(_=="automatic")
          .getOrElse(str(g,"scheduler","Automatic")))
        val schedule=resolvedSchedule(requestedSchedule,samplerName,sampler)
        val supportedSchedules=Set("Uniform","Karras","Exponential","Polyexponential","Normal","Simple","SGM Uniform","Align Your Steps","KL Optimal","DDIM","Beta")
        if !allSupportedSamplers(sampler) || !supportedSchedules(schedule) then
          unsupported(s"unsupported sampler/schedule: $sampler/$schedule")
        if sampler=="Euler a" && prediction==DiffusionPrediction.Sample then
          unsupported("Euler a does not support sample prediction")
        val hiresSamplerName=hires.flatMap(_.sampler).getOrElse(samplerName)
        val hiresSampler=canonicalSampler(hiresSamplerName)
        val hiresScheduleRequest=canonicalSchedule(hires.flatMap(_.scheduler).getOrElse(requestedSchedule))
        val hiresSchedule=resolvedSchedule(hiresScheduleRequest,hiresSamplerName,hiresSampler)
        if hires.nonEmpty && (!allSupportedSamplers(hiresSampler) ||
            !supportedSchedules(hiresSchedule)) then
          unsupported(s"unsupported Hires sampler/schedule: $hiresSampler/$hiresSchedule")
        if hires.nonEmpty && hiresSampler=="Euler a" && prediction==DiffusionPrediction.Sample then
          unsupported("Euler a does not support sample prediction")
        val outputName=str(p,"output_dir","")
        require(outputName.nonEmpty,"output_dir is required")
        val output=Path.of(outputName).toAbsolutePath.normalize()
        require(Files.isDirectory(output),"output_dir must be an existing directory")
        val recoveryName=str(p,"vae_recovery_dir","")
        val recovery=if recoveryName.isEmpty then output else Path.of(recoveryName).toAbsolutePath.normalize()
        val started=System.nanoTime()
        val staging=Files.createTempDirectory(output,".native-staging-")
        val staged=scala.collection.mutable.ArrayBuffer.empty[Path]
        val published=scala.collection.mutable.ArrayBuffer.empty[Path]
        val samplerWarnings=scala.collection.mutable.LinkedHashSet.empty[String]
        def warnUnusedEta(name: String): Unit =
          if etaProvided && name!="DDIM" then samplerWarnings += s"eta is ignored by sampler '$name'"
        if img2img && variationStrength>0 then samplerWarnings+="variation seed is only applied in txt2img"
        if condHighresMode.nonEmpty && !nineChannelModel then samplerWarnings+="Conditioning Highres Fix has no effect without a 9-channel inpainting model"
        if latentModifierHook.nonEmpty && number(g,"cfg_scale",7)<=1 then samplerWarnings+="Latent Modifier requires CFG scale > 1; ignored"
        if softInpaintingHook.nonEmpty && (!img2img || source.flatMap(_.mask).isEmpty) then samplerWarnings+="Soft Inpainting has no effect without an inpaint mask"
        if softInpaintingHook.nonEmpty && nineChannelModel && source.flatMap(_.mask).nonEmpty then samplerWarnings+="Soft Inpainting per-step blending is skipped for 9-channel inpainting models"
        val altActive=img2imgAltHook.exists { _ =>
          val ok=img2img && source.flatMap(_.mask).isEmpty && !nineChannelModel
          if !ok then samplerWarnings+="img2img alternative test needs plain img2img with a normal model; ignored"
          ok
        }
        val intermediateSession=if saveIntermediateHook.isDefined || stepsAnimationHook.isDefined then
          Some(new NativeIntermediateRecorder.Session(output,saveIntermediateHook,stepsAnimationHook,Map("prompt"->Str(imageRequest.prompt),"negative_prompt"->Str(imageRequest.negativePrompt),"sampler"->Str(samplerName),"steps"->Number(steps),"cfg_scale"->Number(number(g,"cfg_scale",7)),"mode"->Str(request.kind),"preview_method"->Str(previewMethod)),context.cancelled))
        else None
        val sonarRefTensor=sonarHook.filter(_.refImg.trim.nonEmpty).flatMap { sCfg =>
          if sCfg.sampler!="Naive" then
            samplerWarnings+="Sonar reference image is only supported with the Naive sampler; ignoring reference image"
            None
          else
            val refBuffered=DiffusionImageInput.readRgb(Path.of(sCfg.refImg.trim))
            val resizedRef=if refBuffered.getWidth==imageRequest.width && refBuffered.getHeight==imageRequest.height then refBuffered else DiffusionImageResize(refBuffered,imageRequest.width,imageRequest.height,0)
            Some(NativeImg2ImgHires.tensor(resizedRef,context.cancelled))
        }
        tokenDownsamplingHook.foreach { td =>
          if td.enabledFor(steps) && (pagHook.isDefined || hyperTileHook.isDefined) then
            unsupported("Token Downsampling cannot combine with PAG or HyperTile: their self-attention layouts differ")
        }
        def ancestralSchedule(count: Int,requested: String): EulerSchedule =
          val spacing=requested match
            case "Uniform" | "Normal" => DiffusionTimestepSpacing.Linspace
            case "Simple" | "SGM Uniform" => DiffusionTimestepSpacing.Trailing
            case "DDIM" => DiffusionTimestepSpacing.Leading
            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler 'Euler a'; using Karras"
              samplerWarnings += s"schedule '$requested' is not supported by sampler 'Euler a'; using the sampler's default spacing"
              config.spacing
            case _ =>
              samplerWarnings += s"Euler a does not support $requested in the NUI scheduler; using checkpoint timestep spacing"
              config.spacing
          activeNoise.euler(count,spacing,if spacing==DiffusionTimestepSpacing.Leading then config.stepsOffset else 0)
        def ddimSchedule(count: Int,requested: String): DdimSchedule =
          val spacing=requested match
            case "Uniform" | "Normal" => DiffusionTimestepSpacing.Linspace
            case "Simple" | "SGM Uniform" => DiffusionTimestepSpacing.Trailing
            case "DDIM" => DiffusionTimestepSpacing.Leading
            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler 'DDIM'; using Karras"
              samplerWarnings += s"schedule '$requested' is not supported by sampler 'DDIM'; using the sampler's default spacing"
              config.spacing
            case _ =>
              samplerWarnings += s"schedule '$requested' is not supported by sampler 'DDIM'; using the sampler's default spacing"
              config.spacing
          DdimSchedule(activeNoise,count,spacing,if spacing==DiffusionTimestepSpacing.Leading then config.stepsOffset else 0)
        def eulerSchedule(count: Int,requested: String): EulerSchedule = requested match









          case "Karras" => activeNoise.karrasEuler(count)
          case "Exponential" | "Polyexponential" => activeNoise.exponentialEuler(count)
          case "Normal" => activeNoise.normalEuler(count)
          case "Simple" => activeNoise.simpleEuler(count)
          case "SGM Uniform" => activeNoise.sgmUniformEuler(count)
          case "Align Your Steps" => activeNoise.alignYourStepsEuler(count,generationPipeline.sdxlConditioning.nonEmpty)
          case "KL Optimal" => activeNoise.klOptimalEuler(count)
          case "DDIM" => activeNoise.ddimEuler(count)
          case "Beta" => activeNoise.betaEuler(count)
          case _ => activeNoise.nuiUniformEuler(count)


        def dpmSchedule(count: Int,requested: String,samplerName: String="DPM++ 2M"): DpmSchedule =
          if count>activeNoise.betas.size && Set("Uniform","Normal","Simple","SGM Uniform")(requested) then
            samplerWarnings += "repeated DPM sigma steps use a stable first-order fallback"
          requested match
            case "Karras" => DpmSchedule.karras(activeNoise,count)
            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler '$samplerName'; using Karras"
              DpmSchedule.karras(activeNoise,count)
            case "Exponential" => DpmSchedule.exponential(activeNoise,count)
            case "Simple" | "SGM Uniform" => DpmSchedule.trailing(activeNoise,count)
            case "DDIM" => DpmSchedule.leading(activeNoise,count,config.stepsOffset)
            case "Beta" => DpmSchedule.beta(activeNoise,count)
            case _ => DpmSchedule.uniform(activeNoise,count)
        def dpmSinglestepSchedule(count: Int,requested: String): DpmSchedule =
          requested match
            case "Karras" => DpmSchedule.karras(activeNoise,count)
            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler 'DPM++ 2S a'; using Karras"
              DpmSchedule.karras(activeNoise,count)
            case "Exponential" => DpmSchedule.exponential(activeNoise,count)
            case "Beta" => DpmSchedule.beta(activeNoise,count)
            case _ =>
              samplerWarnings += s"schedule '$requested' is not supported by sampler 'DPM++ 2S a'; using the sampler's default spacing"
              DpmSchedule.uniform(activeNoise,count)
        def discreteEulerSchedule(count: Int,requested: String,samplerName: String,roundKarrasTimesteps: Boolean=false): EulerSchedule =
          val (spacing,conversion)=requested match
            case "Karras" => (DiffusionTimestepSpacing.Linspace,"karras")
            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler '$samplerName'; using Karras"
              (DiffusionTimestepSpacing.Linspace,"karras")
            case "Exponential" => (DiffusionTimestepSpacing.Linspace,"exponential")
            case "Beta" => (DiffusionTimestepSpacing.Linspace,"beta")
            case "Simple" | "SGM Uniform" => (DiffusionTimestepSpacing.Trailing,"none")
            case "DDIM" => (DiffusionTimestepSpacing.Leading,"none")
            case _ => (DiffusionTimestepSpacing.Linspace,"none")
          activeNoise.discreteSchedule(count,spacing,if spacing==DiffusionTimestepSpacing.Leading then config.stepsOffset else 0,conversion,roundKarrasTimesteps)
        /*

            samplerWarnings += "repeated DPM sigma steps use a stable first-order fallback"
          requested match

            case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
              samplerWarnings += s"schedule '$requested' is not available for sampler 'DPM++ 2M'; using Karras"





        */
        var tinyPreviewFailed=false
        try
          val images=Vector.tabulate(iterations) { index =>
            context.checkCancelled()
            val rawPrompt=keywordPlan.fold(imageRequest.prompt)(_.prompts(index))
            val (cleanPrompt,_)=NativeLoraPrompt.parse(rawPrompt)
            val batchPrompt=keywordPlan.fold(imageRequest.prompt)(_.prompts((index/batchSize)*batchSize))
            val (_,promptLoras)=NativeLoraPrompt.parse(batchPrompt)
            val (cleanNegative,_)=NativeLoraPrompt.parse(imageRequest.negativePrompt)
            val selected=(if keywordPlan.nonEmpty then manualLoras else requestedLoras)++promptLoras
            require(selected.size<=16,"at most 16 LoRAs may be applied per request")
            adaptersByName.clear()
            val adapters=loadAdapters(selected)
            val namedAdps=adaptersByName.view.mapValues(_.toVector).toMap
            val effPassPrompt=if composableLoraHook.exists(_.enabled) then rawPrompt else cleanPrompt
            val current=imageRequest.copy(seed=seed+index,prompt=effPassPrompt,negativePrompt=cleanNegative,
              variationSeed=Option.when(!img2img && variationStrength>0)(subseed+index),
              variationStrength=if img2img then 0.0 else variationStrength,
              latentHookSeed=Some(seed+(index/batchSize)*batchSize),latentHookBatchSize=batchSize,latentHookBatchIndex=index%batchSize,
              loraStepFilter=Some((step,total,adapter)=>Option(adapterWindows.get(adapter)).forall(_.active(step,total))),sonarReferenceImage=sonarRefTensor,namedLoraAdapters=namedAdps,onWarning=msg => samplerWarnings+=msg)
            def passPrompt(text: String,negative: String,explicit: Option[Map[scalatorch.nn.Parameter,Vector[LinearAdapterScope.Applied]]]) =
              val (clean,tags)=NativeLoraPrompt.parse(text)
              val (cleanNeg,_)=NativeLoraPrompt.parse(negative)
              val selection=explicit.orElse(Option.when(tags.nonEmpty) {
                val refs=(if keywordPlan.nonEmpty then manualLoras else requestedLoras)++tags
                require(refs.size<=16,"at most 16 LoRAs may be applied per request")
                loadAdapters(refs)
              })
              (clean,cleanNeg,selection)
            def withPassAdapters[A](selection: Option[Map[scalatorch.nn.Parameter,Vector[LinearAdapterScope.Applied]]])(body: => A): A =
              selection match
                case None => body
                case Some(chosen) =>
                  val cleared=adapters.keysIterator.map(_ -> Vector.empty[LinearAdapterScope.Applied]).toMap
                  LinearAdapterScope.withAdapters(cleared++chosen)(body)
            def progress(pass: String): DiffusionImageProgress => Unit =
              val guard=temperatureGuard.map { config =>
                new NativeTemperatureGuard.Guard(config,()=>NativeTemperatureGuard.read(config),()=>context.checkCancelled(),(kind,temp)=>
                  if kind!="reading" || config.printReadings then
                    val message=s"GPU temperature $kind: $temp C (provider=${config.provider}, device=${config.device})"+
                      (if kind=="timeout" then "; maximum pause reached, continuing before wake temperature" else "")
                    context.event("log",Map("level"->Str(if kind=="timeout" then "warn" else "info"),"message"->Str(message))))
              }
              value =>
                guard.foreach(_.step())
                context.event("progress",Map(
                  "step"->Number(value.completedSteps),"total"->Number(value.totalSteps),"pass"->Str(pass),
                  "eta_s"->Number(0),"image"->Number(index),"images"->Number(iterations),"batch_size"->Number(batchSize)))
                if previewEvery>0 && previewMethod!="none" && value.completedSteps%previewEvery==0 && value.completedSteps<value.totalSteps then
                  context.checkCancelled()
                  NativePreview.emitSafely(context,value.completedSteps) {
                    val xl=generationPipeline.sdxlConditioning.nonEmpty
                    if previewMethod!="taesd" || tinyPreviewFailed then NativePreview.jpegBase64(value.latents,sdxl=xl)
                    else try tinyPreview.render(value.latents,xl,()=>context.checkCancelled())
                    catch
                      case e: java.util.concurrent.CancellationException => throw e
                      case scala.util.control.NonFatal(e) =>
                        context.checkCancelled()
                        tinyPreviewFailed=true
                        context.event("log",Map("level"->Str("warn"),"message"->Str(s"TAESD preview unavailable: ${e.getMessage}; using approximate preview")))
                        NativePreview.jpegBase64(value.latents,sdxl=xl)
                  }
            val image=try LinearAdapterScope.withAdapters(adapters) { FreeU.withConfig(freeU) {
              def runImagePass(request: DiffusionImageRequest,count: Int,callback: DiffusionImageProgress => Unit,
                  imageSource: Option[DiffusionImageSource],latentSource: Option[DiffusionLatentSource],
                  passSampler: String,passSchedule: String): scalatorch.Tensor[Float] =
                // Diffusers' SD1 image/inpaint pipelines accept the keyword through
                // **kwargs but do not apply guidance_rescale. SDXL does apply it.
                val actualRequest=if generationPipeline.sdxlConditioning.isEmpty && (imageSource.nonEmpty || latentSource.nonEmpty) then
                  request.copy(guidanceRescale=0f) else request
                warnUnusedEta(passSampler)
                if passSampler=="DPM++ 2M SDE" || passSampler=="DPM++ 2M SDE Heun" || passSampler=="DPM++ 3M SDE" then
                  val layout=dpmSchedule(count,passSchedule,passSampler)
                  generationPipeline.generateDpmMultistepSde(actualRequest,layout,order=if passSampler=="DPM++ 3M SDE" then 3 else 2,
                    heunVariant=(passSampler=="DPM++ 2M SDE Heun"),onProgress=callback,cancelled=context.cancelled,prediction=prediction,
                    source=imageSource,latentSource=latentSource)
                else if passSampler=="UniPC" then
                  val layout=dpmSchedule(count,passSchedule,"UniPC")
                  generationPipeline.generateUniPc(actualRequest,layout,onProgress=callback,cancelled=context.cancelled,prediction=prediction,
                    source=imageSource,latentSource=latentSource)
                else if passSampler=="LCM" then
                  passSchedule match
                    case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
                      samplerWarnings += s"schedule '$passSchedule' is not available for sampler 'LCM'; using Karras"
                      samplerWarnings += s"schedule '$passSchedule' is not supported by sampler 'LCM'; using the sampler's default spacing"
                    case "Karras" | "Exponential" | "Beta" =>
                      samplerWarnings += s"schedule '$passSchedule' is not supported by sampler 'LCM'; using the sampler's default spacing"
                    case _ => ()
                  if passSchedule!="Uniform" then samplerWarnings += "LCM ignores the schedule type"
                  generationPipeline.generateLcm(actualRequest,LcmSchedule(activeNoise,count),onProgress=callback,cancelled=context.cancelled,
                    prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="LMS" then
                  generationPipeline.generateLms(actualRequest,discreteEulerSchedule(count,passSchedule,"LMS"),onProgress=callback,
                    cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="DPM2" || passSampler=="DPM2 a" then
                  generationPipeline.generateKdpm2(actualRequest,discreteEulerSchedule(count,passSchedule,passSampler,roundKarrasTimesteps=true),ancestral=(passSampler=="DPM2 a"),
                    onProgress=callback,cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="DPM++ 2S a" then
                  generationPipeline.generateDpm2Sa(actualRequest,dpmSinglestepSchedule(count,passSchedule),onProgress=callback,
                    cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="DPM++ SDE" then
                  generationPipeline.generateDpmSde(actualRequest,discreteEulerSchedule(count,passSchedule,"DPM++ SDE"),onProgress=callback,
                    cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="DDIM" then
                  generationPipeline.generateDdim(actualRequest,ddimSchedule(count,passSchedule),config.ddimClipSample,config.ddimClipRange,
                    eta=ddimEta,onProgress=callback,cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="DPM++ 2M" then
                  val layout=dpmSchedule(count,passSchedule)
                  generationPipeline.generateDpm(actualRequest,layout,onProgress=callback,cancelled=context.cancelled,prediction=prediction,
                    source=imageSource,latentSource=latentSource)
                else if passSampler=="Euler a" then
                  if prediction==DiffusionPrediction.Sample then unsupported("Euler a does not support sample prediction")
                  val layout=ancestralSchedule(count,passSchedule)
                  generationPipeline.generateAncestral(actualRequest,layout,onProgress=callback,cancelled=context.cancelled,
                    prediction=prediction,source=imageSource,latentSource=latentSource)
                else if passSampler=="Heun" then
                  generationPipeline.generateHeun(actualRequest,discreteEulerSchedule(count,passSchedule,"Heun"),onProgress=callback,
                    cancelled=context.cancelled,prediction=prediction,source=imageSource,latentSource=latentSource)
                else
                  val layout=eulerSchedule(count,passSchedule)
                  generationPipeline.generate(actualRequest,layout,onProgress=callback,cancelled=context.cancelled,prediction=prediction,
                    source=imageSource,latentSource=latentSource)
              def stepObserverFor(passName: String,stepOffset: Int,passSeed: Long): Option[(Int,Int,scalatorch.Tensor[Float],scalatorch.Tensor[Float],scalatorch.Tensor[Float])=>Unit] =
                intermediateSession.filter(_ => passName=="base" || passName=="hires").map { sess =>
                  (step1Based,_,noisyLat,denoisedLat,nextLat) =>
                    sess.observeStep(passName,step1Based,stepOffset,passSeed,index/batchSize,index%batchSize,noisyLat,denoisedLat,nextLat,
                      lat => DiffusionImageIO.toBufferedImage(generationPipeline.decodeLatents(lat,context.cancelled)),
                      lat => NativePreview.jpegBase64(lat,sdxl=generationPipeline.sdxlConditioning.nonEmpty))
                }
              def basePass(): scalatorch.Tensor[Float]={
                val (reqBase,srcBase)=if altActive then
                  val invNoise=generationPipeline.invertImageForAlt(current,source.get.image,img2imgAltHook.get,activeNoise,steps,prediction,context.cancelled)
                  (current.copy(initialNoiseOverride=Some(invNoise),onStepObservation=stepObserverFor("base",0,current.seed)),None)
                else (current.copy(onStepObservation=stepObserverFor("base",0,current.seed)),source)
                runImagePass(reqBase,if altActive then steps else baseSteps,progress("base"),srcBase,None,sampler,schedule)
              }
              def baseLatentsPass(): scalatorch.Tensor[Float]=
                val actualRequest=if generationPipeline.sdxlConditioning.isEmpty && source.nonEmpty then
                  current.copy(guidanceRescale=0f) else current
                warnUnusedEta(sampler)
                if sampler=="DPM++ 2M SDE" || sampler=="DPM++ 2M SDE Heun" || sampler=="DPM++ 3M SDE" then
                  val layout=dpmSchedule(baseSteps,schedule,sampler)
                  generationPipeline.generateDpmMultistepSde(actualRequest,layout,order=if sampler=="DPM++ 3M SDE" then 3 else 2,
                    heunVariant=(sampler=="DPM++ 2M SDE Heun"),onProgress=progress("base"),cancelled=context.cancelled,
                    prediction=prediction,source=source,returnLatents=true)
                else if sampler=="UniPC" then
                  val layout=dpmSchedule(baseSteps,schedule,"UniPC")
                  generationPipeline.generateUniPc(actualRequest,layout,onProgress=progress("base"),cancelled=context.cancelled,
                    prediction=prediction,source=source,returnLatents=true)
                else if sampler=="LCM" then
                  schedule match
                    case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
                      samplerWarnings += s"schedule '$schedule' is not available for sampler 'LCM'; using Karras"
                      samplerWarnings += s"schedule '$schedule' is not supported by sampler 'LCM'; using the sampler's default spacing"
                    case "Karras" | "Exponential" | "Beta" =>
                      samplerWarnings += s"schedule '$schedule' is not supported by sampler 'LCM'; using the sampler's default spacing"
                    case _ => ()
                  if schedule!="Uniform" then samplerWarnings += "LCM ignores the schedule type"
                  generationPipeline.generateLcm(actualRequest,LcmSchedule(activeNoise,baseSteps),onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="LMS" then
                  generationPipeline.generateLms(actualRequest,discreteEulerSchedule(baseSteps,schedule,"LMS"),onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="DPM2" || sampler=="DPM2 a" then
                  generationPipeline.generateKdpm2(actualRequest,discreteEulerSchedule(baseSteps,schedule,sampler,roundKarrasTimesteps=true),ancestral=(sampler=="DPM2 a"),
                    onProgress=progress("base"),cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="DPM++ 2S a" then
                  generationPipeline.generateDpm2Sa(actualRequest,dpmSinglestepSchedule(baseSteps,schedule),onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="DPM++ SDE" then
                  generationPipeline.generateDpmSde(actualRequest,discreteEulerSchedule(baseSteps,schedule,"DPM++ SDE"),onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="DDIM" then
                  generationPipeline.generateDdim(actualRequest,ddimSchedule(baseSteps,schedule),config.ddimClipSample,config.ddimClipRange,
                    eta=ddimEta,onProgress=progress("base"),cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else if sampler=="DPM++ 2M" then
                  val layout=dpmSchedule(baseSteps,schedule)
                  generationPipeline.generateDpmLatents(actualRequest,layout,onProgress=progress("base"),cancelled=context.cancelled,
                    prediction=prediction,source=source)
                else if sampler=="Euler a" then
                  val layout=ancestralSchedule(baseSteps,schedule)
                  generationPipeline.generateAncestralLatents(actualRequest,layout,onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source)
                else if sampler=="Heun" then
                  generationPipeline.generateHeun(actualRequest,discreteEulerSchedule(baseSteps,schedule,"Heun"),onProgress=progress("base"),
                    cancelled=context.cancelled,prediction=prediction,source=source,returnLatents=true)
                else
                  val layout=eulerSchedule(baseSteps,schedule)
                  generationPipeline.generateLatents(actualRequest,layout,onProgress=progress("base"),cancelled=context.cancelled,
                    prediction=prediction,source=source)
              def firstPass(): scalatorch.Tensor[Float]=hires match
                case None => basePass()
                case Some(settings) =>
                  context.checkCancelled()
                  val (targetWidth,targetHeight)=NativeHires.targetSize(current.width,current.height,settings.scale,
                    settings.resizeX,settings.resizeY)
                  val (upscaleWidth,upscaleHeight)=NativeHires.upscaleSize(current.width,current.height,targetWidth,targetHeight,
                    cover=settings.resizeX>0 && settings.resizeY>0)
                  val vaeScale=1 << (generationPipeline.vae.decoder.upBlocks.size-1)
                  require(targetWidth%8==0 && targetHeight%8==0 && upscaleWidth%8==0 && upscaleHeight%8==0 &&
                    targetWidth%vaeScale==0 && targetHeight%vaeScale==0 && upscaleWidth%vaeScale==0 && upscaleHeight%vaeScale==0,
                    "native Hires requires dimensions aligned to the VAE scale")
                  require(targetWidth.toLong*targetHeight<=16777216L && upscaleWidth.toLong*upscaleHeight<=16777216L &&
                    targetWidth<=16384 && targetHeight<=16384,"native Hires output exceeds allocation limits")
                  val latentScale=if nineChannelModel && NativeHires.LatentModes.contains(settings.upscaler) then 8 else vaeScale
                  val (left,top)=NativeHires.centerCrop(upscaleWidth/latentScale,upscaleHeight/latentScale,
                    targetWidth/latentScale,targetHeight/latentScale)
                  val nominalSteps=if settings.secondPassSteps==0 then steps else settings.secondPassSteps
                  val (scheduleSteps,strength,_)=NativeHires.steps(nominalSteps,settings.denoisingStrength)
                  val (hiresImage,hiresLatent)=if NativeHires.LatentModes.contains(settings.upscaler) then
                    val baseLatents=baseLatentsPass()
                    context.checkCancelled()
                    val resized=NativeHires.resize(baseLatents,upscaleWidth/latentScale,upscaleHeight/latentScale,settings.upscaler)
                    val cropped=NativeHires.crop(resized,targetWidth/latentScale,targetHeight/latentScale,left,top)
                    if nineChannelModel then
                      samplerWarnings += "latent hires upscale is not supported for inpainting models; decoding first"
                      val decoded=DiffusionImageIO.toBufferedImage(generationPipeline.decodeLatents(cropped,context.cancelled))
                      val target=if decoded.getWidth==targetWidth && decoded.getHeight==targetHeight then decoded
                        else NativeImg2ImgHires.upscale(decoded,targetWidth,targetHeight,"Lanczos",context.cancelled)
                      val image=NativeImg2ImgHires.tensor(target,context.cancelled)
                      (Some(fullInpaintSource(image,strength)),None)
                    else (None,Some(DiffusionLatentSource(cropped,strength)))
                  else
                    val baseImage=DiffusionImageIO.toBufferedImage(basePass())
                    val resized=NativeImg2ImgHires.upscale(baseImage,upscaleWidth,upscaleHeight,settings.upscaler,settings.upscalerPath,context.cancelled)
                    // NUI aligns pixel-space crop offsets to eight pixels independently of VAE scale.
                    val pixelLeft=((upscaleWidth-targetWidth)/8/2)*8
                    val pixelTop=((upscaleHeight-targetHeight)/8/2)*8
                    val cropped=resized.getSubimage(pixelLeft,pixelTop,targetWidth,targetHeight)
                    val image=NativeImg2ImgHires.tensor(cropped,context.cancelled)
                    (Some(if nineChannelModel then fullInpaintSource(image,strength)
                      else DiffusionImageSource(image,strength,finalOverlay=false)),None)
                  val (passText,passNegative,passAdapters)=passPrompt(
                    if settings.prompt.nonEmpty then settings.prompt else current.prompt,
                    if settings.negativePrompt.nonEmpty then settings.negativePrompt else current.negativePrompt,hiresAdapters)
                  val hiresRequest=current.copy(prompt=passText,negativePrompt=passNegative,
                    guidanceScale=settings.cfgScale.getOrElse(current.guidanceScale),latentHookSeed=current.latentHookSeed.map(_+1),seed=hiresFixTweaksHook.fold(current.seed)(_.seedFor(current.seed)),variationSeed=hiresFixTweaksHook.fold(current.variationSeed)(p=>Option.when(p.subseedStrength(variationStrength)>0)(p.subseedFor(subseed+index))),variationStrength=hiresFixTweaksHook.fold(current.variationStrength)(_.subseedStrength(variationStrength)),noiseResizeFrom=hiresFixTweaksHook.flatMap(_.resizeFrom),isHiresPass=true,onStepObservation=stepObserverFor("hires",steps,hiresFixTweaksHook.fold(current.seed)(_.seedFor(current.seed))))
                  val secondPass=NativeControlNet.resizeRequest(hiresRequest,targetWidth,targetHeight,context.cancelled)
                  def runHires(): scalatorch.Tensor[Float]={ val repeatCount=hiresFixTweaksHook.fold(1)(_.repeat); var out=runImagePass(secondPass,scheduleSteps,progress("hires"),hiresImage,hiresLatent,hiresSampler,hiresSchedule); var repIdx=1; while repIdx<repeatCount do { val repSeed=secondPass.seed+repIdx; val repSub=secondPass.variationSeed.map(_+repIdx); out=runImagePass(secondPass.copy(seed=repSeed,variationSeed=repSub),scheduleSteps,progress("hires"),hiresImage,hiresLatent,hiresSampler,hiresSchedule); repIdx+=1 }; out }

                  withPassAdapters(passAdapters) { runHires() }
              hiresFix match
                case Some(settings) =>
                  context.checkCancelled()
                  val base=DiffusionImageIO.toBufferedImage(firstPass())
                  val firstImage=preparedInpaint.filter(_.pasteBox.nonEmpty)
                    .map(_.pasteBack(base,context.cancelled)).getOrElse(base)
                  val (targetWidth,targetHeight)=settings.target(firstImage.getWidth,firstImage.getHeight)
                  val resized=NativeImg2ImgHires.upscale(firstImage,targetWidth,targetHeight,settings.upscaler,
                    settings.upscalerPath,context.cancelled)
                  val resizedTensor=NativeImg2ImgHires.tensor(resized,context.cancelled)
                  if settings.denoise==0 then resizedTensor
                  else
                    val passSampler=if settings.sampler=="Use same sampler" then sampler else settings.sampler
                    val baseScheduleRequest=canonicalSchedule(str(g,"scheduler","Automatic"))
                    val requestedSchedule=if settings.scheduler=="Use same scheduler" then
                      (if baseScheduleRequest=="Automatic" then "Automatic" else schedule) else settings.scheduler
                    val passSchedule=if requestedSchedule=="Automatic" then
                      (if passSampler=="DPM++ 2M" then "Karras" else "Uniform") else requestedSchedule
                    if !allSupportedSamplers(passSampler) || !supportedSchedules(passSchedule) then
                      unsupported(s"unsupported img2img Hires sampler/schedule: $passSampler/$passSchedule")
                    val count=if settings.steps==0 then steps else settings.steps
                    val (passText,passNegative,passAdapters)=passPrompt(
                      keywordRefinement.getOrElse(if settings.prompt.isEmpty then current.prompt else settings.prompt),
                      if settings.negativePrompt.isEmpty then current.negativePrompt else settings.negativePrompt,
                      keywordRefinementAdapters.orElse(refinementAdapters))
                    val request=current.copy(prompt=passText,negativePrompt=passNegative,
                      width=targetWidth,height=targetHeight,
                      guidanceScale=(if settings.cfg==0 then current.guidanceScale else settings.cfg.toFloat))
                    val hiresRequest=NativeControlNet.resizeRequest(request,targetWidth,targetHeight,context.cancelled)
                    val passSource=if generationPipeline.unet.convIn.weight.data.shape.dims(1)==
                        2*generationPipeline.vae.config.latentChannels+1 then
                      val mask=scalatorch.Tensor.fromFloats(Array.fill(targetWidth*targetHeight)(1f),
                        scalatorch.Shape.from(Seq(1L,1L,targetHeight.toLong,targetWidth.toLong)))
                      DiffusionImageSource(resizedTensor,settings.denoise,mask=Some(mask),originalImage=Some(resizedTensor),
                        finalOverlay=false,maskedImage=Some(condHighresMode.flatMap(m => ConditioningHighresFix.resolveWeight(m,Some(settings.denoise))).fold(NativeInpaint.maskedImage(resizedTensor,mask,context.cancelled))(w => ConditioningHighresFix.conditionPixels(resizedTensor,mask,w))))
                    else DiffusionImageSource(resizedTensor,settings.denoise,finalOverlay=false)
                    withPassAdapters(passAdapters) {
                      runImagePass(hiresRequest,count,progress("img2img_hires"),Some(passSource),None,passSampler,passSchedule)
                    }
                case None =>
                  val outFirst=firstPass()
                  sonarHook.flatMap(sCfg => Sonar.upscaleTarget(outFirst.shape.dims(3).toInt,outFirst.shape.dims(2).toInt,sCfg).map(t => (sCfg,t))).fold(outFirst) { case (sCfg,(tw,th,cover)) =>
                    val baseBuf=DiffusionImageIO.toBufferedImage(outFirst)
                    val (uw,uh)=NativeHires.upscaleSize(baseBuf.getWidth,baseBuf.getHeight,tw,th,cover)
                    val upPath=upscalerPaths.get(sCfg.upscaler).orElse(upscalerPaths.collectFirst { case (k,v) if k.equalsIgnoreCase(sCfg.upscaler) => v })
                    val resizedBuf=NativeImg2ImgHires.upscale(baseBuf,uw,uh,sCfg.upscaler,upPath,context.cancelled)
                    val croppedBuf=if uw==tw && uh==th then resizedBuf else resizedBuf.getSubimage((uw-tw)/2,(uh-th)/2,tw,th)
                    NativeImg2ImgHires.tensor(croppedBuf,context.cancelled)
                  }
              }
            }
            catch
              case failure: VaeDecodeFailure =>
                context.checkCancelled()
                val artifact=NativeVaeRecovery.save(failure.decoderInput,recovery)
                throw new NativeBackendError("vae_decode_failed",s"VAE decode failed (${Option(failure.getCause).getOrElse(failure).toString.take(300)}); exact decoder input saved to $artifact",failure)
            context.checkCancelled()
            val pastedImage=preparedInpaint.filter(_.pasteBox.nonEmpty).filter(_ => hiresFix.isEmpty)
              .map(_.pasteBack(DiffusionImageIO.toBufferedImage(image),context.cancelled))
            val name=s"native-${UUID.randomUUID()}.png"
            val temporary=staging.resolve(name)
            val hasPostHooks=adetailerUnits.nonEmpty || faceRestoreHook.nonEmpty || nsfwCheckerHook.nonEmpty || aestheticScoreHook.nonEmpty
            val (fullImage,postGenMeta)=if !hasPostHooks then (pastedImage,Map.empty[String,NativeJson]) else
              val baseBuf=pastedImage.getOrElse(DiffusionImageIO.toBufferedImage(image))
              val mHash=modelSummary.get("hash").collect { case Str(h) => h }.getOrElse("")
              val (processedBuf,extraMeta)=NativeBatch18Services.applyPostGenerationHooks(
                baseBuf,adetailerUnits,faceRestoreHook,nsfwCheckerHook,aestheticScoreHook,
                (curImg,maskBuf,unit,pText,nText) => {
                  if unit.denoisingStrength<=0.0 then curImg else
                    val adW=if unit.useInpaintWidthHeight then (unit.inpaintWidth/8)*8 else curImg.getWidth
                    val adH=if unit.useInpaintWidthHeight then (unit.inpaintHeight/8)*8 else curImg.getHeight
                    val adPrep=NativeInpaint.prepare(curImg,maskBuf,adW,adH,0,0,false,context.cancelled,
                      fullRes=unit.inpaintOnlyMasked,fullResPadding=unit.inpaintPadding,fill=1,seed=current.seed)
                    val adSteps=if unit.useSteps then unit.steps else steps
                    val adReq=current.copy(prompt=pText,negativePrompt=nText,width=adPrep.image.shape.dims(3).toInt,height=adPrep.image.shape.dims(2).toInt,
                      guidanceScale=if unit.useCfgScale then unit.cfgScale.toFloat else current.guidanceScale,controlNets=Vector.empty)
                    val adSrc=DiffusionImageSource(adPrep.image,unit.denoisingStrength,mask=Some(adPrep.mask),
                      originalImage=Some(adPrep.originalImage),finalOverlay=adPrep.pasteBox.isEmpty)
                    val adOut=generationPipeline.generate(adReq,eulerSchedule(adSteps,schedule),cancelled=context.cancelled,prediction=prediction,source=Some(adSrc))
                    val adBuf=DiffusionImageIO.toBufferedImage(adOut)
                    if adPrep.pasteBox.nonEmpty then adPrep.pasteBack(adBuf,context.cancelled) else adBuf
                },
                cleanPrompt,cleanNegative,steps,samplerName,number(g,"cfg_scale",7).toDouble,current.seed,mHash,
                staging,output,name,staged
              )
              (Some(processedBuf),extraMeta)
            fullImage match
              case Some(value) => DiffusionImageIO.writePngImage(value,temporary)
              case None => DiffusionImageIO.writePng(image,temporary)
            staged+=temporary
            intermediateSession.foreach(_.recordFinal(fullImage.getOrElse(DiffusionImageIO.toBufferedImage(image)),current.seed,index/batchSize,index%batchSize))
            val resultWidth=fullImage.fold(image.shape.dims(3).toInt)(_.getWidth)
            val resultHeight=fullImage.fold(image.shape.dims(2).toInt)(_.getHeight)
            val hardwareRecord=hardware.map { remark =>
              val now=System.nanoTime()
              val result=NativeHardwareInfo.record(hardwareProfile.get,remark,(now-hardwareStarted)/1e9)
              hardwareStarted=now
              "hardware_info"->result
            }.toMap
            Obj(Map("path"->Str(output.resolve(name).toString),"seed"->Number(BigDecimal(current.seed)),
              "subseed"->Number(BigDecimal(subseed+index)),
              "width"->Number(resultWidth),"height"->Number(resultHeight))++hardwareRecord++postGenMeta++(if hires.nonEmpty then hiresFixTweaksHook.fold(Map.empty[String,NativeJson])(p=>Map("hires_seed"->Number(BigDecimal(p.seedFor(current.seed))),"hires_subseed"->Number(BigDecimal(p.subseedFor(subseed+index))),"hires_subseed_strength"->Number(BigDecimal(p.subseedStrength(variationStrength))),"hires_repeat"->Number(p.repeat),"hires_seed_resize_from_w"->Number(p.seedResizeFromW),"hires_seed_resize_from_h"->Number(p.seedResizeFromH))) else Map.empty)++keywordPlan.fold(Map.empty[String,NativeJson]) { _ =>
                Map("effective_prompt"->Str(NativeLoraPrompt.replay(batchPrompt,cleanPrompt)),
                  "model_keyword_source_prompt"->Str(rawPrompt))++
                  keywordRefinement.map(value => "model_keyword_refinement_prompt"->Str(value))
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
            "model_name"->Str(root.getFileName.toString),"scheduler"->Str(schedule),
            "warnings"->Arr((embeddingWarnings++samplerWarnings.toVector).map(Str.apply)),
            "intermediate_artifacts"->Arr(intermediateSession.map(_.finish()).getOrElse(Vector.empty)))++
            Option.when(systemInfoFields.nonEmpty) {
              val samplerInfo=executeNativeRequest(NativeRequest(request.id,"list_samplers",Map.empty,request.owner),context)
              val fullInfo=NativeSystemInfo.execute(Map.empty,systemStatus,features,samplerInfo,context)
              "system_info"->Obj(NativeSystemInfo.metadata(fullInfo,systemInfoFields))
            }
        catch
          case error: Throwable =>
            intermediateSession.foreach(_.closeIncomplete())
            (staged.toVector++published.toVector :+ staging).foreach { path =>
              try Files.deleteIfExists(path) catch case cleanup: Throwable => error.addSuppressed(cleanup)
            }
            throw error
      case _ => unsupported(s"unsupported request: ${request.kind}")

object NativeWorkerMain:
  private val DefaultMaxAttentionScoreElements=134217728L
  private[models] def attentionScoreLimit(configured: Option[String],cudaUnet: Boolean): Long =
    // CUDA attention slices its score scratch to a fixed byte budget; keep a high-resolution total-work cap.
    val cap=if cudaUnet then 1L<<36 else Int.MaxValue.toLong
    val limit=configured.fold(if cudaUnet then cap else DefaultMaxAttentionScoreElements)(_.toLong)
    require(limit>=0 && limit<=cap,"invalid attention work limit")
    limit
  private[models] def vaeAttentionLimit(configured: Long,explicit: Boolean,cuda: Boolean): Long =
    val c=math.min(configured,Int.MaxValue.toLong)
    if cuda && !explicit then math.max(c,268435456L) else c
  private[worker] def schedulerFor(initial: Path,configured: Path,selected: Path): Path =
    val default=initial.resolve("scheduler/scheduler_config.json").toAbsolutePath.normalize()
    val automatic=configured.toString=="-" || configured.toAbsolutePath.normalize()==default
    if !automatic then configured
    else if Files.isRegularFile(selected) then Path.of("-")
    else selected.resolve("scheduler/scheduler_config.json")
  def main(args: Array[String]): Unit=
    require(args.length==2,"usage: NativeWorkerMain <diffusers-root> <explicit-euler-config.json>")
    val initial=Path.of(args(0)).toRealPath()
    val config=Path.of(args(1))
    val environment=CudaDeviceMemoryBudget.withAutomaticBudgets(sys.env,Seq(
      "SDFX_SCALATORCH_VAE_DEVICE"->"SDFX_SCALATORCH_VAE_MAX_BYTES",
      "SDFX_SCALATORCH_UNET_DEVICE"->"SDFX_SCALATORCH_UNET_MAX_BYTES"))
    val vae=VaeExecution.fromEnvironment(environment)
    val unet=UNetExecution.fromEnvironment(environment)
    val configured=environment.get("SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS")
    val maxAttention=attentionScoreLimit(configured,unet.device!="cpu")
    val maxVaeAttention=vaeAttentionLimit(maxAttention,configured.nonEmpty,vae.device!="cpu")
    val backend=new NativeSwitchingBackend(initial,path =>
      new NativeDiffusionBackend(path,schedulerFor(initial,config,path),vae,unet,maxAttention,maxVaeAttention))
    val worker=new NativeWorker(System.out,backend)
    worker.serve(new java.io.InputStreamReader(System.in,java.nio.charset.StandardCharsets.UTF_8))
