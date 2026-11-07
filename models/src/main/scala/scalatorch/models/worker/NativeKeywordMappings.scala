package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.models.{KeywordMappings,KeywordMappingEditor,ModelKeyword}

private[worker] object NativeKeywordMappings:
  import NativeJson.*
  def execute(params: Map[String,NativeJson],modelRoot: Path,directoryHash: () => String,
      context: NativeWorkContext): Map[String,NativeJson]=
    val allowed=Set("action","kind","model_path","mapping_dir","keywords","match_name","confirmed")
    require((params.keySet--allowed).isEmpty,"unknown model_keyword parameter")
    def text(key: String,default: String=""): String=params.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val action=text("action","check")
    if action=="install_upstream_lora_catalog" then
      require((params.keySet--Set("action","mapping_dir","confirmed")).isEmpty,"unknown catalog installation parameter")
      val directory=text("mapping_dir")
      require(directory.nonEmpty,"mapping_dir is required")
      val confirmed=params.get("confirmed").contains(Bool(true))
      return Map("path"->Str(NativeKeywordCatalog.download(Path.of(directory),confirmed,context).toString))
    if !Set("check","save","delete").contains(action) then
      throw new NativeBackendError("unsupported",s"unsupported model_keyword action: $action")
    val kind=text("kind")
    require(Set("checkpoint","lora").contains(kind),"model_keyword kind must be checkpoint or lora")
    val model=text("model_path")
    val directory=text("mapping_dir")
    require(model.nonEmpty && directory.nonEmpty,"model_path and mapping_dir are required")
    val path=Path.of(model).toRealPath()
    val name=text("match_name",path.getFileName.toString)
    require(name.nonEmpty,"match_name must be nonempty")
    context.checkCancelled()
    val hash=if Files.isDirectory(path) then
      require(kind=="checkpoint" && path==modelRoot.toRealPath(),"only the configured Diffusers directory is supported")
      directoryHash()
    else KeywordMappings.legacyHash(path,() => context.checkCancelled())
    val root=Path.of(directory).toAbsolutePath.normalize()
    val (defaultName,userName)=if kind=="checkpoint" then ("model-keyword.txt","custom-mappings.txt")
      else ("lora-keyword.txt","lora-keyword-user.txt")
    val defaultPath=root.resolve(defaultName)
    val userPath=root.resolve(userName)
    val deleted=action match
      case "save" =>
        KeywordMappingEditor.edit(userPath,hash,name,Some(text("keywords")),
          () => context.checkCancelled(),() => context.beginPublication())
        None
      case "delete" =>
        Some(KeywordMappingEditor.edit(userPath,hash,name,None,
          () => context.checkCancelled(),() => context.beginPublication()))
      case _ => None
    // Publication has already become irrevocable for save/delete. Do not treat
    // a late cancellation as if the successful disk write had been rolled back.
    val check: () => Unit=if action=="check" then () => context.checkCancelled() else () => ()
    val found=new KeywordMappings(defaultPath,userPath,allowMissingDefault=true).lookup(hash,name,check)
    val words=found.map(_.keywords.split("\\|",-1).toVector.map(_.strip())).getOrElse(Vector.empty)
    val choices=if kind=="checkpoint" then
      Vector("keyword1, keyword2","random","iterate")++(if words.nonEmpty then words else Vector("keyword1","keyword2"))
    else Vector("None")++(if words.size>1 then Vector(words.mkString(", ")) else Vector.empty)++words++
      (if words.size>1 then Vector("< iterate >","< random >") else Vector.empty)
    if action=="check" then context.beginPublication()
    Map("filename"->Str(name),"hash"->Str(hash),"keywords"->Arr(words.map(Str(_))),
      "source"->Str(found.fold("no match")(e => if e.source==1 then "custom" else if kind=="lora" then "user catalog" else "default")),
      "choices"->Arr(choices.map(Str(_))),"default_catalog_available"->Bool(Files.isRegularFile(defaultPath)))++
      deleted.map(value => "deleted"->Bool(value))

  def planRequest(params: Map[String,NativeJson],modelRoot: Path,directoryHash: () => String,
      state: ModelKeyword.State,context: NativeWorkContext): (Map[String,NativeJson],ModelKeyword.State)=
    val allowed=Set("checkpoint","mapping_dir","lora","preview","prompts","batch_size","choice_seed","options")
    require((params.keySet--allowed).isEmpty,"unknown model_keyword_plan parameter")
    context.checkCancelled()
    val checkpoint=params.get("checkpoint") match
      case Some(Str(s)) if s.nonEmpty => Path.of(s).toRealPath()
      case _ => throw IllegalArgumentException("checkpoint is required")
    val directory=params.get("mapping_dir") match
      case Some(Str(s)) if s.nonEmpty => Path.of(s).toAbsolutePath.normalize()
      case _ => throw IllegalArgumentException("mapping_dir is required")
    val preview=params.get("preview") match
      case None | Some(Null) => ""
      case Some(Str(s)) => s
      case _ => throw IllegalArgumentException("preview must be a string")
    val prompts=params.get("prompts") match
      case Some(Arr(values)) if values.nonEmpty => values.map {
        case Str(s) => s
        case _ => throw IllegalArgumentException("prompts must contain strings")
      }
      case _ => throw IllegalArgumentException("prompts must be a nonempty array")
    val batchSize=params.get("batch_size") match
      case None | Some(Null) => 1
      case Some(Number(n)) if n.isValidInt => n.toInt
      case _ => throw IllegalArgumentException("batch_size must be an integer")
    require(batchSize>=1 && batchSize<=16 && prompts.size%batchSize==0 && (prompts.size/batchSize)<=100,
      "batch size must divide the number of prompts and be 1..16")
    val seed=params.get("choice_seed") match
      case None | Some(Null) => 42L
      case Some(Number(n)) if n.isValidLong => n.toLong
      case _ => throw IllegalArgumentException("choice_seed must be an integer")
    val loraPath=params.get("lora") match
      case None | Some(Null) | Some(Str("")) => None
      case Some(Str(s)) => Some(Path.of(s).toRealPath())
      case _ => throw IllegalArgumentException("LoRA path must be a string")
    val options=params.get("options") match
      case None | Some(Null) => Map.empty[String,NativeJson]
      case Some(Obj(fields)) => fields
      case _ => throw IllegalArgumentException("options must be an object")
    def optStr(key: String,default: String=""): String=options.get(key) match
      case None | Some(Null) => default
      case Some(Str(s)) => s
      case _ => throw IllegalArgumentException(s"options.$key must be a string")
    def optSelected(key: String): Option[String]=Option(optStr(key)).filter(v => v.nonEmpty && v!="None")
    val multiplier=options.get("multiplier").orElse(options.get("lora_weight")) match
      case None | Some(Null) => 0.7
      case Some(Number(n)) if n.toDouble.isFinite => n.toDouble
      case _ => throw IllegalArgumentException("multiplier must be finite")
    val ckptHash=if Files.isDirectory(checkpoint) then
      require(checkpoint==modelRoot.toRealPath(),"only the configured Diffusers directory is supported")
      directoryHash()
    else KeywordMappings.legacyHash(checkpoint,() => context.checkCancelled())
    val ckptIndex=new KeywordMappings(directory.resolve("model-keyword.txt"),directory.resolve("custom-mappings.txt"),allowMissingDefault=true)
    val modelEntry=ckptIndex.lookup(ckptHash,checkpoint.getFileName.toString,() => context.checkCancelled())
    val loraName=loraPath.map(p => optSelected("lora_name").getOrElse(p.getFileName.toString))
    val loraHash=loraPath.map(p => KeywordMappings.legacyHash(p,() => context.checkCancelled()))
    val loraDefaultPath=directory.resolve("lora-keyword.txt")
    val loraIndex=new KeywordMappings(loraDefaultPath,directory.resolve("lora-keyword-user.txt"),allowMissingDefault=true)
    val loraEntry=for (p,h,n) <- loraPath.zip(loraHash).zip(loraName).map { case ((a,b),c) => (a,b,c) } yield
      loraIndex.lookup(h,n,() => context.checkCancelled())
    val loraWords=loraEntry.flatten.map(_.keywords)
    val words=loraWords.toVector.flatMap(_.split("\\|",-1).toVector.map(_.trim))
    val choices=Vector("None")++(if words.size>1 then Vector(words.mkString(", ")) else Vector.empty)++words++
      (if words.size>1 then Vector("< iterate >","< random >") else Vector.empty)
    val reqLoraChoice=optStr("lora_choice","None")
    val loraChoice=if loraPath.nonEmpty && choices.contains(reqLoraChoice) then reqLoraChoice else "None"
    val planned=ModelKeyword.plan(preview,prompts,modelEntry.map(_.keywords),loraWords,
      ModelKeyword.Options(modelChoice=optStr("model_choice","keyword1, keyword2"),loraChoice=loraChoice,
        placement=optStr("placement","keyword prompt"),order=optStr("order","textual inversion first"),
        embedding=optSelected("embedding"),loraModel=loraName,multiplier=multiplier),state,seed)
    context.checkCancelled()
    val directive="""<(\w+):([^>]+)>""".r
    val batches=(0 until planned.prompts.size by batchSize).toVector.map { start =>
      val slice=planned.prompts.slice(start,start+batchSize)
      val cleanPrompts=slice.map(p => directive.replaceAllIn(p,""))
      val firstNetworks=scala.collection.mutable.LinkedHashMap.empty[String,Vector[Vector[String]]]
      val adapters=Vector.newBuilder[NativeJson]
      for m <- directive.findAllMatchIn(slice.head) do
        val kind=m.group(1)
        require(kind=="lora" || kind=="lyco",s"unsupported extra network types: $kind")
        val args=m.group(2).split(":",-1).toVector
        firstNetworks.updateWith(kind)(prev => Some(prev.getOrElse(Vector.empty) :+ args))
      val (_,parsedAdapters)=NativeLoraPrompt.parse(slice.head)
      val kinds=directive.findAllMatchIn(slice.head).map(_.group(1)).toVector
      for ((name,te,unet,dyn,st,sp,lbw),kind) <- parsedAdapters.zip(kinds) do
        val base=Map[String,NativeJson](
          "kind"->Str(kind),
          "name"->Str(name),
          "te"->Number(BigDecimal(te.toDouble)),
          "unet"->Number(BigDecimal(unet.toDouble)),
          "dyn"->dyn.map(d => Number(BigDecimal(d))).getOrElse(Null)
        )++lbw.map("lbw"->Str(_))++st.map(v => "start"->Number(BigDecimal(v)))++sp.map(v => "stop"->Number(BigDecimal(v)))
        adapters += Obj(base)
      Obj(Map(
        "clean_prompts"->Arr(cleanPrompts.map(Str.apply)),
        "networks"->Obj(firstNetworks.iterator.map((k,v) => k->Arr(v.map(a => Arr(a.map(Str.apply))))).toMap),
        "adapters"->Arr(adapters.result())
      ))
    }
    val ckptInfo=Obj(Map(
      "filename"->Str(checkpoint.getFileName.toString),
      "legacy_hash"->Str(ckptHash),
      "mapping_source"->Str(modelEntry.fold("no match")(e => if e.source==1 then "custom" else "default"))
    ))
    val loraInfo=loraPath match
      case None => Null
      case Some(_) =>
        val found=loraEntry.flatten
        Obj(Map(
          "filename"->Str(loraName.get),
          "legacy_hash"->Str(loraHash.get),
          "mapping_source"->Str(found.fold("no match")(e => if e.source==1 then "custom" else "user catalog"))
        ))
    context.beginPublication()
    val res=Map(
      "preview"->Str(planned.preview),
      "prompts"->Arr(planned.prompts.map(Str.apply)),
      "checkpoint"->ckptInfo,
      "lora"->loraInfo,
      "batches"->Arr(batches),
      "lora_catalog_available"->Bool(Files.isRegularFile(loraDefaultPath))
    )
    (res,planned.nextState)
