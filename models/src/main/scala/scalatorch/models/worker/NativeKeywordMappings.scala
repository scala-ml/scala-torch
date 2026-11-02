package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.models.{KeywordMappings,KeywordMappingEditor}

private[worker] object NativeKeywordMappings:
  import NativeJson.*
  def execute(params: Map[String,NativeJson],modelRoot: Path,directoryHash: () => String,
      context: NativeWorkContext): Map[String,NativeJson]=
    val allowed=Set("action","kind","model_path","mapping_dir","keywords","match_name")
    require((params.keySet--allowed).isEmpty,"unknown model_keyword parameter")
    def text(key: String,default: String=""): String=params.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    val action=text("action","check")
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
