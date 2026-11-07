package scalatorch.models.worker

import java.nio.file.Path
import scalatorch.models.{KeywordMappings,ModelKeyword}

private[worker] object NativeKeywordHook:
  import NativeJson.*
  private def enabled(value: NativeJson): Boolean=value match
    case Null => false
    case Bool(flag) => flag
    case Number(n) => n!=0
    case Str(text) => text.trim.toLowerCase(java.util.Locale.ROOT) match
      case "" | "false" | "off" | "no" | "0" | "disable" | "disabled" => false
      case "true" | "on" | "yes" | "1" | "enable" | "enabled" => true
      case _ => throw IllegalArgumentException("invalid hook enable flag")
    case Obj(fields) => enabled(fields.getOrElse("enabled",Bool(true)))
    case _ => throw IllegalArgumentException("invalid hook configuration")

  def plan(hooks: Option[NativeJson],prompt: String,hash: String,name: String,count: Int,seed: Long,
      state: ModelKeyword.State,check: () => Unit,allowActive: Set[String]=Set.empty): Option[ModelKeyword.Plan]=
    val fields=hooks match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(values)) => values
      case _ => throw IllegalArgumentException("hooks must be an object")
    val active=fields.filter((_,value) => enabled(value))
    val unknown=(active.keySet-"model_keyword")--allowActive
    if unknown.nonEmpty then throw new NativeBackendError("unsupported",s"unsupported hooks: ${unknown.toVector.sorted.mkString(", ")}")
    active.get("model_keyword").map { value =>
      val config=value match
        case Obj(values) => values
        case _ => Map.empty[String,NativeJson]
      val allowed=Set("enabled","placement","model_choice","embedding","order","lora","lora_weight",
        "multiplier","lora_choice","lora_name","_base_prompt","_mapping_dir","_lora_path")
      require((config.keySet--allowed).isEmpty,"unknown model_keyword option")
      def text(key: String,default: String=""): String=config.get(key) match
        case None | Some(Null) => default
        case Some(Str(s)) => s
        case _ => throw IllegalArgumentException(s"model_keyword $key must be a string")
      def selected(key: String): Option[String]=Option(text(key)).filter(value => value.nonEmpty && value!="None")
      val loraPath=selected("_lora_path").map(path => Path.of(path).toAbsolutePath.normalize())
      require(selected("lora").isEmpty || loraPath.nonEmpty,"selected model_keyword LoRA requires a resolved path")
      val weight=config.get("multiplier").orElse(config.get("lora_weight")) match
        case None => 0.7
        case Some(Number(n)) => n.toDouble
        case _ => throw IllegalArgumentException("model_keyword multiplier must be numeric")
      val directory=text("_mapping_dir")
      require(directory.nonEmpty,"model_keyword _mapping_dir is required")
      // A Diffusers directory has no A1111 legacy file hash. Native catalogs use
      // the full modelSummary SHA256, including relative source names and bytes.
      require(hash.matches("[0-9a-f]{64}"),"invalid native checkpoint identity")
      val root=Path.of(directory).toAbsolutePath.normalize()
      val mappings=new KeywordMappings(root.resolve("model-keyword.txt"),root.resolve("custom-mappings.txt"),allowMissingDefault=true)
      val entry=mappings.lookup(hash,name,check)
      val loraName=loraPath.map(path => selected("lora_name").getOrElse(path.getFileName.toString))
      val loraWords=loraPath.flatMap { path =>
        val loraHash=KeywordMappings.legacyHash(path,check)
        new KeywordMappings(root.resolve("lora-keyword.txt"),root.resolve("lora-keyword-user.txt"),allowMissingDefault=true)
          .lookup(loraHash,loraName.get,check).map(_.keywords)
      }
      val words=loraWords.toVector.flatMap(_.split("\\|",-1).toVector.map(_.trim))
      val choices=Vector("None")++(if words.size>1 then Vector(words.mkString(", ")) else Vector.empty)++words++
        (if words.size>1 then Vector("< iterate >","< random >") else Vector.empty)
      val requestedChoice=text("lora_choice","None")
      val loraChoice=if loraPath.nonEmpty && choices.contains(requestedChoice) then requestedChoice else "None"
      val result=ModelKeyword.plan(text("_base_prompt",prompt),Vector.fill(count)(text("_base_prompt",prompt)),
        entry.map(_.keywords),loraWords,ModelKeyword.Options(modelChoice=text("model_choice","keyword1, keyword2"),
          loraChoice=loraChoice,placement=text("placement","keyword prompt"),
          order=text("order","textual inversion first"),embedding=selected("embedding"),loraModel=loraName,multiplier=weight),state,seed)
      check()
      result
    }
