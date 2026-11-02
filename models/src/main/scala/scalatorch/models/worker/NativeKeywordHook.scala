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
      state: ModelKeyword.State,check: () => Unit): Option[ModelKeyword.Plan]=
    val fields=hooks match
      case None => Map.empty[String,NativeJson]
      case Some(Obj(values)) => values
      case _ => throw IllegalArgumentException("hooks must be an object")
    val active=fields.filter((_,value) => enabled(value))
    val unknown=active.keySet-"model_keyword"
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
      for key <- Vector("embedding","lora","_lora_path","lora_name") do
        val selected=text(key)
        if selected.nonEmpty && selected!="None" then
          throw new NativeBackendError("unsupported",s"model_keyword $key requires native network inference")
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
      val result=ModelKeyword.plan(text("_base_prompt",prompt),Vector.fill(count)(text("_base_prompt",prompt)),
        entry.map(_.keywords),None,ModelKeyword.Options(modelChoice=text("model_choice","keyword1, keyword2"),
          loraChoice=text("lora_choice","None"),placement=text("placement","keyword prompt"),
          order=text("order","textual inversion first"),multiplier=weight),state,seed)
      if (result.preview+:result.prompts).exists(s => "<\\w+:[^>]+>".r.findFirstIn(s).nonEmpty) then
        throw new NativeBackendError("unsupported","model_keyword extra network tags require native network inference")
      check()
      result
    }
