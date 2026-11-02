package scalatorch.models.worker

/** Validate inactive NUI options before removing them from the core generation request. */
private[worker] object NativeGenerationDefaults:
  import NativeJson.*
  private val hires=Obj(Map("enabled"->Bool(false),"scale"->Number(2),"upscaler"->Str("Latent"),
    "second_pass_steps"->Number(0),"resize_x"->Number(0),"resize_y"->Number(0),
    "denoising_strength"->Number(BigDecimal("0.55")),"sampler_name"->Null,"scheduler"->Null,
    "prompt"->Str(""),"negative_prompt"->Str(""),"cfg_scale"->Null))
  val generation: Map[String,NativeJson]=Map("subseed_strength"->Number(0),
    "batch_size"->Number(1),"eta"->Null,"hires"->hires,
    "img2img"->Null,"loras"->Arr(Vector.empty),"controlnet"->Arr(Vector.empty),"hooks"->Obj(Map.empty),
    "styles"->Arr(Vector.empty),"extra"->Obj(Map.empty))
  val envelope: Map[String,NativeJson]=Map("lora_paths"->Obj(Map.empty),"embedding_paths"->Arr(Vector.empty),
    "controlnet_paths"->Obj(Map.empty),"upscaler_paths"->Obj(Map.empty),"lbw_presets"->Obj(Map.empty),
    "system_info_fields"->Arr(Vector.empty),
    "hardware_info"->Obj(Map("enabled"->Bool(false),"remark"->Str(""))),
    "hires_loras"->Null,"manual_loras"->Null,"img2img_hires_loras"->Null)
  private def matches(value: NativeJson,default: NativeJson): Boolean=(value,default) match
    case (Obj(actual),Obj(expected)) => actual.forall((key,v) => expected.get(key).exists(matches(v,_)))
    case _ => value==default
  def strip(fields: Map[String,NativeJson],defaults: Map[String,NativeJson]): Map[String,NativeJson]=
    defaults.foreach { (key,value) =>
      fields.get(key).foreach { actual =>
        val emptyLoras=Set("manual_loras","hires_loras","img2img_hires_loras").contains(key) && actual==Arr(Vector.empty)
        if !emptyLoras && !matches(actual,value) then throw new NativeBackendError("unsupported",s"unsupported non-default option: $key")
      }
    }
    fields--defaults.keySet
