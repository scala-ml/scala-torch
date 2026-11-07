package scalatorch.models.worker

/** Schedule slicing for the source-resolution img2img latent upscale workflow. */
private[worker] object NativeLatentUpscale:
  final case class Settings(method: String,scheduler: String)
  def parse(value: NativeJson): Settings =
    import NativeJson.*
    val fields=value match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("latent upscale settings must be an object")
    require((fields.keySet--Set("enabled","method","scheduler")).isEmpty,"unknown latent upscale option")
    def text(key: String,default: String): String=fields.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"latent upscale $key must be a string")
    val method=text("method","nearest-exact")
    val scheduler=text("scheduler","simple")
    require(Set("bilinear","bilinear-antialiased","bicubic","bicubic-antialiased","area","nearest","nearest-exact")(method),
      "unsupported latent upscale interpolation method")
    require(Set("simple","normal","karras","exponential","polyexponential","automatic","uniform","sgm_uniform","kl_optimal","align_your_steps","ddim","beta")(scheduler),
      "unknown latent upscale scheduler")
    Settings(method,scheduler)

  def mode(method: String): String=method match
    case "bilinear" => "Latent"
    case "bilinear-antialiased" => "Latent (antialiased)"
    case "bicubic" => "Latent (bicubic)"
    case "bicubic-antialiased" => "Latent (bicubic antialiased)"
    case "nearest" => "Latent (nearest)"
    case "nearest-exact" => "Latent (nearest-exact)"
    case "area" => "area"
    case _ => throw IllegalArgumentException("unsupported latent upscale interpolation method")
  def fullSteps(requested: Int,strength: Double): (Int,Double,Int) =
    require(strength.isFinite && strength>0 && strength<=1,"latent upscale strength must be in (0,1]")
    val steps=math.max(1,requested)
    require(strength>=steps/10000.0,"latent upscale schedule exceeds 10000 steps")
    val count=math.max(steps,math.ceil(steps/strength-1e-9).toInt)
    require(count<=10000,"latent upscale schedule exceeds 10000 steps")
    (count,math.min(1.0,(steps+1e-6)/count),steps)
