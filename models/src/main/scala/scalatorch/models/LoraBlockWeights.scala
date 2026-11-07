package scalatorch.models

/** NUI's LoRA Block Weight layouts and built-in/user preset expansion. */
object LoraBlockWeights:
  private val sd15short=Vector("BASE","IN01","IN02","IN04","IN05","IN07","IN08","MID",
    "OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11")
  private val sdxlshort=Vector("BASE","IN04","IN05","IN07","IN08","MID","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05")
  private def ids(prefix: String,n: Int)=Vector.tabulate(n)(i=>f"$prefix$i%02d")
  private val sd15full=Vector("BASE")++ids("IN",12)++Vector("MID")++ids("OUT",12)
  private val sxlfull=Vector("BASE")++ids("IN",9)++Vector("MID")++ids("OUT",9)
  private val all=(sd15full++sxlfull).toSet
  private val presets: Map[String,Set[String]]=Map(
    "NONE"->Set.empty,"NOT"->Set.empty,"ALL"->all,
    "INS"->(Set("BASE")++ids("IN",4)),"IND"->(Set("BASE")++ids("IN",9).drop(4)),
    "INALL"->(Set("BASE")++ids("IN",12)),
    "MIDD"->(Set("BASE","MID")++ids("IN",12).drop(4)++ids("OUT",6)),
    "OUTD"->(Set("BASE")++ids("OUT",6)),"OUTS"->(Set("BASE")++ids("OUT",12).drop(6)),
    "OUTALL"->(Set("BASE")++ids("OUT",12)))

  def parse(spec: String,sdxl: Boolean,userPresets: Map[String,String]=Map.empty): Map[String,Double]=
    val user=userPresets.iterator.map((k,v)=>k.trim.toUpperCase(java.util.Locale.ROOT)->v).toMap
    val raw=user.getOrElse(spec.trim.toUpperCase(java.util.Locale.ROOT),spec).trim
    val key=raw.toUpperCase(java.util.Locale.ROOT)
    val preset=if key=="ALL0.5" then Some((presets("ALL"),.5)) else presets.get(key).map(_->1.0)
    preset match
      case Some((selected,scale)) => (if sdxl then sxlfull else sd15full).map(name=>name->(if selected(name) then scale else 0.0)).toMap
      case None =>
        val values=raw.replace(" ","").split(",",-1).toVector.filter(_.nonEmpty).map { value =>
          val n=value.toDoubleOption.getOrElse(throw IllegalArgumentException(s"invalid LoRA block weight: $value"))
          require(n.isFinite,"LoRA block weights must be finite")
          n
        }
        val names=if sdxl then if values.size==20 then sxlfull else sdxlshort
          else if values.size==26 then sd15full else sd15short
        names.zip(values.padTo(names.size,0.0).take(names.size)).toMap

  final case class Plan(text: Double,down: Vector[Vector[Double]],mid: Double,up: Vector[Vector[Double]])
  /** Resolve Diffusers' expanded attention keys and its per-module fallback.
    * None means Diffusers' fallback cannot index this short module path.
    */
  def targetScale(plan: Plan,sdxl: Boolean,target: String): Option[Double] =
    if target.startsWith("text_encoder.") || target.startsWith("text_encoder_2.") then Some(plan.text)
    else if !target.startsWith("unet.") then None
    else
      val name=target.stripPrefix("unet.")
      val downIndices=if sdxl then Vector(1,2) else Vector(0,1,2)
      val upIndices=if sdxl then Vector(0,1) else Vector(1,2,3)
      def keys(prefix: String,indices: Vector[Int],groups: Vector[Vector[Double]],count: Int) =
        indices.flatMap { block =>
          val values=groups(block)
          Vector.tabulate(count) { i =>
            val value=if values.size==1 then values.head else values(i)
            s"${prefix}_blocks.$block.attentions.$i"->value
          }
        }
      val expanded=Vector("mid_block.attentions.0"->plan.mid)++keys("down",downIndices,plan.down,2)++keys("up",upIndices,plan.up,3)
      expanded.collectFirst { case (key,value) if name.contains(key) => value }.orElse {
        val parts=name.split("\\.")
        if parts.length<4 then None
        else
          val key=s"${parts(0)}.${parts(1)}.attentions.${parts(3)}"
          Some(expanded.find(_._1==key).fold(1.0)(_._2))
      }
  def plan(weights: Map[String,Double],sdxl: Boolean,unet: Double=1.0,text: Double=1.0): Plan =
    require(unet.isFinite && text.isFinite && weights.values.forall(_.isFinite),"LoRA block scales must be finite")
    def get(key: String)=weights.getOrElse(key,0.0)*unet
    def mean(keys: Vector[String]) =
      val values=keys.flatMap(weights.get)
      if values.isEmpty then 0.0 else values.sum/values.size*unet
    if sdxl then Plan(weights.getOrElse("BASE",0.0)*text,
      Vector(Vector(mean(ids("IN",4))),Vector(get("IN04"),get("IN05")),Vector(get("IN07"),get("IN08"))),get("MID"),
      Vector(Vector(get("OUT00"),get("OUT01"),get("OUT02")),Vector(get("OUT03"),get("OUT04"),get("OUT05")),
        Vector(mean(Vector("OUT06","OUT07","OUT08")))))
    else Plan(weights.getOrElse("BASE",0.0)*text,
      Vector(Vector(get("IN01"),get("IN02")),Vector(get("IN04"),get("IN05")),Vector(get("IN07"),get("IN08")),
        Vector(mean(Vector("IN10","IN11")))),get("MID"),
      Vector(Vector(mean(Vector("OUT00","OUT01","OUT02"))),Vector(get("OUT03"),get("OUT04"),get("OUT05")),
        Vector(get("OUT06"),get("OUT07"),get("OUT08")),Vector(get("OUT09"),get("OUT10"),get("OUT11"))))
