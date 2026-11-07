package scalatorch.models.worker

private[worker] object NativeLoraPrompt:
  private val directive="""<(\w+):([^>]+)>""".r
  def replay(batchFirst: String,cleanPrompt: String): String =
    directive.findAllIn(batchFirst).mkString+cleanPrompt
  def parse(prompt: String): (String,Vector[(String,Float,Float,Option[Int],Option[Double],Option[Double],Option[String])]) =
    val adapters=Vector.newBuilder[(String,Float,Float,Option[Int],Option[Double],Option[Double],Option[String])]
    val cleaned=directive.replaceAllIn(prompt, m => {
      val kind=m.group(1)
      if kind!="lora" && kind!="lyco" then
        throw new NativeBackendError("unsupported",s"unsupported extra network type: $kind")
      val parts=m.group(2).split(":",-1).toVector.map(_.trim)
      val positional=parts.filterNot(_.contains("="))
      val named=parts.filter(_.contains("=")).map { item =>
        val pair=item.split("=",2)
        pair(0).trim.toLowerCase(java.util.Locale.ROOT)->pair(1).trim
      }.toMap
      require(positional.nonEmpty && positional.head.nonEmpty,"LoRA activation requires a name")
      if positional.size>4 || named.keys.exists(k => !Set("te","unet","dyn","start","stop","step","lbw")(k)) then
        throw new NativeBackendError("unsupported","LoRA block weights or tag options are not implemented")
      def number(value: String): Float =
        val n=value.toFloat
        require(n.isFinite,"LoRA weights must be finite")
        n
      val text=number(named.getOrElse("te",positional.lift(1).getOrElse("1")))
      val unet=number(named.getOrElse("unet",positional.lift(2).getOrElse(text.toString)))
      val rank=named.get("dyn").orElse(positional.lift(3)).map { value =>
        value.toIntOption.getOrElse(throw IllegalArgumentException(s"invalid LoRA dynamic rank: $value"))
      }
      def bound(key: String): Option[Double]=named.get(key).map { value =>
        val result=value.toDouble
        require(result.isFinite,"LoRA step bounds must be finite")
        result
      }
      val step=named.get("step").map(_.split("-")) match
        case Some(Array(a,b)) => (a.trim.toDoubleOption,b.trim.toDoubleOption)
        case _ => (None,None)
      require((step._1.toVector++step._2.toVector).forall(_.isFinite),"LoRA step bounds must be finite")
      adapters += ((positional.head,text,unet,rank,bound("start").orElse(step._1),bound("stop").orElse(step._2),named.get("lbw").filter(_.nonEmpty)))
      ""
    })
    (cleaned,adapters.result())
