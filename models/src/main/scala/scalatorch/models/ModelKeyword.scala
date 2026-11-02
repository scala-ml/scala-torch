package scalatorch.models

/** Pure NUI keyword planning. Commit nextState only after the caller's transaction succeeds. */
object ModelKeyword:
  final case class State(model: BigInt=0,lora: BigInt=0):
    require(model>=0 && lora>=0,"keyword iteration counters must be nonnegative")
  final case class Options(modelChoice: String="keyword1, keyword2",loraChoice: String="None",
      placement: String="keyword prompt",order: String="textual inversion first",
      embedding: Option[String]=None,loraModel: Option[String]=None,multiplier: Double=0.7)
  final case class Plan(preview: String,prompts: Vector[String],nextState: State)

  private def validate(options: Options): Unit=
    require(Set("keyword prompt","prompt keyword","keyword, prompt","prompt, keyword").contains(options.placement),"invalid keyword placement")
    require(Set("textual inversion first","model keyword first").contains(options.order),"invalid keyword order")
    require(options.multiplier.isFinite && options.multiplier>=0 && options.multiplier<=2,"invalid LoRA multiplier")
  private def optional(value: Option[String]): Option[String]=value.filter(v => v.nonEmpty && v!="None")
  private[models] def choose(keywords: Option[String],mode: String,index: BigInt,rng: KeywordRandom): Option[String]=
    require(index>=0,"keyword index must be nonnegative")
    keywords.map { text =>
      val items=text.split("\\|",-1).toVector.map(_.dropWhile(_==' ').reverse.dropWhile(_==' ').reverse)
      if items.size==1 then items.head
      else if mode=="keyword1, keyword2" || mode==items.mkString(", ") then items.mkString(", ")
      else mode match
        case "random" | "< random >" => items(rng.below(items.size))
        case "iterate" | "< iterate >" => items((index%items.size).toInt)
        case "keyword1" => items.head
        case "keyword2" => items(1)
        case _ => if items.contains(mode) then mode else items.head
    }
  private def weightText(value: Double): String=
    // Python uses scientific notation below 1e-4, Java below 1e-3.
    val decimal=java.math.BigDecimal.valueOf(value)
    if value==0 || value==math.floor(value) then java.lang.Double.toString(value)
    else if value>=0.0001 then decimal.stripTrailingZeros().toPlainString
    else
      val stripped=decimal.stripTrailingZeros()
      val exponent=stripped.precision()-stripped.scale()-1
      s"${stripped.movePointLeft(exponent).toPlainString}e-${f"${-exponent}%02d"}"
  def render(prompt: String,keyword: Option[String],loraKeyword: Option[String],options: Options): String=
    validate(options)
    val embedding=optional(options.embedding)
    val ordered=if options.order=="model keyword first" then Vector(keyword,loraKeyword,embedding)
      else Vector(embedding,loraKeyword,keyword)
    val parts=ordered.flatten
    val placed=if options.placement.startsWith("keyword") then parts:+prompt else prompt+:parts
    val tag=optional(options.loraModel).map { path =>
      val file=path.replace('\\','/').split("/",-1).last
      val dot=file.lastIndexOf('.')
      val name=if dot>=0 then file.substring(0,dot) else file
      s"<lora:$name:${weightText(options.multiplier)}>"
    }
    (tag.toVector++placed).mkString(if options.placement.contains(',') then ", " else " ")

  def plan(preview: String,prompts: Vector[String],modelKeywords: Option[String],loraKeywords: Option[String],
      options: Options,state: State,seed: Long): Plan=
    validate(options)
    require(prompts.nonEmpty && prompts.size<=1600,"keyword plan must contain 1..1600 prompts")
    val rng=new KeywordRandom(seed)
    def one(prompt: String,index: Int): String=
      val model=choose(modelKeywords,options.modelChoice,state.model+index,rng)
      val lora=if optional(options.loraModel).nonEmpty && options.loraChoice!="None" then
        choose(loraKeywords,options.loraChoice,state.lora+index,rng) else None
      render(prompt,model,lora,options)
    val renderedPreview=one(preview,0)
    val rendered=prompts.zipWithIndex.map((prompt,index) => one(prompt,index))
    val next=State(state.model+(if options.modelChoice=="iterate" && modelKeywords.exists(_.contains('|')) then prompts.size else 0),
      state.lora+(if options.loraChoice=="< iterate >" && loraKeywords.exists(_.contains('|')) then prompts.size else 0))
    Plan(renderedPreview,rendered,next)
