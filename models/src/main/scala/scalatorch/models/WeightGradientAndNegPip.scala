package scalatorch.models

import scalatorch.*
import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

/** Port of sdfx/weight_gradient.py: {tokens:interval:weights:mode} prompt weight gradients. */
object WeightGradient:
  val Modes: Set[String] = Set("a", "e", "ei", "eo", "c", "ci", "co")
  private val numPat = """[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"""
  private val weightsRe = Pattern.compile(s"""^\\s*($numPat)\\s*-\\s*($numPat)(?:\\s*-\\s*($numPat))?\\s*$$""")
  private val intervalRe = Pattern.compile(s"""^\\s*($numPat%?)\\s*-\\s*($numPat%?)\\s*$$""")

  def curve(mode: String, rawX: Double): Double =
    val x = math.max(0.0, math.min(1.0, rawX))
    if x == 0.0 || x == 1.0 then x
    else mode match
      case "a" => x
      case "ei" => math.pow(2.0, 10.0 * x - 10.0)
      case "eo" => 1.0 - math.pow(2.0, -10.0 * x)
      case "e" =>
        if x < 0.5 then math.pow(2.0, 20.0 * x - 10.0) / 2.0
        else 1.0 - math.pow(2.0, -20.0 * x + 10.0) / 2.0
      case "ci" => 1.0 - math.sqrt(math.max(0.0, 1.0 - x * x))
      case "co" => math.sqrt(math.max(0.0, 1.0 - (x - 1.0) * (x - 1.0)))
      case "c" =>
        if x < 0.5 then (1.0 - math.sqrt(math.max(0.0, 1.0 - (2.0 * x) * (2.0 * x)))) / 2.0
        else (1.0 + math.sqrt(math.max(0.0, 1.0 - (2.0 - 2.0 * x) * (2.0 - 2.0 * x)))) / 2.0
      case other => throw IllegalArgumentException(s"unknown weight gradient mode: $other")

  final case class Gradient(
      tokens: String,
      interval: Option[((Double, Boolean), (Double, Boolean))],
      weights: Vector[Double],
      mode: String
  ):
    def weight(step: Int, total: Int): Option[Double] =
      val (start, end) = interval match
        case None => (1, total)
        case Some(((v1, f1), (v2, f2))) =>
          val s = math.max(1, math.floor(if f1 then v1 * total else v1).toInt)
          val e = math.max(1, math.floor(if f2 then v2 * total else v2).toInt)
          (s, e)
      if end < start then None
      else
        val rawX = if start == end && step >= end then 1.0 else (step - start).toDouble / math.max(1, end - start).toDouble
        var x = math.max(0.0, math.min(1.0, rawX))
        var a = weights(0)
        var b = weights(1)
        if weights.size == 3 then
          if x <= 0.5 then x *= 2.0
          else
            a = b
            b = weights(2)
            x = (x - 0.5) * 2.0
        Some(a + (b - a) * curve(mode, x))

  private def splitFields(body: String): Option[Vector[String]] =
    val fields = ArrayBuffer.empty[String]
    val stack = ArrayBuffer.empty[Char]
    var start = 0
    var i = 0
    while i < body.length do
      val c = body.charAt(i)
      if c == '\\' then i += 2
      else
        if c == '(' || c == '[' then
          stack += c
          i += 1
        else if c == ')' || c == ']' then
          val expected = if c == ')' then '(' else '['
          if stack.isEmpty || stack.remove(stack.size - 1) != expected then return None
          i += 1
        else if c == '{' || c == '}' then return None
        else if c == ':' && stack.isEmpty then
          fields += body.substring(start, i).trim
          start = i + 1
          i += 1
        else i += 1
    if stack.nonEmpty then None
    else
      fields += body.substring(start).trim
      Some(fields.toVector)

  private def parsePoint(raw: String): Option[(Double, Boolean)] =
    if raw.endsWith("%") then
      raw.dropRight(1).toDoubleOption.filter(_.isFinite).map(v => (v / 100.0, true))
    else
      raw.toDoubleOption.filter(_.isFinite).map { v =>
        val fractional = v >= 0.0 && v <= 1.0 && raw.toLowerCase(java.util.Locale.ROOT).exists(c => c == '.' || c == 'e')
        (v, fractional)
      }

  private def parseGradient(body: String): Option[Gradient] =
    splitFields(body).flatMap { fields =>
      if (fields.size != 3 && fields.size != 4) || fields(0).isEmpty then None
      else
        val rawMode = if fields.size == 4 then fields(3) else "a"
        val mode = if rawMode.isEmpty then "a" else rawMode
        val wm = weightsRe.matcher(fields(2))
        if !Modes.contains(mode) || !wm.matches() then None
        else
          val w1 = wm.group(1).toDoubleOption
          val w2 = wm.group(2).toDoubleOption
          val w3 = Option(wm.group(3)).flatMap(_.toDoubleOption)
          if w1.isEmpty || w2.isEmpty || (wm.group(3) != null && w3.isEmpty) then None
          else
            val weights = Vector(w1.get, w2.get) ++ w3.toVector
            if !weights.forall(_.isFinite) then None
            else if fields(1).isEmpty then Some(Gradient(fields(0), None, weights, mode))
            else
              val im = intervalRe.matcher(fields(1))
              if !im.matches() then None
              else
                for
                  p1 <- parsePoint(im.group(1))
                  p2 <- parsePoint(im.group(2))
                  if !(p1._2 == p2._2 && p1._1 > p2._1)
                yield Gradient(fields(0), Some((p1, p2)), weights, mode)
    }

  private def findTags(text: String): Vector[(Int, Int, Gradient)] =
    val out = ArrayBuffer.empty[(Int, Int, Gradient)]
    var i = 0
    while i < text.length do
      if text.charAt(i) == '\\' then i += 2
      else if text.charAt(i) != '{' then i += 1
      else
        var end = i + 1
        var depth = 1
        while end < text.length && depth > 0 do
          if text.charAt(end) == '\\' then end += 2
          else
            if text.charAt(end) == '{' then depth += 1
            else if text.charAt(end) == '}' then depth -= 1
            end += 1
        if depth > 0 then i = text.length
        else
          parseGradient(text.substring(i + 1, end - 1)).foreach { g =>
            out += ((i, end, g))
          }
          i = end
    out.toVector

  def hasGradient(text: String): Boolean = findTags(text).nonEmpty

  private def formatWeight(w: Double): String =
    val formatted = String.format(java.util.Locale.ROOT, "%.4f", java.lang.Double.valueOf(w))
      .replaceAll("0+$", "")
      .replaceAll("\\.$", "")
    if formatted == "-0" || formatted.isEmpty then "0" else formatted

  def render(text: String, step: Int, total: Int): String =
    val tags = findTags(text)
    if tags.isEmpty then text
    else
      val sb = new java.lang.StringBuilder()
      var at = 0
      tags.foreach { case (start, end, gradient) =>
        sb.append(text.substring(at, start))
        gradient.weight(step, total) match
          case None => sb.append(text.substring(start, end))
          case Some(w) => sb.append("(").append(gradient.tokens).append(":").append(formatWeight(w)).append(")")
        at = end
      }
      sb.append(text.substring(at))
      sb.toString

  def schedule(
      text: String,
      steps: Int,
      hiresSteps: Option[Int] = None,
      gradientSteps: Option[Int] = None
  ): Vector[(Int, String)] =
    val total = gradientSteps.orElse(hiresSteps).getOrElse(steps)
    if !hasGradient(text) then PromptSchedule.schedule(text, steps, hiresSteps)
    else
      val result = ArrayBuffer.empty[(Int, String)]
      for step <- 1 to math.max(1, total) do
        val expanded = render(text, step, total)
        val value = PromptSchedule.textAt(PromptSchedule.schedule(expanded, steps, hiresSteps), step)
        if result.nonEmpty && result.last._2 == value then
          result(result.size - 1) = (step, value)
        else
          result += ((step, value))
      result.toVector

/** Port of sdfx/negpip.py: negative prompt weights via negated cross-attention value projections. */
object NegPip:
  private val localSigns = new ThreadLocal[Option[Array[Float]]]:
    override def initialValue(): Option[Array[Float]] = None

  def currentSigns: Option[Array[Float]] = localSigns.get()

  def withSigns[A](signs: Option[Array[Float]])(body: => A): A =
    val prev = localSigns.get()
    localSigns.set(signs)
    try body
    finally localSigns.set(prev)

  def escapeText(text: String): String =
    val s1 = text.replace("\\", "\\\\")
    val sb = new java.lang.StringBuilder(s1.length + 8)
    var i = 0
    while i < s1.length do
      val c = s1.charAt(i)
      if c == '(' || c == ')' || c == '[' || c == ']' then sb.append('\\')
      sb.append(c)
      i += 1
    sb.toString

  private def formatG(w: Double): String =
    val bd = BigDecimal(w).bigDecimal.stripTrailingZeros()
    bd.toPlainString

  def extract(text: String): (String, Vector[(String, Double)]) =
    val parsed = DiffusionPrompt.parse(text)
    val targets = parsed.collect {
      case PromptSpan(t, w) if t.nonEmpty && w < 0.0 && !(t == "BREAK" && w == -1.0) => (t, w)
    }
    if targets.isEmpty then (text, Vector.empty)
    else
      val sb = new java.lang.StringBuilder()
      parsed.foreach { case PromptSpan(t, w) =>
        if t == "BREAK" && w == -1.0 then sb.append(" BREAK ")
        else if w >= 0.0 then
          val lit = escapeText(t)
          if w == 1.0 then sb.append(lit)
          else sb.append("(").append(lit).append(":").append(formatG(w)).append(")")
      }
      (sb.toString, targets)

  def tokenPositions(tokenizers: Seq[ClipFastTokenizer], text: String, totalLength: Int): Vector[Int] =
    val escaped = escapeText(text)
    val chunksPerTok = tokenizers.map(tok => (tok, ClipPromptEncoding.chunks(tok, escaped)))
    val out = ArrayBuffer.empty[Int]
    val numChunks = totalLength / 77
    for i <- 0 until numChunks do
      val ends = chunksPerTok.map { case (tok, blocks) =>
        val tokens = blocks(math.min(i, blocks.size - 1)).tokens
        val idx = tokens.indexOf(tok.eosId, 1)
        if idx >= 1 then idx else 76
      }
      val maxEnd = if ends.nonEmpty then ends.max else 1
      for pos <- (i * 77 + 1) until (i * 77 + maxEnd + 1) do
        out += pos
    out.toVector

  def sliceTokens(context: Tensor[Float], positions: Vector[Int]): Tensor[Float] =
    val dims = context.shape.dims
    val seqLen = dims(1).toInt
    val width = dims(2).toInt
    val src = context.toFloatArray
    val out = new Array[Float](positions.size * width)
    var i = 0
    while i < positions.size do
      val p = positions(i)
      if p >= 0 && p < seqLen then
        System.arraycopy(src, p * width, out, i * width, width)
      i += 1
    Tensor.fromFloats(out, Shape.from(Seq(1L, positions.size.toLong, width.toLong)))

  def concatTokens(pieces: Seq[Tensor[Float]]): Tensor[Float] =
    require(pieces.nonEmpty, "NegPiP requires at least one context piece")
    if pieces.size == 1 then pieces.head
    else
      val width = pieces.head.shape.dims(2).toInt
      val totalTokens = pieces.map(_.shape.dims(1).toInt).sum
      val out = new Array[Float](totalTokens * width)
      var offset = 0
      pieces.foreach { t =>
        val arr = t.toFloatArray
        System.arraycopy(arr, 0, out, offset, arr.length)
        offset += arr.length
      }
      Tensor.fromFloats(out, Shape.from(Seq(1L, totalTokens.toLong, width.toLong)))

  /** Multiplies value projection tensor of shape (batch, heads, keys, headDim) by per-key sign (+1 or -1). */
  def applySigns(v: Tensor[Float], signs: Array[Float]): Tensor[Float] =
    val dims = v.shape.dims
    require(dims.size == 4, s"expected 4D value tensor (batch, heads, keys, headDim), got ${dims.mkString("x")}")
    val batch = dims(0).toInt
    val heads = dims(1).toInt
    val keys = dims(2).toInt
    val headDim = dims(3).toInt
    if signs.length != keys || signs.forall(_ == 1f) then v
    else
      val data = v.toFloatArray.clone()
      var bh = 0
      val bhCount = batch * heads
      while bh < bhCount do
        val bhBase = bh * keys * headDim
        var k = 0
        while k < keys do
          val s = signs(k)
          if s != 1f then
            val kBase = bhBase + k * headDim
            var d = 0
            while d < headDim do
              data(kBase + d) = data(kBase + d) * s
              d += 1
          k += 1
        bh += 1
      Tensor.fromFloats(data, v.shape)

/** Port of sdfx/conditioning_highres_fix.py: 9-channel inpainting masked-image conditioning strength. */
object ConditioningHighresFix:
  val Modes: Set[String] = Set("disabled", "empty", "lowest", "low", "medium", "high", "highest", "full")

  def resolveWeight(mode: String, denoising: Option[Double]): Option[Float] =
    val m = mode.trim.toLowerCase(java.util.Locale.ROOT)
    require(Modes.contains(m), s"Conditioning Highres Fix: unknown conditioning highres mode '$mode'")
    if m == "disabled" || denoising.isEmpty then None
    else
      val d = denoising.get
      require(d.isFinite && d >= 0.0 && d <= 1.0, "Conditioning Highres Fix: denoising strength must be a number between 0 and 1")
      val w = m match
        case "empty" => 1.0
        case "lowest" => 1.0 - d / 4.0
        case "low" => 1.0 - d / 2.0
        case "medium" => 0.5
        case "high" => d / 2.0
        case "highest" => d / 4.0
        case "full" => 0.0
      Some(w.toFloat)

  def conditionPixels(image: Tensor[Float], mask: Tensor[Float], weight: Float): Tensor[Float] =
    val iDims = image.shape.dims
    val mDims = mask.shape.dims
    require(iDims.size == 4 && mDims.size == 4 && iDims(2) == mDims(2) && iDims(3) == mDims(3),
      "conditioning image and mask dimensions differ")
    val imgArr = image.toFloatArray
    val maskArr = mask.toFloatArray
    val plane = (iDims(2) * iDims(3)).toInt
    val out = new Array[Float](imgArr.length)
    var i = 0
    while i < imgArr.length do
      val sel = if maskArr(i % plane) >= 0.5f then 1.0f else 0.0f
      out(i) = imgArr(i) * (1.0f - sel * weight)
      i += 1
    Tensor.fromFloats(out, image.shape)
