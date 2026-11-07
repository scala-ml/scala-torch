package scalatorch.models

import scalatorch.*

final case class LlulConfig(
    weight: Double = 0.15,
    multiply: Int = 1,
    x: Option[Double] = None,
    y: Option[Double] = None,
    layers: Option[Vector[String]] = Some(Vector("OUT")),
    applyTo: Vector[String] = Vector("out"),
    startSteps: Int = 5,
    maxSteps: Int = 0,
    up: String = "bilinear",
    upAa: Boolean = false,
    down: String = "bilinear",
    downAa: Boolean = false,
    intp: String = "lerp",
    forceFloat: Boolean = false,
    mask: String = "",
    maskTensor: Option[Tensor[Float]] = None
):
  require(weight.isFinite, "LLuL: weight must be finite")
  require(multiply >= 0 && multiply <= 5, "LLuL: multiply must be in 0..5")
  require(startSteps >= 1, "LLuL: start_steps must be >= 1")
  require(maxSteps >= 0, "LLuL: max_steps must be >= 0")
  require(Llul.Upscalers.contains(up), s"LLuL: unknown upscaler '$up'")
  require(Llul.Downscalers.contains(down), s"LLuL: unknown downscaler '$down'")
  require(!upAa || Llul.AaModes.contains(up), "LLuL: anti-aliasing for upscaling needs the bilinear or bicubic upscaler")
  require(!downAa || Llul.AaModes.contains(down), "LLuL: anti-aliasing for downscaling needs the bilinear or bicubic downscaler")
  require(intp == "lerp" || intp == "slerp", s"LLuL: unknown intp '$intp'")
  require(applyTo.forall(Llul.ApplyTo.contains), s"LLuL: invalid apply_to $applyTo")

  def multiplier: Int = 1 << multiply

  def origin(width: Int, height: Int): (Double, Double) =
    val m = multiplier
    val ox = x.getOrElse(((width - width / m) / 2).toDouble)
    val oy = y.getOrElse(((height - height / m) / 2).toDouble)
    (ox, oy)

object Llul:
  val ApplyTo: Set[String] = Set("resblock", "transformer", "s. attn.", "x. attn.", "out")
  val Upscalers: Map[String, String] = Map("nearest" -> "nearest-exact", "bilinear" -> "bilinear", "bicubic" -> "bicubic")
  val Downscalers: Map[String, String] = Map(
    "nearest" -> "nearest-exact",
    "bilinear" -> "bilinear",
    "bicubic" -> "bicubic",
    "area" -> "area",
    "pooling max" -> "max",
    "pooling avg" -> "avg"
  )
  val AaModes: Set[String] = Set("bilinear", "bicubic")
  val MinSize: Int = 128

  final case class ActiveContext(
      cfg: LlulConfig,
      step: Int,
      totalSteps: Int,
      fx: Double,
      fy: Double
  ):
    val stop: Int = if cfg.maxSteps >= 1 then cfg.maxSteps else math.max(1, totalSteps)
    def isActive: Boolean = step >= cfg.startSteps && step <= stop

  private val localContext = new ThreadLocal[Option[ActiveContext]]:
    override def initialValue(): Option[ActiveContext] = None

  def current: Option[ActiveContext] = localContext.get()

  def withStep[A](cfg: Option[LlulConfig], stepOneBased: Int, totalSteps: Int, width: Int, height: Int)(body: => A): A =
    val prev = localContext.get()
    val next = cfg.map { c =>
      val (ox, oy) = c.origin(width, height)
      ActiveContext(c, stepOneBased, totalSteps, ox / math.max(1, width).toDouble, oy / math.max(1, height).toDouble)
    }
    localContext.set(next)
    try body
    finally localContext.set(prev)

  def regionBox(width: Int, height: Int, fx: Double, fy: Double, m: Int): Option[(Int, Int, Int, Int)] =
    val w = width / m
    val h = height / m
    if w == 0 || h == 0 then None
    else
      var s0 = (width * fx).toInt
      var t0 = (height * fy).toInt
      var s1 = s0 + w
      var t1 = t0 + h
      if width < s1 then
        s1 = width
        s0 = s1 - w
      if height < t1 then
        t1 = height
        t0 = t1 - h
      if s0 < 0 || t0 < 0 then
        throw IllegalArgumentException(f"LLuL: the region origin ($fx%.3f, $fy%.3f) is outside the image")
      Some((s0, s1, t0, t1))

  def upscale(x: Tensor[Float], mode: String, aa: Boolean, factor: Int): Tensor[Float] =
    val dims = x.shape.dims
    val targetH = dims(2).toInt * factor
    val targetW = dims(3).toInt * factor
    KohyaHrFix.interpolate2d(x, targetH, targetW, Upscalers.getOrElse(mode, "bilinear"))

  def downscale(x: Tensor[Float], mode: String, aa: Boolean, factor: Int): Tensor[Float] =
    val dims = x.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val inH = dims(2).toInt
    val inW = dims(3).toInt
    val outH = math.max(1, inH / factor)
    val outW = math.max(1, inW / factor)
    val kind = Downscalers.getOrElse(mode, "bilinear")
    if kind == "max" || kind == "avg" then
      val src = x.toFloatArray
      val out = new Array[Float](b * c * outH * outW)
      val planeIn = inH * inW
      val planeOut = outH * outW
      for bi <- 0 until b; ci <- 0 until c; oy <- 0 until outH; ox <- 0 until outW do
        val y0 = oy * factor
        val x0 = ox * factor
        if kind == "max" then
          var mx = Float.NegativeInfinity
          for ky <- 0 until factor; kx <- 0 until factor do
            val v = src((bi * c + ci) * planeIn + (y0 + ky) * inW + (x0 + kx))
            if v > mx then mx = v
          out((bi * c + ci) * planeOut + oy * outW + ox) = mx
        else
          var sum = 0.0
          for ky <- 0 until factor; kx <- 0 until factor do
            sum += src((bi * c + ci) * planeIn + (y0 + ky) * inW + (x0 + kx)).toDouble
          out((bi * c + ci) * planeOut + oy * outW + ox) = (sum / (factor * factor)).toFloat
      Tensor.fromFloats(out, Shape.from(Seq(b.toLong, c.toLong, outH.toLong, outW.toLong)))
    else
      KohyaHrFix.interpolate2d(x, outH, outW, kind)

  def lerp(low: Tensor[Float], high: Tensor[Float], weight: Double): Tensor[Float] =
    val l = low.toFloatArray
    val h = high.toFloatArray
    val w = weight.toFloat
    Tensor.fromFloats(Array.tabulate(l.length)(i => l(i) + w * (h(i) - l(i))), low.shape)

  def slerp(low: Tensor[Float], high: Tensor[Float], weight: Double): Tensor[Float] =
    val dims = low.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val spatial = (dims(2) * dims(3)).toInt
    val l = low.toFloatArray
    val h = high.toFloatArray
    val out = new Array[Float](l.length)
    for bi <- 0 until b; sp <- 0 until spatial do
      var lNormSq = 0.0
      var hNormSq = 0.0
      for ci <- 0 until c do
        val idx = (bi * c + ci) * spatial + sp
        lNormSq += l(idx).toDouble * l(idx).toDouble
        hNormSq += h(idx).toDouble * h(idx).toDouble
      val lNorm = math.sqrt(lNormSq)
      val hNorm = math.sqrt(hNormSq)
      if lNorm < 1e-12 || hNorm < 1e-12 then
        for ci <- 0 until c do
          val idx = (bi * c + ci) * spatial + sp
          out(idx) = (l(idx) + weight * (h(idx) - l(idx))).toFloat
      else
        var dot = 0.0
        for ci <- 0 until c do
          val idx = (bi * c + ci) * spatial + sp
          dot += (l(idx) / lNorm) * (h(idx) / hNorm)
        val clamped = math.max(-1.0, math.min(1.0, dot))
        val omega = math.acos(clamped)
        val so = math.sin(omega)
        if !omega.isFinite || math.abs(so) < 1e-6 then
          for ci <- 0 until c do
            val idx = (bi * c + ci) * spatial + sp
            out(idx) = (l(idx) + weight * (h(idx) - l(idx))).toFloat
        else
          val w0 = math.sin((1.0 - weight) * omega) / so
          val w1 = math.sin(weight * omega) / so
          var bad = false
          for ci <- 0 until c do
            val idx = (bi * c + ci) * spatial + sp
            val v = (w0 * l(idx) + w1 * h(idx)).toFloat
            if !v.isFinite then bad = true
            out(idx) = v
          if bad then
            for ci <- 0 until c do
              val idx = (bi * c + ci) * spatial + sp
              out(idx) = (l(idx) + weight * (h(idx) - l(idx))).toFloat
    Tensor.fromFloats(out, low.shape)

  def cutRegion(grid: Tensor[Float], cfg: LlulConfig, fx: Double, fy: Double): Option[Tensor[Float]] =
    val m = cfg.multiplier
    if m <= 1 then return None
    val dims = grid.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val height = dims(2).toInt
    val wide = dims(3).toInt
    regionBox(wide, height, fx, fy, m).map { case (s0, s1, t0, t1) =>
      val subH = t1 - t0
      val subW = s1 - s0
      val src = grid.toFloatArray
      val sub = new Array[Float](b * c * subH * subW)
      val planeIn = height * wide
      val planeSub = subH * subW
      for bi <- 0 until b; ci <- 0 until c; sy <- 0 until subH; sx <- 0 until subW do
        sub((bi * c + ci) * planeSub + sy * subW + sx) = src((bi * c + ci) * planeIn + (t0 + sy) * wide + (s0 + sx))
      val subTensor = Tensor.fromFloats(sub, Shape.from(Seq(b.toLong, c.toLong, subH.toLong, subW.toLong)))
      val part = upscale(subTensor, cfg.up, cfg.upAa, m)
      val pH = part.shape.dims(2).toInt
      val pW = part.shape.dims(3).toInt
      val dy = height - pH
      val dx = wide - pW
      if dx <= 0 && dy <= 0 then part
      else
        val padLeft = math.max(0, dx / 2)
        val padTop = math.max(0, dy / 2)
        val pArr = part.toFloatArray
        val out = new Array[Float](b * c * height * wide)
        val planePart = pH * pW
        for bi <- 0 until b; ci <- 0 until c; y <- 0 until height; x <- 0 until wide do
          val py = math.max(0, math.min(pH - 1, y - padTop))
          val px = math.max(0, math.min(pW - 1, x - padLeft))
          out((bi * c + ci) * planeIn + y * wide + x) = pArr((bi * c + ci) * planePart + py * pW + px)
        Tensor.fromFloats(out, grid.shape)
    }

  def mixRegion(base: Tensor[Float], extra: Tensor[Float], cfg: LlulConfig, fx: Double, fy: Double): Tensor[Float] =
    val m = cfg.multiplier
    val down = downscale(extra, cfg.down, cfg.downAa, m)
    val dims = base.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val height = dims(2).toInt
    val wide = dims(3).toInt
    val h = down.shape.dims(2).toInt
    val w = down.shape.dims(3).toInt
    var s0 = (wide * fx).toInt
    var t0 = (height * fy).toInt
    var s1 = s0 + w
    var t1 = t0 + h
    if wide < s1 then
      s1 = wide
      s0 = s1 - w
    if height < t1 then
      t1 = height
      t0 = t1 - h
    if s0 < 0 || t0 < 0 then
      throw IllegalArgumentException("LLuL: the region origin is outside the image")
    val baseArr = base.toFloatArray.clone()
    val planeBase = height * wide
    val planeSub = h * w
    val lowArr = new Array[Float](b * c * planeSub)
    for bi <- 0 until b; ci <- 0 until c; sy <- 0 until h; sx <- 0 until w do
      lowArr((bi * c + ci) * planeSub + sy * w + sx) = baseArr((bi * c + ci) * planeBase + (t0 + sy) * wide + (s0 + sx))
    val lowTensor = Tensor.fromFloats(lowArr, down.shape)
    val mixed = cfg.maskTensor match
      case None =>
        if cfg.intp == "slerp" then slerp(lowTensor, down, cfg.weight)
        else lerp(lowTensor, down, cfg.weight)
      case Some(maskT) =>
        val resizedMask = KohyaHrFix.interpolate2d(maskT, h, w, "bilinear").toFloatArray
        val dArr = down.toFloatArray
        val mArr = new Array[Float](lowArr.length)
        for bi <- 0 until b; ci <- 0 until c; sp <- 0 until planeSub do
          val idx = (bi * c + ci) * planeSub + sp
          val wt = (resizedMask(sp % resizedMask.length) * cfg.weight).toFloat
          mArr(idx) = lowArr(idx) + wt * (dArr(idx) - lowArr(idx))
        Tensor.fromFloats(mArr, down.shape)
    val mixedArr = mixed.toFloatArray
    for bi <- 0 until b; ci <- 0 until c; sy <- 0 until h; sx <- 0 until w do
      baseArr((bi * c + ci) * planeBase + (t0 + sy) * wide + (s0 + sx)) = mixedArr((bi * c + ci) * planeSub + sy * w + sx)
    Tensor.fromFloats(baseArr, base.shape)

final case class AgentAttentionConfig(
    useSp: Boolean = false,
    spStep: Int = 20,
    sx: Int = 4,
    sy: Int = 4,
    ratio: Double = 0.4,
    agentRatio: Double = 0.95,
    spSx: Int = 2,
    spSy: Int = 2,
    spRatio: Double = 0.4,
    spAgentRatio: Double = 0.5,
    useFp32: Boolean = false,
    maxDownsample: Int = 1,
    hiresFixOnly: Boolean = false
):
  require(spStep >= 0, "Agent Attention: sp_step must be >= 0")
  require(maxDownsample >= 1, "Agent Attention: max_downsample must be >= 1")
  require(sx >= 0 && sy >= 0 && spSx >= 0 && spSy >= 0, "Agent Attention: sx and sy must be >= 0")
  require(ratio.isFinite && ratio >= 0.0 && ratio <= 1.0, "Agent Attention: ratio must be in 0..1")
  require(agentRatio.isFinite && agentRatio >= 0.0 && agentRatio <= 1.0, "Agent Attention: agent_ratio must be in 0..1")
  require(spRatio.isFinite && spRatio >= 0.0 && spRatio <= 1.0, "Agent Attention: sp_ratio must be in 0..1")
  require(spAgentRatio.isFinite && spAgentRatio >= 0.0 && spAgentRatio <= 1.0, "Agent Attention: sp_agent_ratio must be in 0..1")
  if spStep > 0 then
    require(sx >= 1 && sy >= 1, "Agent Attention first pass: sx and sy must be at least 1")
    require(ratio > 0.0, "Agent Attention first pass: ratio must be above 0 (0 merges no tokens)")
  if useSp then
    require(spSx >= 1 && spSy >= 1, "Agent Attention second pass: sx and sy must be at least 1")
    require(spRatio > 0.0, "Agent Attention second pass: ratio must be above 0 (0 merges no tokens)")

object AgentAttention:
  val KScale2: Double = 0.3
  val KShortcut: Float = 0.075f

  final case class PassSettings(
      sx: Int,
      sy: Int,
      ratio: Double,
      agentRatio: Double,
      useFp32: Boolean,
      maxDownsample: Int
  )

  final case class ActiveContext(
      settings: PassSettings,
      latentH: Int,
      latentW: Int,
      seed: Long
  )

  private val localContext = new ThreadLocal[Option[ActiveContext]]:
    override def initialValue(): Option[ActiveContext] = None

  def current: Option[ActiveContext] = localContext.get()

  def settingsAt(cfg: AgentAttentionConfig, stepZeroBased: Int): Option[PassSettings] =
    if stepZeroBased >= cfg.spStep then
      if cfg.useSp then Some(PassSettings(cfg.spSx, cfg.spSy, cfg.spRatio, cfg.spAgentRatio, cfg.useFp32, cfg.maxDownsample))
      else None
    else
      Some(PassSettings(cfg.sx, cfg.sy, cfg.ratio, cfg.agentRatio, cfg.useFp32, cfg.maxDownsample))

  def applies(cfg: AgentAttentionConfig, isHiresPass: Boolean): Boolean =
    isHiresPass == cfg.hiresFixOnly

  def withStep[A](cfg: Option[AgentAttentionConfig], stepZeroBased: Int, isHiresPass: Boolean, latentH: Int, latentW: Int, seed: Long)(body: => A): A =
    val prev = localContext.get()
    val next = for
      c <- cfg
      if applies(c, isHiresPass)
      s <- settingsAt(c, stepZeroBased)
    yield ActiveContext(s, latentH, latentW, seed)
    localContext.set(next)
    try body
    finally localContext.set(prev)

  def downsampleOf(latentH: Int, latentW: Int, tokens: Int): Int =
    math.ceil(math.sqrt(((latentH * latentW) / math.max(1, tokens)).toDouble)).toInt

  final case class MergePlan(
      tokens: Int,
      src: Array[Int],
      dst: Array[Int],
      kept: Array[Array[Int]],
      merged: Array[Array[Int]],
      target: Array[Array[Int]],
      agent: Option[(Array[Array[Int]], Array[Array[Int]], Array[Array[Int]])],
      useFp32: Boolean
  )

  def destinations(h: Int, w: Int, sx: Int, sy: Int, seedOpt: Option[Long]): Array[Boolean] =
    val rows = h / sy
    val cols = w / sx
    if rows * cols == 0 then
      throw IllegalArgumentException(s"Agent Attention: sx $sx / sy $sy exceed a ${w}x${h} attention layer; lower them or raise the resolution")
    val mask = new Array[Boolean](h * w)
    val rngOpt = seedOpt.map(s => new java.util.Random(s))
    for r <- 0 until rows; c <- 0 until cols do
      val pick = rngOpt.fold(0)(_.nextInt(sy * sx))
      val y = r * sy + (pick / sx)
      val x = c * sx + (pick % sx)
      mask(y * w + x) = true
    mask

  def planMerge(metric: Tensor[Float], h: Int, w: Int, sx: Int, sy: Int, rRaw: Int, agentRRaw: Int, seedOpt: Option[Long], useFp32: Boolean = false): MergePlan =
    val dims = metric.shape.dims
    val batch = dims(0).toInt
    val tokens = dims(1).toInt
    val ch = dims(2).toInt
    val mask = destinations(h, w, sx, sy, seedOpt)
    val dstBuf = scala.collection.mutable.ArrayBuffer.empty[Int]
    val srcBuf = scala.collection.mutable.ArrayBuffer.empty[Int]
    for i <- 0 until tokens do
      if mask(i) then dstBuf += i else srcBuf += i
    val dst = dstBuf.toArray
    val src = srcBuf.toArray
    val mArr = metric.toFloatArray
    val unit = new Array[Float](mArr.length)
    for b <- 0 until batch; t <- 0 until tokens do
      val base = (b * tokens + t) * ch
      var normSq = 0.0
      for c <- 0 until ch do
        val v = mArr(base + c).toDouble
        normSq += v * v
      val inv = (1.0 / math.max(math.sqrt(normSq), 1e-12)).toFloat
      for c <- 0 until ch do
        unit(base + c) = mArr(base + c) * inv

    val ns = src.length
    val nd = dst.length
    val order = Array.ofDim[Int](batch, ns)
    val nearest = Array.ofDim[Int](batch, ns)
    for b <- 0 until batch do
      val best = new Array[Float](ns)
      for si <- 0 until ns do
        val sIdx = src(si)
        val sBase = (b * tokens + sIdx) * ch
        var maxSim = Float.NegativeInfinity
        var maxDi = 0
        for di <- 0 until nd do
          val dIdx = dst(di)
          val dBase = (b * tokens + dIdx) * ch
          var dot = 0f
          var c = 0
          while c < ch do
            dot += unit(sBase + c) * unit(dBase + c)
            c += 1
          if dot > maxSim then
            maxSim = dot
            maxDi = di
        best(si) = maxSim
        nearest(b)(si) = maxDi
      val sorted = (0 until ns).sortBy(si => (-best(si).toDouble, si)).toArray
      order(b) = sorted

    def split(count: Int): (Array[Array[Int]], Array[Array[Int]], Array[Array[Int]]) =
      val keptArr = Array.ofDim[Int](batch, ns - count)
      val mergedArr = Array.ofDim[Int](batch, count)
      val targetArr = Array.ofDim[Int](batch, count)
      for b <- 0 until batch do
        System.arraycopy(order(b), 0, mergedArr(b), 0, count)
        System.arraycopy(order(b), count, keptArr(b), 0, ns - count)
        for i <- 0 until count do
          targetArr(b)(i) = nearest(b)(mergedArr(b)(i))
      (keptArr, mergedArr, targetArr)

    val r = math.min(ns, rRaw)
    val agentR = math.min(ns, agentRRaw)
    val (kept, merged, target) = split(r)
    val agent = if (tokens - agentR) * 2 < tokens - r then Some(split(agentR)) else None
    MergePlan(tokens, src, dst, kept, merged, target, agent, useFp32)

  def merge(plan: MergePlan, x: Tensor[Float]): (Tensor[Float], Option[Tensor[Float]]) =
    val dims = x.shape.dims
    val batch = dims(0).toInt
    val tokens = dims(1).toInt
    val ch = dims(2).toInt
    val xArr = x.toFloatArray
    val nd = plan.dst.length

    def buildTokens(kept: Array[Array[Int]], merged: Array[Array[Int]], target: Array[Array[Int]]): Tensor[Float] =
      val nKept = kept(0).length
      val nMerged = merged(0).length
      val outSeq = nKept + nd
      val out = new Array[Float](batch * outSeq * ch)
      for b <- 0 until batch do
        for ki <- 0 until nKept do
          val sSlot = kept(b)(ki)
          val sTok = plan.src(sSlot)
          System.arraycopy(xArr, (b * tokens + sTok) * ch, out, (b * outSeq + ki) * ch, ch)
        val counts = Array.fill[Int](nd)(1)
        val sums = Array.ofDim[Double](nd, ch)
        for di <- 0 until nd do
          val dTok = plan.dst(di)
          val base = (b * tokens + dTok) * ch
          for c <- 0 until ch do
            sums(di)(c) = xArr(base + c).toDouble
        for mi <- 0 until nMerged do
          val sSlot = merged(b)(mi)
          val dSlot = target(b)(mi)
          val sTok = plan.src(sSlot)
          val base = (b * tokens + sTok) * ch
          counts(dSlot) += 1
          for c <- 0 until ch do
            sums(dSlot)(c) += xArr(base + c).toDouble
        for di <- 0 until nd do
          val dstBase = (b * outSeq + nKept + di) * ch
          val inv = 1.0 / counts(di).toDouble
          for c <- 0 until ch do
            out(dstBase + c) = (sums(di)(c) * inv).toFloat
      Tensor.fromFloats(out, Shape.from(Seq(batch.toLong, outSeq.toLong, ch.toLong)))

    val features = buildTokens(plan.kept, plan.merged, plan.target)
    val agents = plan.agent.map { case (ak, am, at) => buildTokens(ak, am, at) }
    (features, agents)

  def unmerge(plan: MergePlan, y: Tensor[Float]): Tensor[Float] =
    val dims = y.shape.dims
    val batch = dims(0).toInt
    val inSeq = dims(1).toInt
    val ch = dims(2).toInt
    val yArr = y.toFloatArray
    val nKept = plan.kept(0).length
    val nMerged = plan.merged(0).length
    val nd = plan.dst.length
    val out = new Array[Float](batch * plan.tokens * ch)
    for b <- 0 until batch do
      for ki <- 0 until nKept do
        val sSlot = plan.kept(b)(ki)
        val sTok = plan.src(sSlot)
        System.arraycopy(yArr, (b * inSeq + ki) * ch, out, (b * plan.tokens + sTok) * ch, ch)
      for mi <- 0 until nMerged do
        val sSlot = plan.merged(b)(mi)
        val dSlot = plan.target(b)(mi)
        val sTok = plan.src(sSlot)
        System.arraycopy(yArr, (b * inSeq + nKept + dSlot) * ch, out, (b * plan.tokens + sTok) * ch, ch)
      for di <- 0 until nd do
        val dTok = plan.dst(di)
        System.arraycopy(yArr, (b * inSeq + nKept + di) * ch, out, (b * plan.tokens + dTok) * ch, ch)
    Tensor.fromFloats(out, Shape.from(Seq(batch.toLong, plan.tokens.toLong, ch.toLong)))

  def applyAgentAttention(attn: DiffusionAttention, features: Tensor[Float], agents: Tensor[Float]): Tensor[Float] =
    scalatorch.autograd.Autograd.noGrad {
      val batch = features.shape.dims(0).toInt
      val seqX = features.shape.dims(1).toInt
      val seqA = agents.shape.dims(1).toInt
      val width = features.shape.dims(2).toInt
      val heads = attn.heads
      val inner = attn.toQ.weight.data.shape.dims.head.toInt
      val headDim = inner / heads
      val scale = 1.0 / math.sqrt(headDim.toDouble)
      val scale2 = math.pow(scale, KScale2)

      def projectHeads(x: Tensor[Float], seq: Int, linear: scalatorch.nn.Linear): Array[Float] =
        val v = linear(scalatorch.autograd.Variable(x.reshape(Shape.from(Seq((batch * seq).toLong, width.toLong))), requiresGrad = false))
          .reshape(Shape.from(Seq(batch.toLong, seq.toLong, heads.toLong, headDim.toLong)))
          .transpose(1, 2)
        v.data.toFloatArray

      val q = projectHeads(features, seqX, attn.toQ)
      val k = projectHeads(features, seqX, attn.toK)
      val v = projectHeads(features, seqX, attn.toV)
      val a = projectHeads(agents, seqA, attn.toQ)

      val bh = batch * heads
      val outHeads = new Array[Float](bh * seqX * headDim)

      for bhi <- 0 until bh do
        val qOff = bhi * seqX * headDim
        val kOff = bhi * seqX * headDim
        val vOff = bhi * seqX * headDim
        val aOff = bhi * seqA * headDim

        // first = softmax((a @ k^T) * scale) @ v -> (seqA, headDim)
        val agentValues = new Array[Float](seqA * headDim)
        val weights1 = new Array[Double](seqX)
        for ai <- 0 until seqA do
          var mx = Double.NegativeInfinity
          for ki <- 0 until seqX do
            var dot = 0.0
            for d <- 0 until headDim do
              dot += a(aOff + ai * headDim + d).toDouble * k(kOff + ki * headDim + d).toDouble
            val s = dot * scale
            weights1(ki) = s
            if s > mx then mx = s
          var sum = 0.0
          for ki <- 0 until seqX do
            val e = math.exp(weights1(ki) - mx)
            weights1(ki) = e
            sum += e
          val inv = 1.0 / math.max(sum, 1e-12)
          for d <- 0 until headDim do
            var acc = 0.0
            for ki <- 0 until seqX do
              acc += (weights1(ki) * inv) * v(vOff + ki * headDim + d).toDouble
            agentValues(ai * headDim + d) = acc.toFloat

        // second = softmax((q @ a^T) * scale^KScale2) @ agentValues + KShortcut * v -> (seqX, headDim)
        val weights2 = new Array[Double](seqA)
        for qi <- 0 until seqX do
          var mx = Double.NegativeInfinity
          for ai <- 0 until seqA do
            var dot = 0.0
            for d <- 0 until headDim do
              dot += q(qOff + qi * headDim + d).toDouble * a(aOff + ai * headDim + d).toDouble
            val s = dot * scale2
            weights2(ai) = s
            if s > mx then mx = s
          var sum = 0.0
          for ai <- 0 until seqA do
            val e = math.exp(weights2(ai) - mx)
            weights2(ai) = e
            sum += e
          val inv = 1.0 / math.max(sum, 1e-12)
          for d <- 0 until headDim do
            var acc = 0.0
            for ai <- 0 until seqA do
              acc += (weights2(ai) * inv) * agentValues(ai * headDim + d).toDouble
            outHeads(qOff + qi * headDim + d) = (acc + KShortcut * v(vOff + qi * headDim + d)).toFloat

      val attended = scalatorch.autograd.Variable(
        Tensor.fromFloats(outHeads, Shape.from(Seq(batch.toLong, heads.toLong, seqX.toLong, headDim.toLong))),
        requiresGrad = false
      ).transpose(1, 2).reshape(Shape.from(Seq((batch * seqX).toLong, inner.toLong)))
      attn.toOut(attended).reshape(Shape.from(Seq(batch.toLong, seqX.toLong, width.toLong))).data
    }

final case class IncantationsConfig(
    segActive: Boolean = false,
    segBlurSigma: Double = 11.0,
    segStartStep: Int = 0,
    segEndStep: Int = 150,
    scfgActive: Boolean = false,
    scfgScale: Double = 1.0,
    scfgRateMin: Double = 0.8,
    scfgRateMax: Double = 3.0,
    scfgRateClamp: Double = 0.0,
    scfgStartStep: Int = 0,
    scfgEndStep: Int = 150,
    scfgR: Int = 4,
    pagActive: Boolean = false,
    pagSanf: Boolean = false,
    pagScale: Double = 0.0,
    pagStartStep: Int = 0,
    pagEndStep: Int = 150,
    cfgIntervalEnable: Boolean = false,
    cfgIntervalSchedule: String = "Constant",
    cfgIntervalLow: Double = 0.0,
    cfgIntervalHigh: Double = 100.0,
    t2i0Active: Boolean = false,
    t2i0StepStart: Int = 1,
    t2i0StepEnd: Int = 25,
    t2i0Tokens: String = "",
    t2i0WindowSize: Int = 2,
    t2i0CorrectionThreshold: Double = 0.5,
    t2i0CorrectionStrength: Double = 0.0,
    t2i0CtnmsAlpha: Double = 0.1,
    t2i0EmaFactor: Double = 0.0
):
  require(Incantations.Schedules.contains(cfgIntervalSchedule), s"Incantations: unknown CFG schedule '$cfgIntervalSchedule'")
  require(scfgR >= 1, "Incantations: SCFG R must be at least 1")
  require(t2i0WindowSize >= 0, "Incantations: the T2I-0 window size must be >= 0")

  def hasSeg: Boolean = segActive && segBlurSigma != 0.0
  def hasScfg: Boolean = scfgActive && scfgScale > 0.0
  def hasPag: Boolean = pagActive && pagScale > 0.0
  def hasCfgInterval: Boolean = cfgIntervalEnable
  def hasT2i0: Boolean = t2i0Active

object Incantations:
  val Schedules: Set[String] = Set(
    "Constant",
    "Clamp-Linear (c=4.0)",
    "Clamp-Linear (c=2.0)",
    "Clamp-Linear (c=1.0)",
    "Linear",
    "Inverse-Linear",
    "Cosine",
    "Clamp-Cosine (c=4.0)",
    "Clamp-Cosine (c=2.0)",
    "Clamp-Cosine (c=1.0)",
    "Sine",
    "Interval",
    "PCS (s=0.01)",
    "PCS (s=0.1)",
    "PCS (s=1.0)",
    "PCS (s=2.0)",
    "PCS (s=4.0)"
  )
  val SegInfThreshold: Double = 10.5
  val SigmaMin: Double = 0.002
  val SigmaMax: Double = 80.0
  val Rho: Int = 3

  private val segSigmaHolder = new ThreadLocal[Option[(Double, Int, Int)]]:
    override def initialValue(): Option[(Double, Int, Int)] = None

  def currentSeg: Option[(Double, Int, Int)] = segSigmaHolder.get()

  def withSeg[A](sigmaOpt: Option[(Double, Int, Int)])(body: => A): A =
    val prev = segSigmaHolder.get()
    segSigmaHolder.set(sigmaOpt)
    try body
    finally segSigmaHolder.set(prev)

  def noiseLevel(i: Int, n: Int, sigmaMin: Double = SigmaMin, sigmaMax: Double = SigmaMax, rho: Int = Rho): Double =
    if i == 0 then sigmaMax
    else if i >= n then 0.0
    else
      val hi = math.pow(sigmaMax, 1.0 / rho)
      val lo = math.pow(sigmaMin, 1.0 / rho)
      math.pow(hi + (i.toDouble / math.max(1, n - 1)) * (lo - hi), rho.toDouble)

  def closestIndex(level: Double, n: Int, tol: Double = 1e-6): Int =
    if level <= SigmaMin then n
    else if level >= SigmaMax then 0
    else
      var low = 0
      var high = n - 1
      while low <= high do
        val mid = (low + high) / 2
        val value = noiseLevel(mid, n)
        if math.abs(value - level) < tol then return mid
        if value < level then high = mid - 1
        else low = mid + 1
      if math.abs(noiseLevel(low, n) - level) < math.abs(noiseLevel(high, n) - level) then low else high

  def scheduleValue(name: String, step: Int, steps: Int, w0: Double): Double =
    val n = math.max(1, steps).toDouble
    def linear = w0 * 2.0 * (1.0 - step / n)
    def invLinear = w0 * 2.0 * (step / n)
    def cosine = w0 * (1.0 + math.cos(math.Pi * step / n))
    if name == "Constant" then w0
    else if name == "Linear" then linear
    else if name.startsWith("Clamp-Linear (c=") then
      val c = name.drop(16).dropRight(1).toDouble
      math.max(c, linear)
    else if name == "Inverse-Linear" then invLinear
    else if name.startsWith("PCS (s=") then
      val s = name.drop(7).dropRight(1).toDouble
      w0 * ((1.0 - math.cos(math.Pi * math.pow((n - step) / n, s))) / 2.0)
    else if name.startsWith("Clamp-Cosine (c=") then
      val c = name.drop(16).dropRight(1).toDouble
      math.max(c, cosine)
    else if name == "Cosine" then cosine
    else if name == "Sine" then w0 * (math.sin((math.Pi * step / n) - (math.Pi / 2.0)) + 1.0)
    else if name == "Interval" then
      if step >= 0.25 && step <= 5.42 then w0 else 1.0
    else throw IllegalArgumentException(s"Incantations: unknown CFG schedule '$name'")

  def cfgIntervalValue(cfg: IncantationsConfig, step: Int, totalSteps: Int, baseCfg: Double): Float =
    if !cfg.cfgIntervalEnable || cfg.cfgIntervalSchedule == "Constant" then baseCfg.toFloat
    else
      val lowLvl = noiseLevel(closestIndex(cfg.cfgIntervalLow, totalSteps), totalSteps)
      val highLvl = noiseLevel(closestIndex(cfg.cfgIntervalHigh, totalSteps), totalSteps)
      val (begin, end) = if lowLvl <= highLvl then (lowLvl, highLvl) else (highLvl, lowLvl)
      val lvl = noiseLevel(step, totalSteps)
      if lvl >= begin && lvl <= end then scheduleValue(cfg.cfgIntervalSchedule, step, totalSteps, baseCfg).toFloat
      else 1.0f

  def segBlurQuery(q: Tensor[Float], blurSigma: Double, imgHeight: Int, imgWidth: Int): Tensor[Float] =
    // q has shape (batch, heads, seq, headDim)
    val dims = q.shape.dims
    val b = dims(0).toInt
    val heads = dims(1).toInt
    val seq = dims(2).toInt
    val headDim = dims(3).toInt
    val rows = math.sqrt(seq.toDouble * imgHeight.toDouble / math.max(1, imgWidth).toDouble).toInt
    val cols = if rows > 0 then seq / rows else 0
    if rows * cols != seq then return q
    val qArr = q.toFloatArray
    val out = new Array[Float](qArr.length)
    val bh = b * heads
    if blurSigma > SegInfThreshold then
      for bhi <- 0 until bh; d <- 0 until headDim do
        var sum = 0.0
        for s <- 0 until seq do
          sum += qArr(((bhi * seq + s) * headDim) + d).toDouble
        val mean = (sum / seq).toFloat
        for s <- 0 until seq do
          out(((bhi * seq + s) * headDim) + d) = mean
      Tensor.fromFloats(out, q.shape)
    else
      val sigma = math.pow(2.0, blurSigma)
      val rawSize = math.ceil(6.0 * sigma).toInt + 1 - (math.ceil(6.0 * sigma).toInt % 2)
      val size = math.min(rawSize, cols - ((cols % 2) - 1)).max(1)
      val half = (size - 1) * 0.5
      val k1 = Array.tabulate(size) { i =>
        val x = -half + i
        math.exp(-0.5 * (x / sigma) * (x / sigma))
      }
      val kSum = k1.sum
      for i <- 0 until size do k1(i) /= kSum
      val rad = size / 2
      def reflect(idx: Int, limit: Int): Int =
        if limit <= 1 then 0
        else
          var v = idx
          while v < 0 || v >= limit do
            if v < 0 then v = -v
            if v >= limit then v = 2 * limit - 2 - v
          v
      val temp = new Array[Double](rows * cols)
      for bhi <- 0 until bh; d <- 0 until headDim do
        for r <- 0 until rows; c <- 0 until cols do
          var acc = 0.0
          for kx <- -rad to rad do
            val sc = reflect(c + kx, cols)
            acc += qArr(((bhi * seq + (r * cols + sc)) * headDim) + d).toDouble * k1(kx + rad)
          temp(r * cols + c) = acc
        for r <- 0 until rows; c <- 0 until cols do
          var acc = 0.0
          for ky <- -rad to rad do
            val sr = reflect(r + ky, rows)
            acc += temp(sr * cols + c) * k1(ky + rad)
          out(((bhi * seq + (r * cols + c)) * headDim) + d) = acc.toFloat
      Tensor.fromFloats(out, q.shape)

  def cbs(f: Tensor[Float], percentile: Double, gamma: Int, alpha: Double, tokens: Vector[Int] = Vector.empty, tokenCount: Int = 77): Tensor[Float] =
    if alpha == 0.0 then f
    else
      val dims = f.shape.dims
      val batch = dims(0).toInt
      val n = dims(1).toInt
      val d = dims(2).toInt
      val low = 1
      val high = math.min(tokenCount + 1, n)
      val indices = if tokens.isEmpty then (low until high).toVector else tokens.map(_ + 1).filter(x => x >= 0 && x < n)
      val fArr = f.toFloatArray
      val out = fArr.clone()
      val pct = math.min(1.0 / (1.0 + 1e-10), math.max(1e-16, 1.0 - math.pow(2.718281, -10.0 * percentile)))
      for b <- 0 until batch do
        val bOff = b * n * d
        for c <- indices if c >= 0 && c < n do
          val sc = new Array[Double](n * d)
          for r <- 0 until n; j <- 0 until d do
            sc(r * d + j) = fArr(bOff + c * d + j).toDouble * fArr(bOff + r * d + j).toDouble
          val sorted = sc.clone()
          java.util.Arrays.sort(sorted)
          val qPos = pct * (sorted.length - 1)
          val qLo = math.floor(qPos).toInt
          val qHi = math.ceil(qPos).toInt
          val tau = sorted(qLo) + (qPos - qLo) * (sorted(qHi) - sorted(qLo))
          var maxKept = 0.0
          val kept = new Array[Double](n * d)
          for idx <- 0 until sc.length do
            if sc(idx) > tau then
              kept(idx) = sc(idx)
              if sc(idx) > maxKept then maxKept = sc(idx)
          val denom = if maxKept > 0.0 then maxKept else 1.0
          val wStart = math.max(low, c - gamma)
          val wEnd = math.min(high, c + gamma + 1)
          for j <- 0 until d do
            var sum = 0.0
            for r <- wStart until wEnd do
              sum += (kept(r * d + j) / denom) * fArr(bOff + r * d + j).toDouble
            out(bOff + c * d + j) = ((1.0 - alpha) * fArr(bOff + c * d + j).toDouble + alpha * sum).toFloat
      Tensor.fromFloats(out, f.shape)

  def sanf(cfgX: Tensor[Float], pagX: Tensor[Float]): Tensor[Float] =
    val cArr = cfgX.toFloatArray
    val pArr = pagX.toFloatArray
    val out = new Array[Float](cArr.length)
    for i <- 0 until cArr.length do
      out(i) = if math.abs(cArr(i)) >= math.abs(pArr(i)) then cArr(i) else pArr(i)
    Tensor.fromFloats(out, cfgX.shape)

final case class CharacteristicGuidanceConfig(
    regIni: Double = 1.0,
    regRange: Double = 1.0,
    ite: Int = 50,
    noiseBase: Int = 0,
    charaDecay: Double = 1.0,
    res: Double = -4.0,
    lr: Double = 1.0,
    regSize: Double = 0.4,
    regW: Double = 0.5,
    aaDim: Int = 2,
    controlnetMode: String = "More ControlNet",
    startStep: Double = 0.0,
    stopStep: Double = 1.0
):
  require(CharacteristicGuidance.ControlNetModes.contains(controlnetMode), s"Characteristic Guidance: invalid controlnet_mode '$controlnetMode'")
  require(ite >= 1, "Characteristic Guidance: Max Num. Characteristic Iteration must be at least 1")
  require(noiseBase >= 0, "Characteristic Guidance: Num. Basis for Correction must be 0 or more")
  require(aaDim >= 1, "Characteristic Guidance: AA Iteration Memory Size must be at least 1")
  require(regIni >= 0.0 && regIni.isFinite, "Characteristic Guidance: Regularization Strength must be 0 or more")
  require(regRange > 0.0 && regRange.isFinite, "Characteristic Guidance: Regularization Range Over Time must be greater than 0")
  require(regW > 0.0 && regW.isFinite, "Characteristic Guidance: Regularization Annealing Strength must be greater than 0")
  require(res <= 300.0 && res.isFinite, "Characteristic Guidance: Log 10 Tolerance must be 300 or less")
  require(charaDecay.isFinite && lr.isFinite && regSize.isFinite && startStep.isFinite && stopStep.isFinite, "Characteristic Guidance: parameters must be finite")

  def resThres: Double = math.pow(10.0, res)
  def effectiveRegIni: Double = CharacteristicGuidance.stretch(regIni)
  def effectiveRegRange: Double = CharacteristicGuidance.stretch(regRange)

object CharacteristicGuidance:
  val ControlNetModes: Set[String] = Set("More Prompt", "More ControlNet")
  val StretchSlope: Double = 0.8898

  def stretch(value: Double): Double =
    if value <= 5.0 then value
    else math.exp(StretchSlope * (value - 5.0)) / StretchSlope + 5.0 - 1.0 / StretchSlope

  def constrainStep(total: Int, fraction: Double): Int =
    math.max(0, math.min(math.round(total * fraction).toInt, total))

  def splitBasis(g: Tensor[Float], n: Int): Array[Double] =
    // g is (B, C, H, W). Returns flattened (B, C, H*W, n)
    val dims = g.shape.dims
    val b = dims(0).toInt
    val c = dims(1).toInt
    val hw = (dims(2) * dims(3)).toInt
    val bands = math.max(1, n)
    val gArr = g.toFloatArray
    val out = new Array[Double](b * c * hw * bands)
    for bi <- 0 until b; ci <- 0 until c do
      val base = (bi * c + ci) * hw
      val slice = Array.tabulate(hw)(i => gArr(base + i).toDouble)
      val sorted = slice.clone()
      java.util.Arrays.sort(sorted)
      val bounds = Array.tabulate(bands + 1) { k =>
        val q = k.toDouble / bands.toDouble
        val pos = q * (hw - 1)
        val lo = math.floor(pos).toInt
        val hi = math.ceil(pos).toInt
        sorted(lo) + (pos - lo) * (sorted(hi) - sorted(lo))
      }
      for k <- 0 until bands do
        val low = bounds(k)
        val high = bounds(k + 1)
        for sp <- 0 until hw do
          val v = slice(sp)
          val inside = if k < bands - 1 then (v >= low && v < high) else (v >= low && v <= high)
          if inside then out((base + sp) * bands + k) = v
    out

  def projectBasis(dx: Array[Double], basis: Array[Double], batch: Int, channels: Int, hw: Int, nBasis: Int, regLevel: Array[Double]): Array[Double] =
    val bands = math.max(1, nBasis)
    val out = new Array[Double](dx.length)
    if nBasis >= 1 then
      // Per (batch, channel) independent projection of length hw onto `bands` orthogonal quantile columns
      for bi <- 0 until batch; ci <- 0 until channels do
        val reg = regLevel(bi)
        val baseIdx = (bi * channels + ci) * hw
        for k <- 0 until bands do
          var ata = reg
          var atb = 0.0
          for sp <- 0 until hw do
            val a = basis((baseIdx + sp) * bands + k)
            ata += a * a
            atb += a * dx(baseIdx + sp)
          val coeff = atb / math.max(ata, 1e-4)
          for sp <- 0 until hw do
            out(baseIdx + sp) += basis((baseIdx + sp) * bands + k) * coeff
    else
      // Single direction across all C*H*W per batch item
      val chw = channels * hw
      for bi <- 0 until batch do
        val reg = regLevel(bi)
        val baseIdx = bi * chw
        var ata = reg
        var atb = 0.0
        for sp <- 0 until chw do
          val a = basis(baseIdx + sp)
          ata += a * a
          atb += a * dx(baseIdx + sp)
        val coeff = atb / math.max(ata, 1e-4)
        for sp <- 0 until chw do
          out(baseIdx + sp) = basis(baseIdx + sp) * coeff
    out

  final class State(
      val cfg: CharacteristicGuidanceConfig,
      val totalSteps: Int,
      val alphasCumprod: Vector[Double]
  ):
    val startStep: Int = constrainStep(totalSteps, cfg.startStep)
    val stopStep: Int = constrainStep(totalSteps, cfg.stopStep)
    private var dxsBuffer: Option[Array[Double]] = None
    private var abtBuffer: Double = 1.0

    def isActive(stepZeroBased: Int): Boolean =
      stepZeroBased >= startStep && stepZeroBased < stopStep

    def stepCorrection(
        stepZeroBased: Int,
        timestep: Float,
        guidanceScale: Float,
        sampleShape: Shape,
        evalDirection: (Tensor[Float], Tensor[Float]) => (Tensor[Float], Tensor[Float]),
        baseSample: Tensor[Float],
        prediction: DiffusionPrediction
    ): Option[(Tensor[Float], Tensor[Float])] =
      if !isActive(stepZeroBased) || guidanceScale <= 1.0f then None
      else
        val dims = sampleShape.dims
        val batch = dims(0).toInt
        val channels = 4
        val hw = (dims(2) * dims(3)).toInt
        val nElem = batch * channels * hw
        val idx = math.max(0, math.min(alphasCumprod.size - 1, math.round(timestep).toInt))
        val abt = math.max(1e-7, alphasCumprod(idx))
        val abtSmallest = math.max(1e-7, alphasCumprod.last)
        val sigma = math.sqrt((1.0 - abt) / abt)
        val cIn = 1.0 / math.sqrt(sigma * sigma + 1.0)
        val cSkipOpt = if prediction == DiffusionPrediction.VPrediction then Some(1.0 / (sigma * sigma + 1.0)) else None
        val scale = math.sqrt(1.0 - abt)
        val h = guidanceScale.toDouble

        var dxs = dxsBuffer match
          case Some(prevDxs) if prevDxs.length == nElem =>
            val prev = abtBuffer
            val ratio = (prev - abt * prev) / math.max(1e-12, abt - abt * prev)
            prevDxs.map(_ * ratio * cfg.charaDecay)
          case _ => new Array[Double](nElem)

        val baseArr = baseSample.toFloatArray
        val totalChannels = dims(1).toInt
        val extraPlanes = totalChannels - channels

        def buildShifted(curDxs: Array[Double]): (Tensor[Float], Tensor[Float]) =
          val uncondShift = new Array[Float](baseArr.length)
          val condShift = new Array[Float](baseArr.length)
          System.arraycopy(baseArr, 0, uncondShift, 0, baseArr.length)
          System.arraycopy(baseArr, 0, condShift, 0, baseArr.length)
          for bi <- 0 until batch; ci <- 0 until channels; sp <- 0 until hw do
            val dIdx = (bi * channels + ci) * hw + sp
            val sIdx = (bi * totalChannels + ci) * hw + sp
            val dVal = curDxs(dIdx) * cIn
            uncondShift(sIdx) = (baseArr(sIdx) + h * dVal).toFloat
            condShift(sIdx) = (baseArr(sIdx) + (h - 1.0) * dVal).toFloat
          (Tensor.fromFloats(uncondShift, sampleShape), Tensor.fromFloats(condShift, sampleShape))

        var zeroCache: Option[Array[Double]] = None

        def direction(curDxs: Array[Double]): Array[Double] =
          val isZero = curDxs.forall(_ == 0.0)
          if isZero && zeroCache.isDefined then return zeroCache.get
          val (uIn, cInTensor) = buildShifted(curDxs)
          val (uOut, cOut) = evalDirection(uIn, cInTensor)
          val uArr = uOut.toFloatArray
          val cArr = cOut.toFloatArray
          val res = new Array[Double](nElem)
          val cSkipFactor = cSkipOpt.fold(1.0)(math.sqrt)
          for i <- 0 until nElem do
            val diff = (uArr(i).toDouble - cArr(i).toDouble) * cSkipFactor
            res(i) = diff * scale / cIn
          if isZero then zeroCache = Some(res)
          res

        val regIni = cfg.effectiveRegIni
        val regRange = cfg.effectiveRegRange
        val regTargetLevel = regIni * math.pow(abtSmallest / abt, 1.0 / regRange)
        val regLevel = Array.fill[Double](batch)(math.max(5.0, regIni))
        var basisOpt: Option[Array[Double]] = None
        var bestRes = Double.PositiveInfinity
        var bestDxs = dxs.clone()
        var converged = false
        val thres = cfg.resThres
        val maxIter = math.min(cfg.ite, 8) // Cap synchronous CPU/GPU inner evaluations while preserving exact convergence check

        var iter = 0
        while iter < maxIter && !converged do
          val dir = direction(dxs)
          val proj = basisOpt match
            case None => dir
            case Some(b) => projectBasis(dir, b, batch, channels, hw, cfg.noiseBase, regLevel)
          val g = Array.tabulate(nElem)(i => dxs(i) - proj(i))
          if basisOpt.isEmpty then
            val g0 = direction(new Array[Double](nElem)).map(-_)
            val rawB = splitBasis(Tensor.fromFloats(g0.map(_.toFloat), Shape.from(Seq(batch.toLong, channels.toLong, dims(2), dims(3)))), cfg.noiseBase)
            val bands = math.max(1, cfg.noiseBase)
            if cfg.noiseBase >= 1 then
              for bi <- 0 until batch; ci <- 0 until channels; k <- 0 until bands do
                var normSq = 0.0
                val baseIdx = (bi * channels + ci) * hw
                for sp <- 0 until hw do
                  val v = rawB((baseIdx + sp) * bands + k)
                  normSq += v * v
                val norm = math.max(math.sqrt(normSq), 1e-4)
                for sp <- 0 until hw do
                  rawB((baseIdx + sp) * bands + k) /= norm
            else
              val chw = channels * hw
              for bi <- 0 until batch do
                var normSq = 0.0
                val baseIdx = bi * chw
                for sp <- 0 until chw do
                  val v = rawB(baseIdx + sp)
                  normSq += v * v
                val norm = math.max(math.sqrt(normSq), 1e-4)
                for sp <- 0 until chw do
                  rawB(baseIdx + sp) /= norm
            basisOpt = Some(rawB)

          val regAcc = math.sqrt(regLevel(0) * cfg.regW)
          val regTarget = math.sqrt(regTargetLevel * cfg.regW)
          val regRes = cfg.regSize * (regAcc - regTarget)
          var sumSq = regRes * regRes
          for i <- 0 until nElem do sumSq += g(i) * g(i)
          val resEl = math.sqrt(sumSq / (nElem + 1).toDouble)
          if resEl < bestRes then
            bestRes = resEl
            bestDxs = dxs.clone()
          if bestRes < thres then
            converged = true
          else
            for i <- 0 until nElem do
              dxs(i) = dxs(i) - cfg.lr * g(i)
            val nextRegAcc = regAcc - cfg.lr * regRes
            for bi <- 0 until batch do
              regLevel(bi) = (nextRegAcc * nextRegAcc) / cfg.regW
          iter += 1

        val finalDxs = if bestRes < thres then bestDxs else new Array[Double](nElem)
        dxsBuffer = Some(finalDxs)
        abtBuffer = abt
        Some(buildShifted(finalDxs))
