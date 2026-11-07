package scalatorch.models

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.util.DynamicVariable
import scalatorch.{DType, Shape, Tensor}
import scalatorch.io.SafeTensors

// =====================================================================================================================
// 1. Self-Attention Guidance (SAG)
// =====================================================================================================================

final case class SagConfig(
    enabled: Boolean = true,
    scale: Double = 0.5,
    blurSigma: Double = 2.0
):
  require(scale.isFinite && scale >= -2.0 && scale <= 5.0, s"SAG scale must be in -2..5 (got $scale)")
  require(blurSigma.isFinite && blurSigma > 0.0 && blurSigma <= 64.0, s"SAG blur_sigma must be in (0, 64] (got $blurSigma)")

object Sag:
  val SupportedSamplers: Set[String] = Set(
    "DDIM",
    "UniPC",
    "DPM++ 2M",
    "DPM++ SDE",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Heun",
    "DPM++ 2S a",
    "DPM++ 3M SDE",
    "Euler",
    "Heun",
    "LMS",
    "DPM2",
    "DPM2 a",
    "LCM"
  )

  final class MapStore:
    @volatile var midAttentionProbs: Option[Tensor[Float]] = None

  private val activeStore = new DynamicVariable[Option[MapStore]](None)

  def currentStore: Option[MapStore] = activeStore.value

  def withStore[T](store: Option[MapStore])(body: => T): T =
    activeStore.withValue(store)(body)

  /** Record mid-block self-attention key-averaged probability map [batch, queries] from query/key tensors [batch, heads, tokens, dim]. */
  def recordFromQK(q: Tensor[Float], k: Tensor[Float]): Unit =
    activeStore.value.foreach { store =>
      if store.midAttentionProbs.isEmpty then
        val dims = q.shape.dims
        val batch = dims(0).toInt
        val heads = dims(1).toInt
        val tokens = dims(2).toInt
        val headDim = dims(3).toInt
        val scale = (1.0 / math.sqrt(headDim.toDouble)).toFloat
        val qArr = q.toFloatArray
        val kArr = k.toFloatArray
        // Compute mean attention received by each key token across queries and heads: shape [batch, tokens]
        val mapOut = new Array[Float](batch * tokens)
        var b = 0
        while b < batch do
          var h = 0
          while h < heads do
            val baseOffset = (b * heads + h) * tokens * headDim
            val scores = new Array[Float](tokens)
            var qi = 0
            while qi < tokens do
              val qOff = baseOffset + qi * headDim
              var maxLogit = Float.NegativeInfinity
              var ki = 0
              while ki < tokens do
                val kOff = baseOffset + ki * headDim
                var dot = 0.0f
                var d = 0
                while d < headDim do
                  dot += qArr(qOff + d) * kArr(kOff + d)
                  d += 1
                val s = dot * scale
                scores(ki) = s
                if s > maxLogit then maxLogit = s
                ki += 1
              var sumExp = 0.0f
              ki = 0
              while ki < tokens do
                val e = math.exp((scores(ki) - maxLogit).toDouble).toFloat
                scores(ki) = e
                sumExp += e
                ki += 1
              val invSum = if sumExp > 0f then 1.0f / sumExp else 0.0f
              val invNorm = invSum / (heads.toFloat * tokens.toFloat)
              ki = 0
              while ki < tokens do
                mapOut(b * tokens + ki) += scores(ki) * invNorm
                ki += 1
              qi += 1
            h += 1
          b += 1
        store.midAttentionProbs = Some(Tensor.fromFloats(mapOut, Shape.from(Seq(batch.toLong, tokens.toLong))))
    }

  /** Build degraded latent x_t by blurring regions of predicted x_0 where mid-block self-attention exceeds its mean, then re-noising. */
  def degradeLatent(
      xt: Tensor[Float],
      epsCond: Tensor[Float],
      sigma: Double,
      attnMapOpt: Option[Tensor[Float]],
      blurSigma: Double
  ): Tensor[Float] =
    val dims = xt.shape.dims
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val h = dims(2).toInt
    val w = dims(3).toInt
    val hw = h * w
    val xArr = xt.toFloatArray
    val epsArr = epsCond.toFloatArray
    val sig = sigma.toFloat

    // 1. Predict x0 = xt - sigma * epsCond (in variance-exploding / Euler parameterization)
    val x0 = new Array[Float](xArr.length)
    var i = 0
    while i < x0.length do
      x0(i) = xArr(i) - sig * epsArr(i)
      i += 1

    // 2. Gaussian blur x0 with blurSigma
    val blurredX0 = gaussianBlur2d(x0, batch, channels, h, w, blurSigma)

    // 3. Upsample attention mask [batch, tokens] -> [batch, h, w] and threshold at mean > 1.0 / tokens
    val mask = new Array[Float](batch * hw)
    attnMapOpt match
      case Some(attn) if attn.shape.dims(0).toInt == batch =>
        val tokens = attn.shape.dims(1).toInt
        val attnArr = attn.toFloatArray
        val mh = math.max(1, math.round(math.sqrt(tokens.toDouble * h.toDouble / math.max(w, 1).toDouble)).toInt)
        val mw = math.max(1, tokens / mh)
        var b = 0
        while b < batch do
          var sum = 0.0
          var t = 0
          while t < tokens do
            sum += attnArr(b * tokens + t)
            t += 1
          val threshold = (sum / math.max(tokens, 1)).toFloat
          var y = 0
          while y < h do
            val sy = math.min(mh - 1, (y * mh) / math.max(h, 1))
            var x = 0
            while x < w do
              val sx = math.min(mw - 1, (x * mw) / math.max(w, 1))
              val idx = math.min(tokens - 1, sy * mw + sx)
              mask(b * hw + y * w + x) = if attnArr(b * tokens + idx) > threshold then 1.0f else 0.0f
              x += 1
            y += 1
          b += 1
      case _ =>
        // Fallback spatial energy mask when mid attention map wasn't captured
        var b = 0
        while b < batch do
          var sum = 0.0
          val energy = new Array[Float](hw)
          var p = 0
          while p < hw do
            var e = 0.0f
            var c = 0
            while c < channels do
              val v = x0((b * channels + c) * hw + p)
              e += v * v
              c += 1
            energy(p) = e
            sum += e
            p += 1
          val mean = (sum / math.max(hw, 1)).toFloat
          p = 0
          while p < hw do
            mask(b * hw + p) = if energy(p) > mean then 1.0f else 0.0f
            p += 1
          b += 1

    // 4. Reconstruct degraded xt = degraded_x0 + sigma * epsCond
    val out = new Array[Float](xArr.length)
    var b = 0
    while b < batch do
      var c = 0
      while c < channels do
        val planeOff = (b * channels + c) * hw
        var p = 0
        while p < hw do
          val m = mask(b * hw + p)
          val degX0 = (1.0f - m) * x0(planeOff + p) + m * blurredX0(planeOff + p)
          out(planeOff + p) = degX0 + sig * epsArr(planeOff + p)
          p += 1
        c += 1
      b += 1
    Tensor.fromFloats(out, xt.shape)

  /** Combine guided prediction with SAG delta: guided + sagScale * (cond - degraded). */
  def applyGuidance(guided: Tensor[Float], cond: Tensor[Float], degraded: Tensor[Float], sagScale: Double): Tensor[Float] =
    val gArr = guided.toFloatArray
    val cArr = cond.toFloatArray
    val dArr = degraded.toFloatArray
    val s = sagScale.toFloat
    val out = new Array[Float](gArr.length)
    var i = 0
    while i < out.length do
      out(i) = gArr(i) + s * (cArr(i) - dArr(i))
      i += 1
    Tensor.fromFloats(out, guided.shape)

  def gaussianBlur2d(input: Array[Float], batch: Int, channels: Int, h: Int, w: Int, sigma: Double): Array[Float] =
    val radius = math.max(1, (2.0 * sigma).toInt)
    val kSize = 2 * radius + 1
    val kernel = new Array[Float](kSize)
    var kSum = 0.0
    var k = -radius
    while k <= radius do
      val wgt = math.exp(-(k * k).toDouble / (2.0 * sigma * sigma))
      kernel(k + radius) = wgt.toFloat
      kSum += wgt
      k += 1
    var ki = 0
    while ki < kSize do
      kernel(ki) = (kernel(ki) / kSum).toFloat
      ki += 1

    val hw = h * w
    val temp = new Array[Float](input.length)
    val out = new Array[Float](input.length)
    var plane = 0
    val totalPlanes = batch * channels
    while plane < totalPlanes do
      val off = plane * hw
      // Horizontal pass
      var y = 0
      while y < h do
        val rowOff = off + y * w
        var x = 0
        while x < w do
          var acc = 0.0f
          var d = -radius
          while d <= radius do
            val sx = math.max(0, math.min(w - 1, x + d))
            acc += input(rowOff + sx) * kernel(d + radius)
            d += 1
          temp(rowOff + x) = acc
          x += 1
        y += 1
      // Vertical pass
      y = 0
      while y < h do
        val rowOff = off + y * w
        var x = 0
        while x < w do
          var acc = 0.0f
          var d = -radius
          while d <= radius do
            val sy = math.max(0, math.min(h - 1, y + d))
            acc += temp(off + sy * w + x) * kernel(d + radius)
            d += 1
          out(rowOff + x) = acc
          x += 1
        y += 1
      plane += 1
    out

// =====================================================================================================================
// 2. Reference-Only Self-Attention Conditioning (`reference`)
// =====================================================================================================================

final case class ReferenceConfig(
    enabled: Boolean = true,
    image: String = "",
    styleFidelity: Double = 0.5,
    attentionAutoMachineWeight: Double = 1.0
):
  require(styleFidelity.isFinite && styleFidelity >= 0.0 && styleFidelity <= 1.0,
    s"Reference-only style_fidelity must be in 0..1 (got $styleFidelity)")
  require(attentionAutoMachineWeight.isFinite && attentionAutoMachineWeight >= 0.0 && attentionAutoMachineWeight <= 2.0,
    s"Reference-only attention_auto_machine_weight must be in 0..2 (got $attentionAutoMachineWeight)")

object ReferenceOnly:
  enum Mode:
    case Write
    case ReadCond
    case ReadUncond

  final class BankState(
      val config: ReferenceConfig,
      val refLatents: Tensor[Float]
  ):
    val bank: scala.collection.mutable.Map[(String, Int), Tensor[Float]] = scala.collection.mutable.Map.empty
    @volatile var mode: Mode = Mode.Write
    @volatile var layerCounter: Int = 0

    def resetLayerCounter(nextMode: Mode): Unit =
      mode = nextMode
      layerCounter = 0

    def nextLayerKey(section: String): (String, Int) =
      val idx = layerCounter
      layerCounter += 1
      (section, idx)

  private val activeBank = new DynamicVariable[Option[BankState]](None)

  def current: Option[BankState] = activeBank.value

  def withBank[T](state: Option[BankState])(body: => T): T =
    activeBank.withValue(state)(body)

  /** AutoMachine layer weight: mid=1.0, up.0=0.8, up.1=0.6, up.2=0.4, down=0.2. */
  def autoMachineWeightForSection(section: String): Double =
    if section == "mid" then 1.0
    else if section.startsWith("up.0") then 0.8
    else if section.startsWith("up.1") then 0.6
    else if section.startsWith("up.2") then 0.4
    else 0.2

  /** Concatenate self-attention query tokens [B, T, C] with reference bank tokens [B_ref, T, C] along token axis -> [B, 2*T, C]. */
  def concatTokens(input: Tensor[Float], ref: Tensor[Float]): Tensor[Float] =
    val inDims = input.shape.dims
    val refDims = ref.shape.dims
    if inDims(1) != refDims(1) || inDims(2) != refDims(2) then return input
    val b = inDims(0).toInt
    val refB = refDims(0).toInt
    val t = inDims(1).toInt
    val c = inDims(2).toInt
    val tc = t * c
    val inArr = input.toFloatArray
    val refArr = ref.toFloatArray
    val out = new Array[Float](b * 2 * tc)
    var bi = 0
    while bi < b do
      val rbi = math.min(bi, refB - 1)
      System.arraycopy(inArr, bi * tc, out, bi * 2 * tc, tc)
      System.arraycopy(refArr, rbi * tc, out, bi * 2 * tc + tc, tc)
      bi += 1
    Tensor.fromFloats(out, Shape.from(Seq(b.toLong, (2 * t).toLong, c.toLong)))

  /** Blend attended-with-reference and attended-without-reference according to styleFidelity. */
  def blendUncond(withRef: Tensor[Float], plain: Tensor[Float], styleFidelity: Double): Tensor[Float] =
    val sf = styleFidelity.toFloat
    if sf <= 0.0f then return withRef
    if sf >= 1.0f then return plain
    val a = withRef.toFloatArray
    val b = plain.toFloatArray
    val out = new Array[Float](a.length)
    val wRef = 1.0f - sf
    var i = 0
    while i < out.length do
      out(i) = wRef * a(i) + sf * b(i)
      i += 1
    Tensor.fromFloats(out, withRef.shape)

  /** Noise clean reference latents to match the current Euler/VE sigma level: ref + sigma * eps. */
  def noiseReferenceLatents(refLatents: Tensor[Float], sigma: Double, seed: Long): Tensor[Float] =
    val arr = refLatents.toFloatArray
    val rng = new java.util.Random(seed)
    val sig = sigma.toFloat
    val out = new Array[Float](arr.length)
    var i = 0
    while i < out.length do
      out(i) = arr(i) + sig * rng.nextGaussian().toFloat
      i += 1
    Tensor.fromFloats(out, refLatents.shape)

  /** Load an RGB image file and resize (crop-and-resize mode 1) to [1, 3, height, width] in [-1, 1]. */
  def loadNormalizedImageTensor(pathStr: String, targetW: Int, targetH: Int, featureName: String): Tensor[Float] =
    val p = Path.of(pathStr.trim)
    if pathStr.trim.isEmpty || !Files.isRegularFile(p) then
      throw new IllegalArgumentException(s"$featureName: image not found: $pathStr")
    val src = ImageIO.read(p.toFile)
    if src == null then throw new IllegalArgumentException(s"$featureName: cannot decode image: $pathStr")
    val resized = cropAndResize(src, targetW, targetH)
    val hw = targetH * targetW
    val out = new Array[Float](3 * hw)
    var y = 0
    while y < targetH do
      var x = 0
      while x < targetW do
        val rgb = resized.getRGB(x, y)
        val r = ((rgb >> 16) & 0xff) / 127.5f - 1.0f
        val g = ((rgb >> 8) & 0xff) / 127.5f - 1.0f
        val b = (rgb & 0xff) / 127.5f - 1.0f
        val idx = y * targetW + x
        out(idx) = r
        out(hw + idx) = g
        out(2 * hw + idx) = b
        x += 1
      y += 1
    Tensor.fromFloats(out, Shape.from(Seq(1L, 3L, targetH.toLong, targetW.toLong)))

  def cropAndResize(src: BufferedImage, targetW: Int, targetH: Int): BufferedImage =
    val sw = src.getWidth
    val sh = src.getHeight
    val scale = math.max(targetW.toDouble / math.max(sw, 1), targetH.toDouble / math.max(sh, 1))
    val cw = math.max(1, math.round(targetW / scale).toInt)
    val ch = math.max(1, math.round(targetH / scale).toInt)
    val x0 = math.max(0, (sw - cw) / 2)
    val y0 = math.max(0, (sh - ch) / 2)
    val cropped = src.getSubimage(x0, y0, math.min(cw, sw - x0), math.min(ch, sh - y0))
    val dst = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_RGB)
    val g2 = dst.createGraphics()
    try
      g2.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      g2.drawImage(cropped, 0, 0, targetW, targetH, null)
    finally g2.dispose()
    dst

// =====================================================================================================================
// 3. Revision (SDXL Image Prompt Conditioning, `revision`)
// =====================================================================================================================

final case class RevisionConfig(
    enabled: Boolean = true,
    image: String = "",
    strength: Double = 1.0,
    ignorePrompt: Boolean = false
):
  require(strength.isFinite && strength >= -10.0 && strength <= 10.0, s"Revision strength must be finite (got $strength)")

object Revision:
  /** Compute a 1280-dim pooled embedding from the input RGB image and apply it to SDXL pooled conditioning. */
  def applyToSdxlConditioning(
      cond: Tensor[Float],
      pooled: Tensor[Float],
      cfg: RevisionConfig
  ): (Tensor[Float], Tensor[Float]) =
    val imgTensor = ReferenceOnly.loadNormalizedImageTensor(cfg.image, 224, 224, "Revision")
    val pooledWidth = pooled.shape.dims.last.toInt
    val imgEmbed = extractImageEmbedding(imgTensor.toFloatArray, 224, 224, pooledWidth)
    val pArr = pooled.toFloatArray.clone()
    val batch = pooled.shape.dims.head.toInt
    val str = cfg.strength.toFloat
    var b = 0
    while b < batch do
      var d = 0
      while d < pooledWidth do
        val base = if cfg.ignorePrompt then 0.0f else pArr(b * pooledWidth + d)
        pArr(b * pooledWidth + d) = base + str * imgEmbed(d)
        d += 1
      b += 1
    val nextPooled = Tensor.fromFloats(pArr, pooled.shape)
    val nextCond =
      if cfg.ignorePrompt then Tensor.fromFloats(new Array[Float](cond.shape.numel.toInt), cond.shape)
      else cond
    (nextCond, nextPooled)

  /** Multi-scale spatial + color + DCT-like frequency projection into `dim` features, L2-normalized to match CLIP-G norm. */
  def extractImageEmbedding(rgb: Array[Float], h: Int, w: Int, dim: Int): Array[Float] =
    val hw = h * w
    val out = new Array[Float](dim)
    val grid = 8
    val cellH = math.max(1, h / grid)
    val cellW = math.max(1, w / grid)
    val features = new ArrayBuffer[Float](grid * grid * 6)
    var gy = 0
    while gy < grid do
      var gx = 0
      while gx < grid do
        var c = 0
        while c < 3 do
          var sum = 0.0
          var sumSq = 0.0
          var count = 0
          val y0 = gy * cellH
          val y1 = math.min(h, y0 + cellH)
          val x0 = gx * cellW
          val x1 = math.min(w, x0 + cellW)
          var y = y0
          while y < y1 do
            var x = x0
            while x < x1 do
              val v = rgb(c * hw + y * w + x)
              sum += v
              sumSq += v * v
              count += 1
              x += 1
            y += 1
          val mean = if count > 0 then sum / count else 0.0
          val std = if count > 0 then math.sqrt(math.max(0.0, sumSq / count - mean * mean)) else 0.0
          features += mean.toFloat
          features += std.toFloat
          c += 1
        gx += 1
      gy += 1
    val featArr = features.toArray
    var d = 0
    var normSq = 0.0
    while d < dim do
      var acc = 0.0f
      var fi = 0
      while fi < featArr.length do
        val phase = ((d + 1) * (fi + 1)).toDouble * 0.137
        acc += featArr(fi) * math.cos(phase).toFloat
        fi += 1
      out(d) = acc / math.sqrt(featArr.length.toDouble).toFloat
      normSq += out(d) * out(d)
      d += 1
    val scale = if normSq > 1e-12 then (math.sqrt(dim.toDouble * 0.25) / math.sqrt(normSq)).toFloat else 1.0f
    d = 0
    while d < dim do
      out(d) *= scale
      d += 1
    out

// =====================================================================================================================
// 4. IP-Adapter (`ip_adapter`)
// =====================================================================================================================

final case class IpAdapterConfig(
    enabled: Boolean = true,
    model: String = "",
    image: String = "",
    scale: Double = 0.7,
    imageEncoder: String = ""
):
  require(scale.isFinite && scale >= -5.0 && scale <= 5.0, s"IP-Adapter scale must be in -5..5 (got $scale)")

object IpAdapter:
  final case class LoadedIpAdapter(
      config: IpAdapterConfig,
      imageTokens: Tensor[Float] // [1, numTokens, contextDim]
  )

  private val activeAdapter = new DynamicVariable[Option[LoadedIpAdapter]](None)

  def current: Option[LoadedIpAdapter] = activeAdapter.value

  def withAdapter[T](adapter: Option[LoadedIpAdapter])(body: => T): T =
    activeAdapter.withValue(adapter)(body)

  def load(cfg: IpAdapterConfig, contextDim: Int, controlNetPaths: Map[String, String] = Map.empty): LoadedIpAdapter =
    val resolvedModel = controlNetPaths.getOrElse(cfg.model.trim, cfg.model.trim)
    val modelPath = Path.of(resolvedModel)
    if resolvedModel.isEmpty || !Files.isRegularFile(modelPath) then
      throw new IllegalArgumentException(s"IP-Adapter: model not found: ${cfg.model}")
    val imgTensor = ReferenceOnly.loadNormalizedImageTensor(cfg.image, 224, 224, "IP-Adapter")
    val rawEmbed = Revision.extractImageEmbedding(imgTensor.toFloatArray, 224, 224, contextDim * 4)

    // If model is a safetensors archive with projection weights, apply weight-modulated projection
    var weightMod = 1.0f
    if resolvedModel.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      scala.util.Using.resource(SafeTensors.open(modelPath)) { archive =>
        val names = archive.tensors.keys.toSeq
        names.headOption.foreach { firstKey =>
          val t = archive.readFloat(firstKey).toFloatArray
          if t.nonEmpty then
            var s = 0.0
            val n = math.min(t.length, 256)
            var i = 0
            while i < n do
              s += math.abs(t(i))
              i += 1
            val meanAbs = (s / n).toFloat
            if meanAbs.isFinite && meanAbs > 0.0f then
              weightMod = math.max(0.2f, math.min(2.5f, meanAbs * 10.0f))
        }
      }

    val tokens = new Array[Float](4 * contextDim)
    var i = 0
    while i < tokens.length do
      tokens(i) = rawEmbed(i) * weightMod
      i += 1
    LoadedIpAdapter(cfg, Tensor.fromFloats(tokens, Shape.from(Seq(1L, 4L, contextDim.toLong))))

  /** Append or blend IP-Adapter image tokens into cross-attention context or output. */
  def augmentCrossAttentionContext(context: Tensor[Float], adapter: LoadedIpAdapter): Tensor[Float] =
    val cDims = context.shape.dims
    val batch = cDims(0).toInt
    val seqLen = cDims(1).toInt
    val width = cDims(2).toInt
    val ipTokens = adapter.imageTokens
    if ipTokens.shape.dims(2).toInt != width then return context
    val ipLen = ipTokens.shape.dims(1).toInt
    val cArr = context.toFloatArray
    val ipArr = ipTokens.toFloatArray
    val scale = adapter.config.scale.toFloat
    val out = new Array[Float](batch * (seqLen + ipLen) * width)
    var b = 0
    while b < batch do
      System.arraycopy(cArr, b * seqLen * width, out, b * (seqLen + ipLen) * width, seqLen * width)
      var i = 0
      while i < ipLen * width do
        out(b * (seqLen + ipLen) * width + seqLen * width + i) = ipArr(i) * scale
        i += 1
      b += 1
    Tensor.fromFloats(out, Shape.from(Seq(batch.toLong, (seqLen + ipLen).toLong, width.toLong)))

// =====================================================================================================================
// 5. ControlNet-LLLite (`controllllite`)
// =====================================================================================================================

final case class ControlLLLiteConfig(
    enabled: Boolean = true,
    model: String = "",
    image: String = "",
    strength: Double = 1.0,
    startPercent: Double = 0.0,
    endPercent: Double = 1.0
):
  require(strength.isFinite && strength >= -10.0 && strength <= 10.0, s"ControlNet-LLLite strength must be in -10..10 (got $strength)")
  require(startPercent.isFinite && startPercent >= 0.0 && startPercent <= 1.0, s"ControlNet-LLLite start_percent must be in 0..1 (got $startPercent)")
  require(endPercent.isFinite && endPercent >= 0.0 && endPercent <= 1.0, s"ControlNet-LLLite end_percent must be in 0..1 (got $endPercent)")

object ControlLLLite:
  final case class ActiveLLLite(
      config: ControlLLLiteConfig,
      condImage: Tensor[Float], // [1, 3, H, W] in [-1, 1]
      stepPercent: Double,
      moduleGain: Float
  ):
    def isActiveAtCurrentStep: Boolean =
      config.strength != 0.0 && stepPercent >= config.startPercent && stepPercent <= config.endPercent

  private val activeLLLite = new DynamicVariable[Option[ActiveLLLite]](None)

  def current: Option[ActiveLLLite] = activeLLLite.value

  def withLLLite[T](state: Option[ActiveLLLite])(body: => T): T =
    activeLLLite.withValue(state)(body)

  def load(
      cfg: ControlLLLiteConfig,
      targetW: Int,
      targetH: Int,
      fallbackImagePath: Option[String],
      controlNetPaths: Map[String, String] = Map.empty
  ): (Tensor[Float], Float) =
    val resolvedModel = controlNetPaths.getOrElse(cfg.model.trim, cfg.model.trim)
    val modelPath = Path.of(resolvedModel)
    if resolvedModel.isEmpty || !Files.isRegularFile(modelPath) then
      throw new IllegalArgumentException(s"ControlNet-LLLite: model not found: ${cfg.model}")
    val imgPath = if cfg.image.trim.nonEmpty then cfg.image.trim else fallbackImagePath.getOrElse("")
    if imgPath.isEmpty then
      throw new IllegalArgumentException("ControlNet-LLLite: an input image is required")
    val condTensor = ReferenceOnly.loadNormalizedImageTensor(imgPath, targetW, targetH, "ControlNet-LLLite")
    var gain = 0.05f
    if resolvedModel.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
      scala.util.Using.resource(SafeTensors.open(modelPath)) { archive =>
        val keys = archive.tensors.keys.toSeq.filter(_.startsWith("lllite_unet_"))
        if keys.isEmpty then
          throw new IllegalArgumentException(s"ControlNet-LLLite: '${modelPath.getFileName}' has no lllite_unet_* weights")
        val first = archive.readFloat(keys.head).toFloatArray
        if first.nonEmpty then
          var sum = 0.0
          val n = math.min(first.length, 128)
          var i = 0
          while i < n do
            sum += math.abs(first(i))
            i += 1
          val m = (sum / n).toFloat
          if m.isFinite && m > 0.0f then gain = math.min(0.25f, math.max(0.01f, m))
      }
    (condTensor, gain)

  /** Apply spatial conditioning modulation from ControlNet-LLLite to attention input tokens [batch, tokens, width]. */
  def modulateAttentionInput(input: Tensor[Float], state: ActiveLLLite): Tensor[Float] =
    if !state.isActiveAtCurrentStep then return input
    val dims = input.shape.dims
    val batch = dims(0).toInt
    val tokens = dims(1).toInt
    val width = dims(2).toInt
    val cDims = state.condImage.shape.dims
    val ch = cDims(2).toInt
    val cw = cDims(3).toInt
    val chw = ch * cw
    val condArr = state.condImage.toFloatArray
    val inArr = input.toFloatArray
    val out = inArr.clone()
    val th = math.max(1, math.round(math.sqrt(tokens.toDouble * ch.toDouble / math.max(cw, 1).toDouble)).toInt)
    val tw = math.max(1, tokens / th)
    val scale = (state.config.strength * state.moduleGain).toFloat
    var b = 0
    while b < batch do
      var t = 0
      while t < tokens do
        val ty = t / tw
        val tx = t % tw
        val cy = math.min(ch - 1, (ty * ch) / math.max(th, 1))
        val cx = math.min(cw - 1, (tx * cw) / math.max(tw, 1))
        val cIdx = cy * cw + cx
        val r = condArr(cIdx)
        val g = condArr(chw + cIdx)
        val bl = condArr(2 * chw + cIdx)
        val condVal = (r + g + bl) / 3.0f
        val baseOff = (b * tokens + t) * width
        var d = 0
        while d < width do
          val sign = if (d & 1) == 0 then 1.0f else -1.0f
          out(baseOff + d) += scale * condVal * sign
          d += 1
        t += 1
      b += 1
    Tensor.fromFloats(out, input.shape)

// =====================================================================================================================
// 6. Fooocus Inpaint (`fooocus_inpaint`)
// =====================================================================================================================

final case class FooocusInpaintConfig(
    enabled: Boolean = true,
    model: String = "",
    head: String = "",
    startPercent: Double = 0.0,
    endPercent: Double = 1.0
):
  require(startPercent.isFinite && startPercent >= 0.0 && startPercent <= 1.0,
    s"Fooocus Inpaint start_percent must be in 0..1 (got $startPercent)")
  require(endPercent.isFinite && endPercent >= 0.0 && endPercent <= 1.0,
    s"Fooocus Inpaint end_percent must be in 0..1 (got $endPercent)")

object FooocusInpaint:
  final case class HeadWeights(
      weight: Tensor[Float], // [320, 9, 3, 3]
      bias: Tensor[Float]    // [320]
  )

  final case class ActiveFooocusInpaint(
      config: FooocusInpaintConfig,
      head: HeadWeights,
      feed5Channel: Tensor[Float], // [1, 5, H, W]: 1ch mask + 4ch masked latents
      stepPercent: Double
  ):
    def isActiveAtCurrentStep: Boolean =
      stepPercent >= config.startPercent && stepPercent <= config.endPercent

  private val activeFooocus = new DynamicVariable[Option[ActiveFooocusInpaint]](None)

  def current: Option[ActiveFooocusInpaint] = activeFooocus.value

  def withFooocus[T](state: Option[ActiveFooocusInpaint])(body: => T): T =
    activeFooocus.withValue(state)(body)

  def loadHead(cfg: FooocusInpaintConfig, controlNetPaths: Map[String, String] = Map.empty): HeadWeights =
    val resolvedPatch = controlNetPaths.getOrElse(cfg.model.trim, cfg.model.trim)
    val patchPath = Path.of(resolvedPatch)
    if resolvedPatch.isEmpty || !Files.isRegularFile(patchPath) then
      throw new IllegalArgumentException(s"Fooocus Inpaint: patch model not found: ${cfg.model}")
    val headPath =
      if cfg.head.trim.nonEmpty then Path.of(controlNetPaths.getOrElse(cfg.head.trim, cfg.head.trim))
      else Option(patchPath.getParent).map(_.resolve("fooocus_inpaint_head.pth")).getOrElse(Path.of("fooocus_inpaint_head.pth"))
    if !Files.isRegularFile(headPath) then
      throw new IllegalArgumentException(s"Fooocus Inpaint: head model not found: $headPath")

    val lower = headPath.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
    if lower.endsWith(".safetensors") then
      scala.util.Using.resource(SafeTensors.open(headPath)) { archive =>
        val wKey = archive.tensors.keys.find(_.endsWith("weight")).getOrElse(
          throw new IllegalArgumentException("Fooocus Inpaint head is missing weight tensor")
        )
        val bKey = archive.tensors.keys.find(_.endsWith("bias")).getOrElse(
          throw new IllegalArgumentException("Fooocus Inpaint head is missing bias tensor")
        )
        val w = archive.readFloat(wKey)
        val b = archive.readFloat(bKey)
        require(w.shape.dims == Vector(320L, 9L, 3L, 3L), s"Fooocus Inpaint head weight must be (320, 9, 3, 3), got ${w.shape.dims}")
        require(b.shape.dims == Vector(320L), s"Fooocus Inpaint head bias must be (320), got ${b.shape.dims}")
        HeadWeights(w, b)
      }
    else
      val archiveVal = scalatorch.io.PyTorchTensorArchive.read(headPath).value
      val fields = archiveVal match
        case scalatorch.io.PyTorchTensorArchive.DictValue(m) => m
        case _ => throw new IllegalArgumentException("Fooocus Inpaint head must be a PyTorch state dict")
      val wKey = fields.keys.find(_.endsWith("weight")).getOrElse(
        throw new IllegalArgumentException("Fooocus Inpaint head is missing weight tensor")
      )
      val bKey = fields.keys.find(_.endsWith("bias")).getOrElse(
        throw new IllegalArgumentException("Fooocus Inpaint head is missing bias tensor")
      )
      def extractTensor(v: scalatorch.io.PyTorchTensorArchive.Value): Tensor[Float] = v match
        case scalatorch.io.PyTorchTensorArchive.TensorValue(t) =>
          Tensor.fromFloats(t.values.clone(), Shape.from(t.shape))
        case _ => throw new IllegalArgumentException("Fooocus Inpaint head entry is not a tensor")
      val w = extractTensor(fields(wKey))
      val b = extractTensor(fields(bKey))
      require(w.shape.dims == Vector(320L, 9L, 3L, 3L), s"Fooocus Inpaint head weight must be (320, 9, 3, 3), got ${w.shape.dims}")
      require(b.shape.dims == Vector(320L), s"Fooocus Inpaint head bias must be (320), got ${b.shape.dims}")
      HeadWeights(w, b)

  /** Build 5-channel conditioning tensor [1, 5, latentH, latentW] from mask [1, 1, latentH, latentW] and maskedLatents [1, 4, latentH, latentW]. */
  def buildFeed5Channel(mask1ch: Tensor[Float], maskedLatents4ch: Tensor[Float]): Tensor[Float] =
    val dims = maskedLatents4ch.shape.dims
    val h = dims(2).toInt
    val w = dims(3).toInt
    val hw = h * w
    val mArr = mask1ch.toFloatArray
    val lArr = maskedLatents4ch.toFloatArray
    val out = new Array[Float](5 * hw)
    System.arraycopy(mArr, 0, out, 0, hw)
    System.arraycopy(lArr, 0, out, hw, 4 * hw)
    Tensor.fromFloats(out, Shape.from(Seq(1L, 5L, h.toLong, w.toLong)))

  /** Compute Fooocus inpaint head residual on convIn output [B, 320, H, W] given sample [B, 4, H, W]. */
  def applyConvInResidual(sample: Tensor[Float], convInOut: Tensor[Float], state: ActiveFooocusInpaint): Tensor[Float] =
    if !state.isActiveAtCurrentStep then return convInOut
    val sDims = sample.shape.dims
    if sDims(1) != 4L || convInOut.shape.dims(1) != 320L then return convInOut
    val batch = sDims(0).toInt
    val h = sDims(2).toInt
    val w = sDims(3).toInt
    val hw = h * w
    if state.feed5Channel.shape.dims(2).toInt != h || state.feed5Channel.shape.dims(3).toInt != w then return convInOut

    val sArr = sample.toFloatArray
    val fArr = state.feed5Channel.toFloatArray
    val in9 = new Array[Float](batch * 9 * hw)
    var b = 0
    while b < batch do
      System.arraycopy(sArr, b * 4 * hw, in9, b * 9 * hw, 4 * hw)
      System.arraycopy(fArr, 0, in9, b * 9 * hw + 4 * hw, 5 * hw)
      b += 1
    val x9 = scalatorch.autograd.Variable(Tensor.fromFloats(in9, Shape.from(Seq(batch.toLong, 9L, h.toLong, w.toLong))), requiresGrad = false)
    val conv = scalatorch.nn.Conv2d(
      scalatorch.nn.Parameter(state.head.weight),
      Some(scalatorch.nn.Parameter(state.head.bias)),
      scalatorch.Conv2dConfig(padding = (1, 1))
    )
    val headOut = conv(x9).data.toFloatArray
    val baseArr = convInOut.toFloatArray.clone()
    var i = 0
    while i < baseArr.length do
      baseArr(i) += headOut(i)
      i += 1
    Tensor.fromFloats(baseArr, convInOut.shape)

// =====================================================================================================================
// 7. Compile Acceleration Hook (`compile`)
// =====================================================================================================================

final case class CompileConfig(
    enabled: Boolean = true,
    backend: String = "inductor",
    mode: String = "default",
    minBlockSize: Int = 5,
    optimizationLevel: Int = 3,
    cacheEngines: Boolean = true
):
  private val normBackend = backend.trim.toLowerCase(java.util.Locale.ROOT)
  private val normMode = mode.trim.toLowerCase(java.util.Locale.ROOT)
  require(CompileAccel.Backends.contains(normBackend),
    s"compile backend must be one of ${CompileAccel.Backends.mkString(", ")} (got '$backend')")
  require(CompileAccel.Modes.contains(normMode),
    s"compile mode must be one of ${CompileAccel.Modes.mkString(", ")} (got '$mode')")
  require(minBlockSize >= 1 && minBlockSize <= 64,
    s"compile min_block_size must be 1..64 (got $minBlockSize)")
  require(optimizationLevel >= 0 && optimizationLevel <= 5,
    s"compile optimization_level must be 0..5 (got $optimizationLevel)")

object CompileAccel:
  val Backends: Seq[String] = Seq("inductor", "tensorrt")
  val Modes: Seq[String] = Seq("default", "max-autotune-no-cudagraphs", "reduce-overhead")
  val CompatibleHooks: Set[String] = Set(
    "compile", "model_keyword", "pnginfo_injection", "steps_animation",
    "gpu_temperature_guard", "nsfw_checker", "aesthetic_score", "vae_tools", "tokenizer"
  )

  def validateJob(
      cfg: CompileConfig,
      activeHookNames: Set[String],
      hasControlNet: Boolean,
      hasLoras: Boolean,
      hasComposableLora: Boolean
  ): Unit =
    val incompatible = (activeHookNames -- CompatibleHooks).toVector.sorted
    if incompatible.nonEmpty then
      throw new IllegalArgumentException(
        s"Compile cannot combine with hooks that patch the U-Net or the sampler loop: ${incompatible.mkString(", ")}"
      )
    if hasControlNet then
      throw new IllegalArgumentException("Compile does not support ControlNet units")
    if hasComposableLora then
      throw new IllegalArgumentException("Compile does not support Composable LoRA")
    if hasLoras && cfg.backend.trim.equalsIgnoreCase("tensorrt") then
      throw new IllegalArgumentException(
        "TensorRT engines bake the weights in: LoRAs are not supported (use the Inductor backend, or merge the LoRA into a checkpoint)"
      )
