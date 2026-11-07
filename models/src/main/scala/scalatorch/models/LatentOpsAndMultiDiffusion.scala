package scalatorch.models

import scalatorch.*
import java.awt.image.BufferedImage

final case class LatentModifierConfig(
    sharpnessMultiplier: Double = 0.0,
    sharpnessMethod: String = "anisotropic",
    tonemapMultiplier: Double = 0.0,
    tonemapMethod: String = "reinhard",
    tonemapPercentile: Double = 100.0,
    contrastMultiplier: Double = 0.0,
    combatMethod: String = "subtract",
    combatCfgDrift: Double = 0.0,
    extraNoiseType: String = "gaussian",
    extraNoiseMultiplier: Double = 0.0,
    extraNoiseLowpass: Int = 100,
    divisiveNormSize: Int = 127,
    divisiveNormMultiplier: Double = 0.0,
    spectralModMode: String = "hard_clamp",
    spectralModPercentile: Double = 5.0,
    spectralModMultiplier: Double = 0.0
):
  require(
    sharpnessMultiplier.isFinite && tonemapMultiplier.isFinite && tonemapPercentile.isFinite &&
      contrastMultiplier.isFinite && combatCfgDrift.isFinite && extraNoiseMultiplier.isFinite &&
      divisiveNormMultiplier.isFinite && spectralModPercentile.isFinite && spectralModMultiplier.isFinite,
    "invalid latent_modifier parameters"
  )

object LatentModifier:
  private def boxBlur(values: Array[Float], batch: Int, channels: Int, h: Int, w: Int, rawK: Int): Array[Float] =
    val k = math.max(1, rawK | 1)
    val pad = k / 2
    val out = new Array[Float](values.length)
    val plane = h * w
    val inv = 1.0f / (k * k).toFloat
    for b <- 0 until batch; c <- 0 until channels do
      val base = (b * channels + c) * plane
      for y <- 0 until h; x <- 0 until w do
        var sum = 0.0f
        var dy = -pad
        while dy <= pad do
          val sy = math.max(0, math.min(h - 1, y + dy))
          var dx = -pad
          while dx <= pad do
            val sx = math.max(0, math.min(w - 1, x + dx))
            sum += values(base + sy * w + sx)
            dx += 1
          dy += 1
        out(base + y * w + x) = sum * inv
    out

  private def gaussianBlur(values: Array[Float], batch: Int, channels: Int, h: Int, w: Int, sigma: Double): Array[Float] =
    val r = math.max(1, math.ceil(sigma * 2.0).toInt)
    val kernel = Array.tabulate(2 * r + 1) { i =>
      val x = (i - r).toDouble
      math.exp(-(x * x) / (2.0 * sigma * sigma))
    }
    val kSum = kernel.sum
    val kNorm = kernel.map(v => (v / kSum).toFloat)
    val reflect = math.min(h, w) > r
    def coord(pos: Int, limit: Int): Int =
      if pos >= 0 && pos < limit then pos
      else if !reflect || limit <= 1 then math.max(0, math.min(limit - 1, pos))
      else if pos < 0 then math.min(limit - 1, -pos)
      else math.max(0, 2 * limit - 2 - pos)

    val temp = new Array[Float](values.length)
    val out = new Array[Float](values.length)
    val plane = h * w
    for b <- 0 until batch; c <- 0 until channels do
      val base = (b * channels + c) * plane
      for y <- 0 until h; x <- 0 until w do
        var acc = 0.0f
        var d = -r
        while d <= r do
          val sx = coord(x + d, w)
          acc += values(base + y * w + sx) * kNorm(d + r)
          d += 1
        temp(base + y * w + x) = acc
      for y <- 0 until h; x <- 0 until w do
        var acc = 0.0f
        var d = -r
        while d <= r do
          val sy = coord(y + d, h)
          acc += temp(base + sy * w + x) * kNorm(d + r)
          d += 1
        out(base + y * w + x) = acc
    out

  private def quantile(sorted: Array[Double], q: Double): Double =
    if sorted.isEmpty then 0.0
    else
      val clamped = math.max(0.0, math.min(1.0, q))
      if clamped <= 0.0 then sorted(0)
      else if clamped >= 1.0 then sorted(sorted.length - 1)
      else
        val pos = clamped * (sorted.length - 1)
        val lo = math.floor(pos).toInt
        val hi = math.ceil(pos).toInt
        val frac = pos - lo
        sorted(lo) * (1.0 - frac) + sorted(hi) * frac

  def modifyLatents(
      x: Tensor[Float],
      cfg: LatentModifierConfig,
      progress: Double,
      rng: Option[TorchCpuNormal] = None
  ): Tensor[Float] =
    val dims = x.shape.dims
    if dims.size != 4 then return x
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val h = dims(2).toInt
    val w = dims(3).toInt
    val plane = h * w
    var y = x.toFloatArray.clone()

    // 1. Sharpness (unsharp mask)
    if cfg.sharpnessMultiplier != 0.0 then
      val sigma = if cfg.sharpnessMethod != "gaussian" then 1.0 else 2.0
      val blurred = gaussianBlur(y, batch, channels, h, w, sigma)
      val scale = (cfg.sharpnessMultiplier * (1.0 - progress)).toFloat
      var i = 0
      while i < y.length do
        y(i) = y(i) + (y(i) - blurred(i)) * scale
        i += 1

    // 2. Contrast around per-channel mean
    if cfg.contrastMultiplier != 0.0 then
      val mult = (1.0 + cfg.contrastMultiplier).toFloat
      for b <- 0 until batch; c <- 0 until channels do
        val base = (b * channels + c) * plane
        var sum = 0.0
        var i = 0
        while i < plane do
          sum += y(base + i)
          i += 1
        val mean = (sum / plane).toFloat
        i = 0
        while i < plane do
          y(base + i) = mean + (y(base + i) - mean) * mult
          i += 1

    // 3. Tonemapping of large magnitudes across channels
    if cfg.tonemapMultiplier != 0.0 then
      val m = cfg.tonemapMultiplier
      val q = math.max(0.0, math.min(1.0, cfg.tonemapPercentile / 100.0))
      for b <- 0 until batch do
        val bBase = b * channels * plane
        val mag = new Array[Double](plane)
        var p = 0
        while p < plane do
          var sq = 0.0
          var c = 0
          while c < channels do
            val v = y(bBase + c * plane + p).toDouble
            sq += v * v
            c += 1
          mag(p) = math.sqrt(sq)
          p += 1
        val sorted = mag.clone()
        java.util.Arrays.sort(sorted)
        val top = math.max(1e-6, quantile(sorted, q))
        p = 0
        while p < plane do
          val mg = mag(p)
          val newMag = cfg.tonemapMethod match
            case "arctan" => top * (2.0 / math.Pi) * math.atan((mg / top) * (math.Pi / 2.0))
            case "clamp"  => math.min(mg, top)
            case _ =>
              val norm = mg / top
              top * norm / (1.0 + norm)
          val ratio = newMag / math.max(mg, 1e-6)
          val factor = (1.0 + (ratio - 1.0) * m).toFloat
          var c = 0
          while c < channels do
            val idx = bBase + c * plane + p
            y(idx) = y(idx) * factor
            c += 1
          p += 1

    // 4. Combat CFG drift
    if cfg.combatCfgDrift != 0.0 then
      val m = cfg.combatCfgDrift.toFloat
      for b <- 0 until batch; c <- 0 until channels do
        val base = (b * channels + c) * plane
        val center =
          if cfg.combatMethod == "median" then
            val slice = new Array[Float](plane)
            System.arraycopy(y, base, slice, 0, plane)
            java.util.Arrays.sort(slice)
            slice((plane - 1) / 2)
          else
            var sum = 0.0
            var i = 0
            while i < plane do
              sum += y(base + i)
              i += 1
            (sum / plane).toFloat
        var i = 0
        while i < plane do
          y(base + i) = y(base + i) - center * m
          i += 1

    // 5. Divisive normalization (local contrast)
    if cfg.divisiveNormMultiplier != 0.0 then
      val size = math.max(3, math.min(cfg.divisiveNormSize, 2 * math.min(h, w) - 1))
      val absY = y.map(v => math.abs(v))
      val local = boxBlur(absY, batch, channels, h, w, size).map(v => math.max(1e-4f, v))
      val m = cfg.divisiveNormMultiplier.toFloat
      for b <- 0 until batch; c <- 0 until channels do
        val base = (b * channels + c) * plane
        var sum = 0.0
        var i = 0
        while i < plane do
          sum += local(base + i)
          i += 1
        val localMean = (sum / plane).toFloat
        i = 0
        while i < plane do
          val idx = base + i
          val target = y(idx) / local(idx) * localMean
          y(idx) = y(idx) + (target - y(idx)) * m
          i += 1

    // 6. Extra noise (fades out)
    if cfg.extraNoiseMultiplier != 0.0 then
      val gen = rng.getOrElse(new TorchCpuNormal(0L))
      var n =
        if cfg.extraNoiseType == "uniform" then
          val raw = gen.normal(x.shape.numel.toInt)
          val s = (2.0 * math.sqrt(3.0)).toFloat
          raw.map(v => ((math.tanh(v.toDouble) * 0.5).toFloat) * s)
        else
          gen.normal(x.shape.numel.toInt)
      if cfg.extraNoiseLowpass > 0 && cfg.extraNoiseLowpass < 100 then
        val sigma = math.max(0.5, (100 - cfg.extraNoiseLowpass) / 25.0)
        n = gaussianBlur(n, batch, channels, h, w, sigma)
      var mean = 0.0
      var i = 0
      while i < y.length do
        mean += y(i)
        i += 1
      mean /= math.max(1, y.length)
      var sq = 0.0
      i = 0
      while i < y.length do
        val d = y(i) - mean
        sq += d * d
        i += 1
      val std = math.sqrt(sq / math.max(1, y.length - 1))
      val scale = (cfg.extraNoiseMultiplier * (1.0 - progress) * std).toFloat
      i = 0
      while i < y.length do
        y(i) = y(i) + n(i) * scale
        i += 1

    Tensor.fromFloats(y, x.shape)

final case class SoftInpaintingConfig(
    maskBlendPower: Double = 1.0,
    maskBlendScale: Double = 0.5,
    inpaintDetailPreservation: Double = 4.0,
    compositeMaskInfluence: Double = 0.0,
    compositeDifferenceThreshold: Double = 0.5,
    compositeDifferenceContrast: Double = 2.0
):
  require(
    maskBlendPower.isFinite && maskBlendScale.isFinite && inpaintDetailPreservation.isFinite &&
      compositeMaskInfluence.isFinite && compositeDifferenceThreshold.isFinite && compositeDifferenceContrast.isFinite,
    "invalid soft_inpainting parameters"
  )

object SoftInpainting:
  def softMaskThreshold(stepZeroBased: Int, totalSteps: Int, power: Double): Double =
    val p = (stepZeroBased + 1).toDouble / math.max(totalSteps, 1).toDouble
    math.pow(math.max(0.0, 1.0 - p), math.max(power, 1e-3))

  def resizeMaskBilinear(mask: Tensor[Float], targetH: Int, targetW: Int): Tensor[Float] =
    KohyaHrFix.interpolate2d(mask, targetH, targetW, "bilinear")

  def softInpaintBlend(
      latents: Tensor[Float],
      origNoised: Tensor[Float],
      softMask: Tensor[Float],
      stepZeroBased: Int,
      totalSteps: Int,
      cfg: SoftInpaintingConfig
  ): Tensor[Float] =
    require(latents.shape == origNoised.shape, "SoftInpainting latent shape mismatch")
    val dims = latents.shape.dims
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val h = dims(2).toInt
    val w = dims(3).toInt
    val plane = h * w
    val thr = softMaskThreshold(stepZeroBased, totalSteps, cfg.maskBlendPower)
    val width = math.max(1e-3, cfg.maskBlendScale / math.max(cfg.inpaintDetailPreservation, 1e-3))
    val mArr =
      if softMask.shape.dims(2).toInt == h && softMask.shape.dims(3).toInt == w then softMask.toFloatArray
      else resizeMaskBilinear(softMask, h, w).toFloatArray
    val latArr = latents.toFloatArray
    val origArr = origNoised.toFloatArray
    val out = new Array[Float](latArr.length)
    for b <- 0 until batch; c <- 0 until channels do
      val base = (b * channels + c) * plane
      var p = 0
      while p < plane do
        val keep = math.max(0.0, math.min(1.0, (mArr(p).toDouble - thr) / width + 0.5)).toFloat
        out(base + p) = latArr(base + p) * keep + origArr(base + p) * (1.0f - keep)
        p += 1
    Tensor.fromFloats(out, latents.shape)

  /** Final pixel composite in [-1, 1] NCHW space matching sdfx/latent_ops.py composite_mask. */
  def compositePixels(
      origImage: Tensor[Float],
      genImage: Tensor[Float],
      softMask: Tensor[Float],
      cfg: SoftInpaintingConfig
  ): Tensor[Float] =
    require(origImage.shape == genImage.shape, "SoftInpainting composite image shape mismatch")
    val dims = origImage.shape.dims
    val h = dims(2).toInt
    val w = dims(3).toInt
    val plane = h * w
    val mArr =
      if softMask.shape.dims(2).toInt == h && softMask.shape.dims(3).toInt == w then softMask.toFloatArray
      else resizeMaskBilinear(softMask, h, w).toFloatArray
    val origArr = origImage.toFloatArray
    val genArr = genImage.toFloatArray
    val out = new Array[Float](genArr.length)
    val thr = math.max(cfg.compositeDifferenceThreshold, 1e-6)
    val contrast = cfg.compositeDifferenceContrast
    val infl = cfg.compositeMaskInfluence
    var p = 0
    while p < plane do
      val oR = (origArr(p) + 1f) * 0.5f
      val oG = (origArr(plane + p) + 1f) * 0.5f
      val oB = (origArr(2 * plane + p) + 1f) * 0.5f
      val gR = (genArr(p) + 1f) * 0.5f
      val gG = (genArr(plane + p) + 1f) * 0.5f
      val gB = (genArr(2 * plane + p) + 1f) * 0.5f
      val diff = (math.abs(gR - oR) + math.abs(gG - oG) + math.abs(gB - oB)) / 3.0
      val boost = math.max(0.0, math.min(1.0, math.pow(diff / thr, contrast)))
      val sm = mArr(p).toDouble
      val m =
        if infl != 0.0 then sm * (1.0 - infl) + math.max(sm, boost) * infl
        else sm
      val mc = math.max(0.0, math.min(1.0, m)).toFloat
      out(p) = genArr(p) * mc + origArr(p) * (1.0f - mc)
      out(plane + p) = genArr(plane + p) * mc + origArr(plane + p) * (1.0f - mc)
      out(2 * plane + p) = genArr(2 * plane + p) * mc + origArr(2 * plane + p) * (1.0f - mc)
      p += 1
    Tensor.fromFloats(out, genImage.shape)

final case class MultiDiffusionConfig(
    method: String = "MultiDiffusion",
    tileWidth: Int = 768,
    tileHeight: Int = 768,
    tileOverlap: Int = 64,
    tileBatchSize: Int = 4
):
  require(tileWidth > 0 && tileHeight > 0 && tileOverlap >= 0 && tileBatchSize >= 1, "invalid multidiffusion parameters")

object MultiDiffusion:
  def tileStarts(size: Int, tile: Int, overlap: Int): Vector[Int] =
    if tile >= size then Vector(0)
    else
      val stride = math.max(1, tile - overlap)
      val n = math.ceil((size - tile).toDouble / stride.toDouble).toInt + 1
      if n == 1 then Vector(0)
      else
        val step = (size - tile).toDouble / (n - 1).toDouble
        (0 until n).map(i => math.round(i * step).toInt).distinct.sorted.toVector

  def tileGrid(h: Int, w: Int, thRaw: Int, twRaw: Int, overlap: Int): Vector[(Int, Int, Int, Int)] =
    val th = math.min(thRaw, h)
    val tw = math.min(twRaw, w)
    for
      y <- tileStarts(h, th, overlap)
      x <- tileStarts(w, tw, overlap)
    yield (y, y + th, x, x + tw)

  def gaussianWeights(th: Int, tw: Int): Array[Float] =
    val variance = 0.01
    def g(n: Int): Array[Double] =
      val mid = (n - 1) / 2.0
      val denom = math.max(n, 1).toDouble
      val norm = math.sqrt(2.0 * math.Pi * variance)
      Array.tabulate(n) { i =>
        val u = (i - mid) / denom
        math.exp(-(u * u) / (2.0 * variance)) / norm
      }
    val gy = g(th)
    val gx = g(tw)
    val peak = gy(gy.length / 2) * gx(gx.length / 2)
    val floor = peak * 1e-3
    val out = new Array[Float](th * tw)
    var y = 0
    while y < th do
      var x = 0
      while x < tw do
        out(y * tw + x) = math.max(gy(y) * gx(x), floor).toFloat
        x += 1
      y += 1
    out

  def sliceSpatial(tensor: Tensor[Float], y0: Int, y1: Int, x0: Int, x1: Int): Tensor[Float] =
    val dims = tensor.shape.dims
    val batch = dims(0).toInt
    val channels = dims(1).toInt
    val h = dims(2).toInt
    val w = dims(3).toInt
    val th = y1 - y0
    val tw = x1 - x0
    if y0 == 0 && y1 == h && x0 == 0 && x1 == w then tensor
    else
      val src = tensor.toFloatArray
      val out = new Array[Float](batch * channels * th * tw)
      val inPlane = h * w
      val outPlane = th * tw
      for b <- 0 until batch; c <- 0 until channels do
        val inBase = (b * channels + c) * inPlane
        val outBase = (b * channels + c) * outPlane
        var ty = 0
        while ty < th do
          System.arraycopy(src, inBase + (y0 + ty) * w + x0, out, outBase + ty * tw, tw)
          ty += 1
      Tensor.fromFloats(out, Shape.from(Seq(batch.toLong, channels.toLong, th.toLong, tw.toLong)))

  def sliceFeat(r: Tensor[Float], latentH: Int, y0: Int, y1: Int, x0: Int, x1: Int): Tensor[Float] =
    val rH = r.shape.dims(2).toInt
    val rW = r.shape.dims(3).toInt
    val s = rH.toDouble / latentH.toDouble
    val fy0 = (y0 * s).toInt.min(rH - 1)
    val fy1 = math.max(fy0 + 1, math.round(y1 * s).toInt.min(rH))
    val fx0 = (x0 * s).toInt.min(rW - 1)
    val fx1 = math.max(fx0 + 1, math.round(x1 * s).toInt.min(rW))
    sliceSpatial(r, fy0, fy1, fx0, fx1)

  def sliceResiduals(residuals: ControlNetResiduals, latentH: Int, y0: Int, y1: Int, x0: Int, x1: Int): ControlNetResiduals =
    ControlNetResiduals(
      residuals.down.map(r => sliceFeat(r, latentH, y0, y1, x0, x1)),
      sliceFeat(residuals.middle, latentH, y0, y1, x0, x1)
    )

final case class NeverOomConfig(
    unetOffload: Boolean = false,
    vaeTiling: Boolean = false,
    encoderTileSize: Int = 1024,
    decoderTileSize: Int = 128
)
