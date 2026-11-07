package scalatorch.models

import scalatorch.Tensor
import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.UUID
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import javax.imageio.metadata.IIOMetadataNode

/** Intermediate sampling frame recorder (`save_intermediate`) and Steps Animation (`steps_animation`),
  * plus DDIM inversion (`img2imgalt`) and CD Tuner (`cd_tuner`) helpers.
  */
final case class SaveIntermediateConfig(
    everyN: Int = 5,
    startAt: Int = 0,
    stopAt: Int = 0,
    imageType: String = "Denoised",
    mode: String = "Standard",
    finalSave: Boolean = false,
    format: String = "mp4",
    codec: String = "h264",
    fps: Double = 2.0,
    addFirstFrames: Int = 0,
    addLastFrames: Int = 0,
    smooth: Boolean = false,
    seconds: Double = 0.0,
    lores: Boolean = true,
    hires: Boolean = true,
    ffmpegBat: Boolean = false,
    batOnly: Boolean = false,
    outputDirOverride: Option[String] = None
):
  def isSelected(step: Int): Boolean =
    step > 0 && step % everyN == 0 && step >= startAt && (stopAt == 0 || step <= stopAt)

object SaveIntermediateConfig:
  val ValidImageTypes: Set[String] = Set("Denoised", "Noisy", "Preview")
  val ValidModes: Set[String] = Set("Standard", "Suffix", "Video")
  val ValidFormats: Set[String] = Set("gif", "webm", "mp4")
  val ValidCodecs: Set[String] = Set("h264", "h265/hevc", "av1")

  def validate(raw: SaveIntermediateConfig): SaveIntermediateConfig =
    require(raw.everyN >= 1, "save intermediate: every_n must be a positive integer")
    require(raw.startAt >= 0, "save intermediate: start_at must be a nonnegative integer")
    require(raw.stopAt >= 0, "save intermediate: stop_at must be a nonnegative integer")
    require(raw.addFirstFrames >= 0, "save intermediate: add_first_frames must be a nonnegative integer")
    require(raw.addLastFrames >= 0, "save intermediate: add_last_frames must be a nonnegative integer")
    require(raw.fps.isFinite && raw.fps >= 0.01, "save intermediate: invalid fps")
    require(raw.seconds.isFinite && raw.seconds >= 0.0, "save intermediate: invalid seconds")
    require(ValidImageTypes.contains(raw.imageType), "save intermediate: invalid image_type")
    require(ValidModes.contains(raw.mode), "save intermediate: invalid mode")
    require(ValidFormats.contains(raw.format), "save intermediate: invalid format")
    require(ValidCodecs.contains(raw.codec), "save intermediate: invalid codec")
    require(!raw.smooth || raw.fps > 1.0, "save intermediate: smoothing requires output fps greater than 1")
    require(raw.mode != "Video" || raw.lores || raw.hires, "save intermediate: select base or hires frames for video")
    raw.copy(
      startAt = (raw.startAt / raw.everyN) * raw.everyN,
      stopAt = (raw.stopAt / raw.everyN) * raw.everyN
    )

final case class StepsAnimationConfig(
    duration: Double = 5.0,
    fps: Int = 30,
    codec: String = "h264",
    interpolation: String = "blend",
    skipInitial: Int = 0,
    debug: Boolean = false,
    runIncomplete: Boolean = false,
    deleteIntermediate: Boolean = false,
    createOutput: Boolean = true,
    framesDir: String = "steps-animation/frames",
    outputDir: String = "steps-animation/videos"
)

object StepsAnimationConfig:
  val ValidCodecs: Set[String] = Set("h264", "h265", "vp9", "av1", "prores")
  val ValidInterpolations: Set[String] = Set("none", "blend", "mci")

  def validateRelativePath(key: String, raw: String): String =
    require(raw.nonEmpty && !raw.exists(_.toInt < 32), s"steps animation: invalid $key")
    val normalized = raw.replace('\\', '/')
    require(!normalized.startsWith("/") && !normalized.contains(":"), s"steps animation: invalid $key")
    val parts = normalized.split("/").filter(_.nonEmpty).toVector
    require(parts.nonEmpty && !parts.contains("..") && parts != Vector("."), s"steps animation: invalid $key")
    parts.mkString("/")

  def validate(raw: StepsAnimationConfig): StepsAnimationConfig =
    require(raw.duration.isFinite && raw.duration >= 0.01 && raw.duration <= 3600.0, "steps animation: invalid duration")
    require(raw.fps >= 1 && raw.fps <= 240, "steps animation: invalid fps")
    require(raw.skipInitial >= 0 && raw.skipInitial <= 10000, "steps animation: invalid skip_initial")
    require(ValidCodecs.contains(raw.codec), "steps animation: invalid codec")
    require(ValidInterpolations.contains(raw.interpolation), "steps animation: invalid interpolation")
    val cleanFrames = validateRelativePath("frames_dir", raw.framesDir)
    val cleanOutput = validateRelativePath("output_dir", raw.outputDir)
    raw.copy(framesDir = cleanFrames, outputDir = cleanOutput)

final case class IntermediateFrameRecord(
    path: String,
    seed: Long,
    passName: String,
    step: Option[Int],
    passStep: Option[Int],
    iteration: Int,
    batchIndex: Int,
    width: Int,
    height: Int,
    imageType: String
)

final case class IntermediateArtifact(
    kind: String,
    path: String,
    seed: Option[Long] = None,
    passName: Option[String] = None,
    step: Option[Int] = None,
    passStep: Option[Int] = None,
    iteration: Option[Int] = None,
    batchIndex: Option[Int] = None,
    width: Option[Int] = None,
    height: Option[Int] = None,
    imageType: Option[String] = None,
    frames: Option[Int] = None,
    format: Option[String] = None
)

object IntermediatePngWriter:
  def writePngWithMetadata(image: BufferedImage, path: Path, texts: Vector[(String, String)]): Unit =
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    val out = ImageIO.createImageOutputStream(path.toFile)
    try
      writer.setOutput(out)
      val metadata = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), writer.getDefaultWriteParam)
      if texts.nonEmpty then
        val root = new IIOMetadataNode("javax_imageio_png_1.0")
        val entries = new IIOMetadataNode("iTXt")
        texts.foreach { (key, value) =>
          val node = new IIOMetadataNode("iTXtEntry")
          node.setAttribute("keyword", key)
          node.setAttribute("compressionFlag", "FALSE")
          node.setAttribute("compressionMethod", "0")
          node.setAttribute("languageTag", "")
          node.setAttribute("translatedKeyword", "")
          node.setAttribute("text", value)
          entries.appendChild(node)
        }
        root.appendChild(entries)
        metadata.mergeTree("javax_imageio_png_1.0", root)
      writer.write(null, new IIOImage(image, null, metadata), writer.getDefaultWriteParam)
    finally
      writer.dispose()
      out.close()

final case class Img2ImgAltConfig(
    overridePrompt: String = "",
    overrideNegative: String = "",
    decodeCfgScale: Double = 1.0,
    randomness: Double = 0.0,
    inversionSteps: Int = 0
)

object Img2ImgAlt:
  /** DDIM inversion of clean scaled VAE latents `x0` over `steps` ascending timesteps. */
  def invertLatents(
      x0: Tensor[Float],
      steps: Int,
      cfgScale: Double,
      schedule: DiffusionNoiseSchedule,
      predictEps: (Tensor[Float], Int, Boolean) => Tensor[Float],
      cancelled: () => Boolean = () => false
  ): Tensor[Float] =
    require(steps >= 1 && steps <= 1000, "img2imgalt inversion_steps must be in 1..1000")
    require(cfgScale.isFinite, "img2imgalt decode_cfg_scale must be finite")
    require(schedule.alphasCumprod.nonEmpty, "img2imgalt requires alphasCumprod schedule")
    val numTrain = schedule.alphasCumprod.length
    val stepRatio = math.max(1, numTrain / steps)
    val timesteps = Vector.tabulate(steps)(i => math.min(numTrain - 1, i * stepRatio))
    var x = x0.toFloatArray.clone()
    val guided = math.abs(cfgScale - 1.0) > 1e-6
    for idx <- timesteps.indices do
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("DDIM inversion cancelled")
      val t = timesteps(idx)
      val tNext = if idx + 1 < timesteps.length then timesteps(idx + 1) else math.min(numTrain - 1, t + stepRatio)
      val alphaBarT = math.max(1e-7, schedule.alphasCumprod(t).toDouble)
      val alphaBarNext = math.max(1e-7, schedule.alphasCumprod(tNext).toDouble)
      val xTensor = Tensor.fromFloats(x.clone(), x0.shape)
      val epsArr =
        if guided then
          val u = predictEps(xTensor, t, false).toFloatArray
          val c = predictEps(xTensor, t, true).toFloatArray
          val comb = new Array[Float](u.length)
          val g = cfgScale.toFloat
          var i = 0
          while i < u.length do
            comb(i) = u(i) + g * (c(i) - u(i))
            i += 1
          comb
        else predictEps(xTensor, t, true).toFloatArray
      val sqrtAlphaT = math.sqrt(alphaBarT)
      val sqrtOneMinusAlphaT = math.sqrt(math.max(0.0, 1.0 - alphaBarT))
      val sqrtAlphaNext = math.sqrt(alphaBarNext)
      val sqrtOneMinusAlphaNext = math.sqrt(math.max(0.0, 1.0 - alphaBarNext))
      val nextX = new Array[Float](x.length)
      var i = 0
      while i < x.length do
        val predX0 = (x(i) - sqrtOneMinusAlphaT * epsArr(i)) / sqrtAlphaT
        nextX(i) = (sqrtAlphaNext * predX0 + sqrtOneMinusAlphaNext * epsArr(i)).toFloat
        i += 1
      x = nextX
    Tensor.fromFloats(x, x0.shape)

  def mixRandomness(inverted: Tensor[Float], noise: Tensor[Float], randomness: Double): Tensor[Float] =
    require(randomness.isFinite && randomness >= 0.0 && randomness <= 1.0, "img2imgalt randomness must be in [0, 1]")
    if randomness <= 0.0 then inverted
    else
      val r = randomness
      val norm = math.sqrt((1.0 - r) * (1.0 - r) + r * r)
      val a = ((1.0 - r) / norm).toFloat
      val b = (r / norm).toFloat
      val invArr = inverted.toFloatArray
      val rndArr = noise.toFloatArray
      val out = new Array[Float](invArr.length)
      var i = 0
      while i < out.length do
        out(i) = a * invArr(i) + b * rndArr(i)
        i += 1
      Tensor.fromFloats(out, inverted.shape)

final case class CdTunerConfig(
    d1: Double = 0.0,
    d2: Double = 0.0,
    con1: Double = 0.0,
    con2: Double = 0.0,
    bri: Double = 0.0,
    col1: Double = 0.0,
    col2: Double = 0.0,
    col3: Double = 0.0,
    hd1: Double = 0.0,
    hd2: Double = 0.0,
    hrs: Double = 0.0,
    st1: Int = -1,
    st2: Int = -1,
    dis: Int = 0,
    sat: Double = 0.0,
    sat2: Double = 0.0,
    splitRatio: String = "1,1",
    splitBy: String = "Horizontal",
    colors: String = "",
    mapStopStep: Int = 2,
    mapStrength: Double = 1.0,
    applyOnce: Boolean = false
)

object CdTuner:
  private val ColorTableSd1: Array[Array[Double]] = Array(
    Array(-1.0, 1.0 / 3.0, 2.0 / 3.0),
    Array(1.0, 1.0, 0.0),
    Array(0.0, -1.0, -1.0),
    Array(1.0, 0.0, 1.0)
  )
  private val ColorTableSdxl: Array[Array[Double]] = Array(
    Array(0.0, 0.0, 1.0),
    Array(1.0, 0.0, 0.0),
    Array(-1.0, -1.0, 0.0),
    Array(-1.0, 1.0, 0.0)
  )

  def colorOffsets(cols: Array[Double], sdxl: Boolean): Array[Float] =
    val table = if sdxl then ColorTableSdxl else ColorTableSd1
    Array.tabulate(3) { j =>
      var sum = 0.0
      var i = 0
      while i < 4 && i < cols.length do
        sum += table(i)(j) * cols(i) * 0.02
        i += 1
      sum.toFloat
    }

  /** Apply CD Tuner denoised latent shift at step `stepIndex` (0-based) of `totalSteps`. */
  def applyStepShift(
      denoised: Tensor[Float],
      cfg: CdTunerConfig,
      sdxl: Boolean,
      stepIndex: Int,
      totalSteps: Int,
      isHires: Boolean
  ): Tensor[Float] =
    val b = denoised.shape.dims(0).toInt
    val c = denoised.shape.dims(1).toInt
    val h = denoised.shape.dims(2).toInt
    val w = denoised.shape.dims(3).toInt
    val spatial = h * w
    if c < 4 then return denoised
    val arr = denoised.toFloatArray.clone()

    // 1. Detail / Contrast 1 adjustment during active step window
    val stopStep = if isHires then cfg.st2 else cfg.st1
    val dActive = stopStep < 0 || stepIndex < stopStep
    val effD1 = if isHires && cfg.hd1 != 0.0 then cfg.hd1 else cfg.d1
    val effD2 = if isHires && cfg.hd2 != 0.0 then cfg.hd2 else cfg.d2
    if dActive && (effD1 != 0.0 || effD2 != 0.0 || cfg.con1 != 0.0) then
      val detailScale = (1.0 + (effD1 + effD2) * 0.002).toFloat
      val con1Offset = (cfg.con1 * 0.02 / math.max(1, totalSteps)).toFloat
      for bi <- 0 until b do
        for ch <- 0 until c do
          val base = (bi * c + ch) * spatial
          var mean = 0.0
          var p = 0
          while p < spatial do
            mean += arr(base + p)
            p += 1
          mean /= spatial
          val mF = mean.toFloat
          p = 0
          while p < spatial do
            val centered = arr(base + p) - mF
            arr(base + p) = mF + centered * detailScale + (if ch == 0 then con1Offset else 0.0f)
            p += 1

    // 2. Contrast 2, Brightness, and Color axes at target step `max(0, totalSteps - 2 - dis)`
    val targetStep = math.max(0, totalSteps - 2 - cfg.dis)
    if stepIndex == targetStep && (cfg.con2 != 0.0 || cfg.bri != 0.0 || cfg.col1 != 0.0 || cfg.col2 != 0.0 || cfg.col3 != 0.0) then
      val c0Shift = (-20.0 / 3.0 * (cfg.con2 / 50.0)).toFloat
      val cols = colorOffsets(Array(cfg.bri, cfg.col1, cfg.col2, cfg.col3), sdxl)
      val shifts = Array(-c0Shift, -cols(0), -cols(1), -cols(2))
      for bi <- 0 until b; ch <- 0 until 4 do
        val delta = shifts(ch)
        if delta != 0.0f then
          val base = (bi * c + ch) * spatial
          var p = 0
          while p < spatial do
            arr(base + p) += delta
            p += 1

    // 3. Regional Color Map during first `mapStopStep` steps
    if cfg.colors.trim.nonEmpty && cfg.mapStrength != 0.0 && stepIndex < cfg.mapStopStep then
      val (outerSpans, innerSpans) = makeCells(cfg.splitRatio)
      val regionColors = parseColors(cfg.colors)
      val hrAtt = if isHires then 0.25 else 1.0
      val satt = if cfg.mapStopStep < 5 then 1.15 - cfg.mapStopStep * 0.15 else 0.4
      val att = cfg.mapStrength * 0.5
      val vertical = !cfg.splitBy.contains("Hor")
      var cellIdx = 0
      for (ocell, i) <- outerSpans.zipWithIndex; icell <- innerSpans(i) do
        if cellIdx < regionColors.length then
          val vec = colorOffsets(regionColors(cellIdx), sdxl)
          val cratio = (vec.map(v => math.abs(v * 50.0)).sum / 10.0) *
            (1.0 / (1.0 + math.pow(1.0 + stepIndex, 1.5) / 10.0)) * att * hrAtt * satt
          if cratio > 0.0 then
            val (r0, r1, c0, c1) =
              if vertical then
                ((h * ocell._1).toInt, (h * ocell._2).toInt, (w * icell._1).toInt, (w * icell._2).toInt)
              else
                ((h * icell._1).toInt, (h * icell._2).toInt, (w * ocell._1).toInt, (w * ocell._2).toInt)
            val blend = (1.0 - math.min(0.95, cratio)).toFloat
            for bi <- 0 until b; ch <- 1 to 3 do
              val base = (bi * c + ch) * spatial
              var absMean = 0.0
              var p = 0
              while p < spatial do
                absMean += math.abs(arr(base + p))
                p += 1
              absMean /= math.max(1, spatial)
              val shift = (vec(ch - 1) * 6.0 * absMean * cratio).toFloat
              for row <- math.max(0, r0) until math.min(h, r1); col <- math.max(0, c0) until math.min(w, c1) do
                val idx = base + row * w + col
                arr(idx) = blend * arr(idx) - shift
        cellIdx += 1

    // 4. Saturation adjustment on final step
    if (cfg.sat != 0.0 || cfg.sat2 != 0.0) && stepIndex == totalSteps - 1 then
      val satScale = ((1.0 + cfg.sat * 0.075) * (1.0 + cfg.sat2 * 0.02)).toFloat
      for bi <- 0 until b; ch <- 1 to 3 do
        val base = (bi * c + ch) * spatial
        var p = 0
        while p < spatial do
          arr(base + p) *= satScale
          p += 1

    Tensor.fromFloats(arr, denoised.shape)

  def makeCells(ratio: String): (Vector[(Double, Double)], Vector[Vector[(Double, Double)]]) =
    val text = ratio.replace(',', ' ')
    val rawRows = (if text.contains('|') then text.split('|') else text.split(';')).map(_.trim).toVector
    val rows = if rawRows.size == 1 then Vector("1 " + rawRows.head) else rawRows
    def spans(weights: Vector[Double]): Vector[(Double, Double)] =
      val total = weights.sum
      require(total != 0.0, "CD Tuner: Split Ratio values must not sum to 0")
      var start = 0.0
      weights.map { w =>
        val next = start + w / total
        val pair = (start, next)
        start = next
        pair
      }
    val parsed = rows.map { row =>
      val nums = try row.split("\\s+").filter(_.nonEmpty).map(_.toDouble).toVector
      catch case _: NumberFormatException =>
        throw IllegalArgumentException(s"CD Tuner: Split Ratio '$ratio' must be numbers separated by single commas or spaces")
      require(nums.nonEmpty && nums.forall(_.isFinite), s"CD Tuner: invalid Split Ratio '$ratio'")
      if nums.size == 1 then Vector(nums.head, nums.head) else nums
    }
    val outer = spans(parsed.map(_.head))
    val inner = parsed.map(v => spans(v.tail))
    (outer, inner)

  def parseColors(text: String): Vector[Array[Double]] =
    val regions = (if text.contains('|') then text.split('|') else text.split(';')).map(_.trim).toVector
    regions.map { reg =>
      val parts = if reg.contains(',') then reg.split(',') else reg.split("\\s+")
      val raw = try parts.map(_.trim).filter(_.nonEmpty).map(_.toDouble).toVector
      catch case _: NumberFormatException =>
        throw IllegalArgumentException(s"CD Tuner: colors region '$reg' must be 3 or 4 numbers separated by spaces or commas")
      require(raw.forall(_.isFinite), "CD Tuner: colors must be finite numbers")
      val four =
        if raw.size == 3 then (raw.sum / 3.0) +: raw
        else if raw.size == 4 then raw
        else throw IllegalArgumentException(s"CD Tuner: colors region '$reg' must have 3 or 4 values")
      four.toArray
    }

  def validate(cfg: CdTunerConfig): CdTunerConfig =
    if cfg.colors.trim.nonEmpty then
      val (_, inner) = makeCells(cfg.splitRatio)
      val cols = parseColors(cfg.colors)
      val needed = inner.map(_.size).sum
      require(cols.size >= needed, s"CD Tuner: Split Ratio has $needed regions but colors gives ${cols.size}")
    cfg
