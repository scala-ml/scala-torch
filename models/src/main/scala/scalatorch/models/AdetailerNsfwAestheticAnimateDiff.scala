package scalatorch.models

import java.awt.{ BasicStroke, Color, RenderingHints }
import java.awt.geom.{ Ellipse2D, Rectangle2D, RoundRectangle2D }
import java.awt.image.BufferedImage
import java.io.{ ByteArrayOutputStream, File }
import java.nio.charset.StandardCharsets
import java.nio.file.{ Files, Path }
import java.security.MessageDigest
import java.util.Locale
import javax.imageio.{ IIOImage, ImageIO, ImageTypeSpecifier, ImageWriteParam }
import javax.imageio.metadata.{ IIOMetadata, IIOMetadataNode }
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scalatorch.{ Shape, Tensor }

// =====================================================================================================================
// 1. ADetailer (`adetailer`) & Batch Face Swap (`batch_face_swap`)
// =====================================================================================================================

final case class AdetailerUnit(
    model: String = "face_yolov8n.pt",
    prompt: String = "",
    negativePrompt: String = "",
    confidence: Double = 0.3,
    maskKLargest: Int = 0,
    maskMinRatio: Double = 0.0,
    maskMaxRatio: Double = 1.0,
    xOffset: Int = 0,
    yOffset: Int = 0,
    dilateErode: Int = 4,
    maskMergeInvert: String = "None",
    maskBlur: Int = 4,
    denoisingStrength: Double = 0.4,
    inpaintOnlyMasked: Boolean = true,
    inpaintPadding: Int = 32,
    useInpaintWidthHeight: Boolean = false,
    inpaintWidth: Int = 512,
    inpaintHeight: Int = 512,
    useSteps: Boolean = false,
    steps: Int = 28,
    useCfgScale: Boolean = false,
    cfgScale: Double = 7.0,
    useSampler: Boolean = false,
    sampler: String = "",
    useNoiseMultiplier: Boolean = false,
    noiseMultiplier: Double = 1.0,
    colorCorrection: Boolean = true,
    maskShape: String = "",
    maskScale: Double = 1.0
)

object Adetailer:
  final case class Detection(
      boxes: Vector[(Double, Double, Double, Double)], // (x1, y1, x2, y2)
      masks: Vector[Array[Int]],                      // grayscale 0..255 of size w * h
      confidences: Vector[Double],
      width: Int,
      height: Int,
      classes: Vector[Int] = Vector.empty // detector class index per box (NudeNet: index into NsfwChecker.Labels); empty for single-class detectors
  )

  val HfModels: Set[String] = Set(
    "face_yolov8n.pt", "face_yolov8s.pt", "face_yolov8m.pt", "face_yolov9c.pt",
    "hand_yolov8n.pt", "hand_yolov8s.pt", "hand_yolov9c.pt",
    "person_yolov8n-seg.pt", "person_yolov8s-seg.pt", "person_yolov8m-seg.pt",
    "deepfashion2_yolov8s-seg.pt", "opencv_face"
  )

  def boxMasks(width: Int, height: Int, boxes: Vector[(Double, Double, Double, Double)]): Vector[Array[Int]] =
    boxes.map { case (x1, y1, x2, y2) =>
      val m = new Array[Int](width * height)
      val ix1 = math.max(0, math.min(width - 1, x1.toInt))
      val iy1 = math.max(0, math.min(height - 1, y1.toInt))
      val ix2 = math.max(0, math.min(width - 1, x2.toInt))
      val iy2 = math.max(0, math.min(height - 1, y2.toInt))
      var y = iy1
      while y <= iy2 do
        var x = ix1
        while x <= ix2 do
          m(y * width + x) = 255
          x += 1
        y += 1
      m
    }

  /** Weights file of a trained detector (converted by scripts/convert_yolov8_to_safetensors.py), relative to the annotator folder. */
  private val YoloName = """((?:face|hand|person|deepfashion2)_yolov8[nsmlx](?:-seg)?)\.pt""".r
  private val yoloCache = new java.util.concurrent.ConcurrentHashMap[java.nio.file.Path, YoloV8]()

  /** Whether `model` names a detector this runtime implements (YOLOv8 face/hand/person/deepfashion2 checkpoints). */
  def isImplemented(model: String): Boolean = YoloName.matches(model) || model == NudeNetModel

  /** NudeNet v3 (320n) detector id and its converted weights, relative to the annotator folder. NudeNet and its weights are AGPL-3.0
    * (notAI-tech/NudeNet); this runtime re-implements the documented pre/post-processing and ships no upstream code. */
  val NudeNetModel = "nudenet"
  val NudeNetWeights = "nudenet/320n.safetensors"
  val NudeNetSize = 320
  val NudeNetNmsIou = 0.45

  /** NudeNet v3: zero-pad to a square, resize to 320, YOLOv8n with 18 classes (indices = NsfwChecker.Labels), best-class score >= `confidence`,
    * class-agnostic NMS at IoU 0.45. Boxes are box masks; `classes` carries the class index of each box. */
  private def detectNudeNet(image: BufferedImage, confidence: Double): Detection =
    val file = scalatorch.models.worker.NativeAnnotatorFiles.locate(NudeNetWeights, "NudeNet v3 320n")
    val net = yoloCache.computeIfAbsent(file.toAbsolutePath.normalize, f => YoloV8.load(f))
    require(net.spec.nc == NsfwChecker.Labels.size && !net.spec.segment, s"NudeNet weights must be an 18-class detector (got nc=${net.spec.nc})")
    val r = net.detect(image, confidence, iou = NudeNetNmsIou, imgsz = NudeNetSize, withMasks = false, squarePad = true, agnostic = true)
    Detection(r.boxes, boxMasks(image.getWidth, image.getHeight, r.boxes), r.scores, image.getWidth, image.getHeight, r.classes)

  /** Detect regions with a real YOLOv8 detector (face/hand/person; `-seg` models give instance masks, others box masks). The weights are
    * `<annotator dir>/adetailer/<name>.safetensors` (SDFX_SCALATORCH_ANNOTATOR_DIR or models/annotators); there is no heuristic fallback:
    * unavailable weights throw IllegalStateException and models without an implementation throw NativeBackendError("unsupported"). */
  def detect(image: BufferedImage, model: String, confidence: Double): Detection =
    if model == NudeNetModel then return detectNudeNet(image, confidence)
    val stem = model match
      case YoloName(s) => s
      case other =>
        throw new scalatorch.models.worker.NativeBackendError("unsupported",
          s"detection model '$other' is not implemented (no trained network or weights available in the native runtime); only YOLOv8 face/hand/person models and nudenet are supported")
    val file = scalatorch.models.worker.NativeAnnotatorFiles.locate(s"adetailer/$stem.safetensors", s"YOLOv8 $stem")
    val net = yoloCache.computeIfAbsent(file.toAbsolutePath.normalize, f => YoloV8.load(f))
    val r = net.detect(image, confidence, withMasks = stem.endsWith("-seg"))
    val boxes = r.boxes
    val masks = r.masks.getOrElse(boxMasks(image.getWidth, image.getHeight, boxes))
    Detection(boxes, masks, r.scores, image.getWidth, image.getHeight)

  private def keepClasses(d: Detection, idx: Seq[Int]): Vector[Int] = if d.classes.isEmpty then Vector.empty else idx.map(d.classes).toVector

  def filterByRatio(d: Detection, lo: Double, hi: Double): Detection =
    val area = math.max(1.0, (d.width * d.height).toDouble)
    val keep = d.boxes.indices.filter { i =>
      val (x1, y1, x2, y2) = d.boxes(i)
      val r = (x2 - x1) * (y2 - y1) / area
      r >= lo && r <= hi
    }.toVector
    Detection(keep.map(d.boxes), keep.map(d.masks), keep.map(d.confidences), d.width, d.height, keepClasses(d, keep))

  def filterKLargest(d: Detection, k: Int): Detection =
    if k <= 0 || d.boxes.size <= k then d
    else
      val order = d.boxes.indices.sortBy { i =>
        val (x1, y1, x2, y2) = d.boxes(i)
        -((x2 - x1) * (y2 - y1))
      }
      val keep = order.take(k).sorted.toVector
      Detection(keep.map(d.boxes), keep.map(d.masks), keep.map(d.confidences), d.width, d.height, keepClasses(d, keep))

  def sortLeftToRight(d: Detection): Detection =
    val order = d.boxes.indices.sortBy(i => (d.boxes(i)._1, d.boxes(i)._2)).toVector
    Detection(order.map(d.boxes), order.map(d.masks), order.map(d.confidences), d.width, d.height, keepClasses(d, order))

  def offsetDilate(mask: Array[Int], width: Int, height: Int, x: Int, y: Int, dilateErode: Int): Array[Int] =
    var a = mask.clone()
    if x != 0 || y != 0 then
      val shifted = new Array[Int](width * height)
      val dx = x
      val dy = -y // y up is positive
      val (xs, xd) = if dx >= 0 then (0, dx) else (-dx, 0)
      val (ys, yd) = if dy >= 0 then (0, dy) else (-dy, 0)
      val cw = width - math.abs(dx)
      val ch = height - math.abs(dy)
      if cw > 0 && ch > 0 then
        var row = 0
        while row < ch do
          System.arraycopy(a, (ys + row) * width + xs, shifted, (yd + row) * width + xd, cw)
          row += 1
      a = shifted

    if dilateErode != 0 then
      val kSize = math.abs(dilateErode)
      val half = kSize / 2
      val next = new Array[Int](width * height)
      val isDilate = dilateErode > 0
      var py = 0
      while py < height do
        var px = 0
        while px < width do
          var best = if isDilate then 0 else 255
          var ky = -half
          while ky < -half + kSize do
            val sy = py + ky
            if sy >= 0 && sy < height then
              var kx = -half
              while kx < -half + kSize do
                val sx = px + kx
                if sx >= 0 && sx < width then
                  val v = a(sy * width + sx)
                  if isDilate then { if v > best then best = v }
                  else { if v < best then best = v }
                kx += 1
            ky += 1
          next(py * width + px) = best
          px += 1
        py += 1
      a = next
    a

  def mergeInvert(masks: Vector[Array[Int]], mode: String): Vector[Array[Int]] =
    val m = Option(mode).getOrElse("None").trim.toLowerCase(Locale.ROOT)
    if masks.isEmpty || m == "none" then masks
    else
      val len = masks.head.length
      val merged = new Array[Int](len)
      for mask <- masks do
        var i = 0
        while i < len do
          if mask(i) > merged(i) then merged(i) = mask(i)
          i += 1
      if m.contains("invert") then
        var i = 0
        while i < len do
          merged(i) = 255 - merged(i)
          i += 1
      Vector(merged)

  def shapedMasks(d: Detection, shape: String, scale: Double): Vector[Array[Int]] =
    val isEllipse = shape.equalsIgnoreCase("ellipse")
    d.boxes.map { case (x1, y1, x2, y2) =>
      val cx = (x1 + x2) / 2.0
      val cy = (y1 + y2) / 2.0
      val hw = math.max(1.0, (x2 - x1) * scale / 2.0)
      val hh = math.max(1.0, (y2 - y1) * scale / 2.0)
      val out = new Array[Int](d.width * d.height)
      var y = math.max(0, math.floor(cy - hh).toInt)
      val maxY = math.min(d.height - 1, math.ceil(cy + hh).toInt)
      val minX = math.max(0, math.floor(cx - hw).toInt)
      val maxX = math.min(d.width - 1, math.ceil(cx + hw).toInt)
      while y <= maxY do
        var x = minX
        while x <= maxX do
          val inside =
            if isEllipse then
              val nx = (x - cx) / hw
              val ny = (y - cy) / hh
              nx * nx + ny * ny <= 1.0
            else true
          if inside then out(y * d.width + x) = 255
          x += 1
        y += 1
      out
    }

  def unitMasks(image: BufferedImage, u: AdetailerUnit): Vector[Array[Int]] =
    var d = detect(image, u.model, u.confidence)
    d = filterByRatio(d, u.maskMinRatio, u.maskMaxRatio)
    d = filterKLargest(d, u.maskKLargest)
    d = sortLeftToRight(d)
    if u.maskShape.equalsIgnoreCase("ellipse") || u.maskShape.equalsIgnoreCase("box") then
      d = d.copy(masks = shapedMasks(d, u.maskShape.toLowerCase(Locale.ROOT), u.maskScale))
    val processed = d.masks.map(m => offsetDilate(m, d.width, d.height, u.xOffset, u.yOffset, u.dilateErode))
    mergeInvert(processed, u.maskMergeInvert).filter(_.exists(_ > 0))

  /** Per-channel LAB mean/std transfer from `reference` to `image` (Reinhard style). */
  def matchColors(image: BufferedImage, reference: BufferedImage): BufferedImage =
    val w = image.getWidth
    val h = image.getHeight
    val refResized =
      if reference.getWidth == w && reference.getHeight == h then reference
      else
        val b = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = b.createGraphics()
        try
          g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
          g.drawImage(reference, 0, 0, w, h, null)
        finally g.dispose()
        b

    def rgbToLab(rgb: Int): (Double, Double, Double) =
      def pivotRgb(c: Double): Double =
        val v = c / 255.0
        if v > 0.04045 then math.pow((v + 0.055) / 1.055, 2.4) else v / 12.92
      val r = pivotRgb(((rgb >> 16) & 0xff).toDouble)
      val g = pivotRgb(((rgb >> 8) & 0xff).toDouble)
      val b = pivotRgb((rgb & 0xff).toDouble)
      val x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
      val y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
      val z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
      def f(t: Double): Double = if t > 0.008856 then math.cbrt(t) else (7.787 * t) + (16.0 / 116.0)
      val fx = f(x); val fy = f(y); val fz = f(z)
      val l = (116.0 * fy - 16.0) * 255.0 / 100.0
      val a = 500.0 * (fx - fy) + 128.0
      val bb = 200.0 * (fy - fz) + 128.0
      (l, a, bb)

    def labToRgb(l: Double, a: Double, bb: Double): Int =
      val fy = ((l * 100.0 / 255.0) + 16.0) / 116.0
      val fx = (a - 128.0) / 500.0 + fy
      val fz = fy - (bb - 128.0) / 200.0
      def invF(t: Double): Double =
        val t3 = t * t * t
        if t3 > 0.008856 then t3 else (t - 16.0 / 116.0) / 7.787
      val x = invF(fx) * 0.95047
      val y = invF(fy) * 1.00000
      val z = invF(fz) * 1.08883
      val rl = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
      val gl = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
      val bl = x * 0.0556434 - y * 0.2040259 + z * 1.0572252
      def unpivot(v: Double): Int =
        val c = if v > 0.0031308 then 1.055 * math.pow(v, 1.0 / 2.4) - 0.055 else 12.92 * v
        math.max(0, math.min(255, math.round(c * 255.0).toInt))
      (0xff << 24) | (unpivot(rl) << 16) | (unpivot(gl) << 8) | unpivot(bl)

    val n = math.max(1, w * h)
    val aL = new Array[Double](n); val aA = new Array[Double](n); val aB = new Array[Double](n)
    val rL = new Array[Double](n); val rA = new Array[Double](n); val rB = new Array[Double](n)
    var i = 0
    while i < n do
      val px = i % w; val py = i / w
      val (l1, a1, b1) = rgbToLab(image.getRGB(px, py))
      val (l2, a2, b2) = rgbToLab(refResized.getRGB(px, py))
      aL(i) = l1; aA(i) = a1; aB(i) = b1
      rL(i) = l2; rA(i) = a2; rB(i) = b2
      i += 1

    def stats(arr: Array[Double]): (Double, Double) =
      val mean = arr.sum / n
      var sq = 0.0
      var k = 0
      while k < n do
        val d = arr(k) - mean
        sq += d * d
        k += 1
      (mean, math.sqrt(sq / n))

    val (amL, asdL) = stats(aL); val (rmL, rsdL) = stats(rL)
    val (amA, asdA) = stats(aA); val (rmA, rsdA) = stats(rA)
    val (amB, asdB) = stats(aB); val (rmB, rsdB) = stats(rB)

    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    i = 0
    while i < n do
      val px = i % w; val py = i / w
      val nl = math.max(0.0, math.min(255.0, (aL(i) - amL) / (asdL + 1e-6) * rsdL + rmL))
      val na = math.max(0.0, math.min(255.0, (aA(i) - amA) / (asdA + 1e-6) * rsdA + rmA))
      val nb = math.max(0.0, math.min(255.0, (aB(i) - amB) / (asdB + 1e-6) * rsdB + rmB))
      out.setRGB(px, py, labToRgb(nl, na, nb))
      i += 1
    out

  /** `[PROMPT]` -> main prompt; `[SEP]` splits per detection (the last part repeats). */
  def unitPrompts(text: String, main: String, n: Int): Vector[String] =
    val raw = Option(text).map(_.trim).filter(_.nonEmpty).getOrElse(main)
    val parts = raw.split("\\[SEP\\]", -1).map(p => p.trim.replace("[PROMPT]", main)).toVector
    val count = math.max(n, 1)
    Vector.tabulate(count)(i => parts(math.min(i, parts.size - 1)))

final case class BatchFaceSwapConfig(
    detector: String = "face_yolov8n.pt",
    confidence: Double = 0.3,
    maxFaces: Int = 0,
    maskShape: String = "ellipse",
    maskSize: Double = 1.2,
    maskBlur: Int = 8,
    denoisingStrength: Double = 0.5,
    padding: Int = 32,
    resolution: Int = 0,
    prompt: String = "",
    negativePrompt: String = "",
    steps: Int = 0,
    cfgScale: Double = 0.0,
    sampler: String = "",
    colorCorrection: Boolean = true
):
  require(detector.trim.nonEmpty && !detector.trim.equalsIgnoreCase("none"), "Batch Face Swap: a detector is required")
  require(confidence.isFinite && confidence >= 0.0 && confidence <= 1.0, s"Batch Face Swap: confidence must be between 0 and 1 (got $confidence)")
  require(maxFaces >= 0 && maxFaces <= 64, s"Batch Face Swap: max_faces must be an integer between 0 and 64 (got $maxFaces)")
  require(BatchFaceSwap.Shapes.contains(maskShape.toLowerCase(Locale.ROOT)), s"Batch Face Swap: mask_shape must be one of ${BatchFaceSwap.Shapes.mkString(", ")}")
  require(maskSize.isFinite && maskSize >= 0.3 && maskSize <= 3.0, s"Batch Face Swap: mask_size must be between 0.3 and 3 (got $maskSize)")
  require(maskBlur >= 0 && maskBlur <= 64, s"Batch Face Swap: mask_blur must be an integer between 0 and 64 (got $maskBlur)")
  require(denoisingStrength.isFinite && denoisingStrength >= 0.0 && denoisingStrength <= 1.0, s"Batch Face Swap: denoising_strength must be between 0 and 1 (got $denoisingStrength)")
  require(padding >= 0 && padding <= 512, s"Batch Face Swap: padding must be an integer between 0 and 512 (got $padding)")
  require(resolution >= 0 && resolution <= 2048 && (resolution == 0 || resolution >= 64), "Batch Face Swap: resolution must be 0 (the generation size) or at least 64")
  require(steps >= 0 && steps <= 150, s"Batch Face Swap: steps must be an integer between 0 and 150 (got $steps)")
  require(cfgScale.isFinite && cfgScale >= 0.0 && cfgScale <= 30.0, s"Batch Face Swap: cfg_scale must be between 0 and 30 (got $cfgScale)")

object BatchFaceSwap:
  val Shapes: Set[String] = Set("ellipse", "box")

  def toAdetailerUnit(cfg: BatchFaceSwapConfig): AdetailerUnit =
    val side = (cfg.resolution / 8) * 8
    AdetailerUnit(
      model = cfg.detector.trim,
      confidence = cfg.confidence,
      maskKLargest = cfg.maxFaces,
      maskShape = cfg.maskShape.toLowerCase(Locale.ROOT),
      maskScale = cfg.maskSize,
      dilateErode = 0,
      maskBlur = cfg.maskBlur,
      denoisingStrength = cfg.denoisingStrength,
      inpaintOnlyMasked = true,
      inpaintPadding = cfg.padding,
      useInpaintWidthHeight = side > 0,
      inpaintWidth = if side > 0 then side else 512,
      inpaintHeight = if side > 0 then side else 512,
      prompt = cfg.prompt,
      negativePrompt = cfg.negativePrompt,
      useSteps = cfg.steps > 0,
      steps = if cfg.steps > 0 then cfg.steps else 28,
      useCfgScale = cfg.cfgScale > 0.0,
      cfgScale = if cfg.cfgScale > 0.0 then cfg.cfgScale else 7.0,
      useSampler = cfg.sampler.trim.nonEmpty,
      sampler = cfg.sampler.trim,
      colorCorrection = cfg.colorCorrection
    )

// =====================================================================================================================
// 2. NudeNet NSFW Censor (`nsfw_checker` hook & `image_tool: "nsfw_checker"`)
// =====================================================================================================================

object NsfwChecker:
  val Labels: Vector[String] = Vector(
    "Female_genitalia_covered", "Face_female", "Buttocks_exposed", "Female_breast_exposed",
    "Female_genitalia_exposed", "Male_breast_exposed", "Anus_exposed", "Feet_exposed", "Belly_covered",
    "Feet_covered", "Armpits_covered", "Armpits_exposed", "Face_male", "Belly_exposed", "Male_genitalia_exposed",
    "Anus_covered", "Female_breast_covered", "Buttocks_covered"
  )
  val Friendly: Vector[String] = Labels.map(_.replace("_", " "))
  val DefaultSelected: Set[String] = Set("Anus exposed", "Female breast exposed", "Female genitalia exposed", "Male genitalia exposed")
  val DefaultThresholds: Vector[Float] = Friendly.map(name => if DefaultSelected.contains(name) then 0.25f else 1.0f)
  val DefaultExpand: Vector[Float] = Vector.fill(Labels.size)(1.0f)

  val MaskShapes: Vector[String] = Vector("Ellipse", "Rectangle", "Rounded rectangle", "Entire image")
  val EntireImageAlways: String = "Entire image (always)"
  val Filters: Vector[String] = Vector("Variable blur", "Gaussian Blur", "Pixelate", "Fill color", "No censor", "Disable")
  val ToolCheckers: Vector[String] = Vector("nudenet", "nsfwjs", "safety_checker")
  val AuxiliarySuffixes: Vector[String] = Vector("-before-censor", "-censor-mask")

  final case class NsfwConfig(
      genFilterType: String = "Gaussian Blur",
      livePreviewFilterType: String = "Gaussian Blur",
      saveBeforeCensor: Boolean = true,
      saveMask: Boolean = false,
      maskShape: String = "Ellipse",
      rectangleRoundRadius: Double = 0.5,
      blurRadius: Double = 10.0,
      maskBlendRadius: Double = 0.0,
      maskBlendRadiusVariableBlur: Double = 10.0,
      blurStrengthCurve: Double = 3.0,
      pixelationFactor: Double = 5.0,
      fillColor: String = "#000000",
      nmsThreshold: Double = 0.5,
      verboseDetection: Boolean = false,
      thresholds: Vector[Float] = DefaultThresholds,
      expandHorizontal: Vector[Float] = DefaultExpand,
      expandVertical: Vector[Float] = DefaultExpand,
      nudenet: Boolean = true,
      safetyChecker: Boolean = false,
      safetyCheckerLivePreview: Boolean = false,
      nsfwjs: Boolean = false,
      nsfwjsSafetyLevel: String = "safe"
  ):
    require(Filters.contains(genFilterType), s"gen_filter_type must be one of ${Filters.mkString(", ")} (got $genFilterType)")
    require(Filters.contains(livePreviewFilterType), s"live_preview_filter_type must be one of ${Filters.mkString(", ")} (got $livePreviewFilterType)")
    require(MaskShapes.contains(maskShape), s"mask_shape must be one of ${MaskShapes.mkString(", ")} (got $maskShape)")
    require(rectangleRoundRadius.isFinite, "rectangle_round_radius must be a finite number")
    require(blurRadius.isFinite && blurRadius >= 0.0, "blur_radius must be >= 0")
    require(maskBlendRadius.isFinite && maskBlendRadius >= 0.0, "mask_blend_radius must be >= 0")
    require(maskBlendRadiusVariableBlur.isFinite && maskBlendRadiusVariableBlur >= 0.0, "mask_blend_radius_variable_blur must be >= 0")
    require(blurStrengthCurve.isFinite, "blur_strength_curve must be a finite number")
    require(pixelationFactor.isFinite && pixelationFactor >= 1e-6, "pixelation_factor must be >= 1e-6")
    require(nmsThreshold.isFinite && nmsThreshold >= 0.0 && nmsThreshold <= 1.0, "nms_threshold must be in 0..1")
    require(thresholds.size == Labels.size && thresholds.forall(_.isFinite), s"thresholds must list ${Labels.size} finite numbers")
    require(expandHorizontal.size == Labels.size && expandHorizontal.forall(v => v.isFinite && v >= 0f), s"expand_horizontal must list ${Labels.size} numbers >= 0")
    require(expandVertical.size == Labels.size && expandVertical.forall(v => v.isFinite && v >= 0f), s"expand_vertical must list ${Labels.size} numbers >= 0")
    parseColor(fillColor)

  def parseColor(text: String): Color =
    val s = Option(text).getOrElse("#000000").trim
    try
      if s.startsWith("#") && s.length == 7 then
        new Color(Integer.parseInt(s.substring(1, 3), 16), Integer.parseInt(s.substring(3, 5), 16), Integer.parseInt(s.substring(5, 7), 16))
      else if s.startsWith("#") && s.length == 4 then
        val r = Integer.parseInt(s.substring(1, 2), 16) * 17
        val g = Integer.parseInt(s.substring(2, 3), 16) * 17
        val b = Integer.parseInt(s.substring(3, 4), 16) * 17
        new Color(r, g, b)
      else Color.decode(s)
    catch
      case _: Exception => throw new IllegalArgumentException(s"fill_color must be a colour such as #000000 (got $text)")

  def scaleFactor(width: Int, height: Int): Double =
    math.sqrt((width.toDouble * width.toDouble + height.toDouble * height.toDouble) / 524288.0)

  def blurRadii(levels: Array[Int], blurRadius: Double, curve: Double): Array[Double] =
    levels.map { lv =>
      math.pow(lv.toDouble / 255.0, (12.0 - 2.0 * curve) / curve) * blurRadius
    }

  def drawMaskShape(
      width: Int,
      height: Int,
      boxes: Vector[(Double, Double, Double, Double)],
      maskShape: String,
      rectangleRoundRadius: Double
  ): BufferedImage =
    if maskShape == "Entire image" || maskShape == EntireImageAlways then
      val img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      val g = img.createGraphics()
      try
        g.setColor(Color.WHITE)
        g.fillRect(0, 0, width, height)
      finally g.dispose()
      img
    else
      val img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      val g = img.createGraphics()
      try
        g.setColor(Color.WHITE)
        for (x1, y1, x2, y2) <- boxes do
          val bw = math.max(1.0, x2 - x1)
          val bh = math.max(1.0, y2 - y1)
          maskShape match
            case "Ellipse" =>
              g.fill(new Ellipse2D.Double(x1, y1, bw, bh))
            case "Rectangle" =>
              g.fill(new Rectangle2D.Double(x1, y1, bw, bh))
            case "Rounded rectangle" =>
              val r =
                if rectangleRoundRadius > 1.0 then rectangleRoundRadius * math.sqrt((bw * bw + bh * bh) / 524288.0)
                else if rectangleRoundRadius < 0.0 then -rectangleRoundRadius
                else (math.min(bw, bh) / 2.0) * rectangleRoundRadius
              val arc = math.round(r).toDouble * 2.0
              g.fill(new RoundRectangle2D.Double(x1, y1, bw, bh, arc, arc))
            case _ =>
              throw new IllegalArgumentException(s"mask_shape must be one of ${MaskShapes.mkString(", ")} (got $maskShape)")
      finally g.dispose()
      img

  def gaussianBlurImage(img: BufferedImage, radius: Double): BufferedImage =
    if radius <= 0.1 then
      val copy = new BufferedImage(img.getWidth, img.getHeight, img.getType)
      val g = copy.createGraphics()
      try g.drawImage(img, 0, 0, null) finally g.dispose()
      return copy
    val w = img.getWidth
    val h = img.getHeight
    val rad = math.min(64, math.max(1, math.ceil(radius * 2.5).toInt))
    val sigma = math.max(0.1, radius)
    val kernel = Array.tabulate(2 * rad + 1) { i =>
      val d = (i - rad).toDouble
      math.exp(-(d * d) / (2.0 * sigma * sigma))
    }
    val kSum = kernel.sum
    var ki = 0
    while ki < kernel.length do { kernel(ki) /= kSum; ki += 1 }

    val isGray = img.getType == BufferedImage.TYPE_BYTE_GRAY
    if isGray then
      val raster = img.getRaster
      val src = new Array[Double](w * h)
      val tmp = new Array[Double](w * h)
      var y = 0
      while y < h do
        var x = 0
        while x < w do
          src(y * w + x) = raster.getSample(x, y, 0).toDouble
          x += 1
        y += 1
      y = 0
      while y < h do
        var x = 0
        while x < w do
          var acc = 0.0
          var k = -rad
          while k <= rad do
            val sx = math.max(0, math.min(w - 1, x + k))
            acc += src(y * w + sx) * kernel(k + rad)
            k += 1
          tmp(y * w + x) = acc
          x += 1
        y += 1
      val out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
      val outRaster = out.getRaster
      y = 0
      while y < h do
        var x = 0
        while x < w do
          var acc = 0.0
          var k = -rad
          while k <= rad do
            val sy = math.max(0, math.min(h - 1, y + k))
            acc += tmp(sy * w + x) * kernel(k + rad)
            k += 1
          outRaster.setSample(x, y, 0, math.max(0, math.min(255, math.round(acc).toInt)))
          x += 1
        y += 1
      out
    else
      val rArr = new Array[Double](w * h); val gArr = new Array[Double](w * h); val bArr = new Array[Double](w * h)
      val rTmp = new Array[Double](w * h); val gTmp = new Array[Double](w * h); val bTmp = new Array[Double](w * h)
      var y = 0
      while y < h do
        var x = 0
        while x < w do
          val rgb = img.getRGB(x, y)
          val idx = y * w + x
          rArr(idx) = ((rgb >> 16) & 0xff).toDouble
          gArr(idx) = ((rgb >> 8) & 0xff).toDouble
          bArr(idx) = (rgb & 0xff).toDouble
          x += 1
        y += 1
      y = 0
      while y < h do
        var x = 0
        while x < w do
          var ar = 0.0; var ag = 0.0; var ab = 0.0
          var k = -rad
          while k <= rad do
            val sx = math.max(0, math.min(w - 1, x + k))
            val wgt = kernel(k + rad)
            val sIdx = y * w + sx
            ar += rArr(sIdx) * wgt; ag += gArr(sIdx) * wgt; ab += bArr(sIdx) * wgt
            k += 1
          val dIdx = y * w + x
          rTmp(dIdx) = ar; gTmp(dIdx) = ag; bTmp(dIdx) = ab
          x += 1
        y += 1
      val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      y = 0
      while y < h do
        var x = 0
        while x < w do
          var ar = 0.0; var ag = 0.0; var ab = 0.0
          var k = -rad
          while k <= rad do
            val sy = math.max(0, math.min(h - 1, y + k))
            val wgt = kernel(k + rad)
            val sIdx = sy * w + x
            ar += rTmp(sIdx) * wgt; ag += gTmp(sIdx) * wgt; ab += bTmp(sIdx) * wgt
            k += 1
          val rr = math.max(0, math.min(255, math.round(ar).toInt))
          val gg = math.max(0, math.min(255, math.round(ag).toInt))
          val bb = math.max(0, math.min(255, math.round(ab).toInt))
          out.setRGB(x, y, (0xff << 24) | (rr << 16) | (gg << 8) | bb)
          x += 1
        y += 1
      out

  def combineWithMask(img: BufferedImage, maskGray: BufferedImage, processed: BufferedImage): BufferedImage =
    val w = img.getWidth
    val h = img.getHeight
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val mRaster = maskGray.getRaster
    var y = 0
    while y < h do
      var x = 0
      while x < w do
        val alpha = mRaster.getSample(x, y, 0).toDouble / 255.0
        val c0 = img.getRGB(x, y)
        val c1 = processed.getRGB(x, y)
        val r = math.round(((c0 >> 16) & 0xff) * (1.0 - alpha) + ((c1 >> 16) & 0xff) * alpha).toInt
        val g = math.round(((c0 >> 8) & 0xff) * (1.0 - alpha) + ((c1 >> 8) & 0xff) * alpha).toInt
        val b = math.round((c0 & 0xff) * (1.0 - alpha) + (c1 & 0xff) * alpha).toInt
        out.setRGB(x, y, (0xff << 24) | (math.max(0, math.min(255, r)) << 16) | (math.max(0, math.min(255, g)) << 8) | math.max(0, math.min(255, b)))
        x += 1
      y += 1
    out

  def applyFilter(
      img: BufferedImage,
      maskGray: BufferedImage,
      filterType: String,
      blurRadius: Double = 10.0,
      blurStrengthCurve: Double = 3.0,
      pixelationFactor: Double = 5.0,
      color: String = "#000000"
  ): BufferedImage =
    require(Filters.contains(filterType), s"filter_type must be one of ${Filters.mkString(", ")} (got $filterType)")
    filterType match
      case "Variable blur" =>
        if blurStrengthCurve <= 0.0 then img
        else if blurStrengthCurve > 6.0 then gaussianBlurImage(img, blurRadius)
        else
          val w = img.getWidth
          val h = img.getHeight
          val mRaster = maskGray.getRaster
          val levelsSet = mutable.TreeSet[Int]()
          var y = 0
          while y < h do
            var x = 0
            while x < w do
              levelsSet += mRaster.getSample(x, y, 0)
              x += 1
            y += 1
          val levels = levelsSet.toArray
          val radii = blurRadii(levels, blurRadius, blurStrengthCurve)
          val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
          for (lv, r) <- levels.zip(radii) do
            val layer = gaussianBlurImage(img, r)
            var py = 0
            while py < h do
              var px = 0
              while px < w do
                if mRaster.getSample(px, py, 0) == lv then
                  out.setRGB(px, py, layer.getRGB(px, py))
                px += 1
              py += 1
          out
      case "Gaussian Blur" =>
        combineWithMask(img, maskGray, gaussianBlurImage(img, blurRadius))
      case "Pixelate" =>
        val w = img.getWidth
        val h = img.getHeight
        val sw = math.max(1, math.round(w.toDouble / (pixelationFactor * pixelationFactor)).toInt)
        val sh = math.max(1, math.round(h.toDouble / (pixelationFactor * pixelationFactor)).toInt)
        val small = new BufferedImage(sw, sh, BufferedImage.TYPE_INT_RGB)
        val g1 = small.createGraphics()
        try
          g1.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
          g1.drawImage(img, 0, 0, sw, sh, null)
        finally g1.dispose()
        val big = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g2 = big.createGraphics()
        try
          g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR)
          g2.drawImage(small, 0, 0, w, h, null)
        finally g2.dispose()
        combineWithMask(img, maskGray, big)
      case "Fill color" =>
        val c = parseColor(color)
        val filled = new BufferedImage(img.getWidth, img.getHeight, BufferedImage.TYPE_INT_RGB)
        val g = filled.createGraphics()
        try
          g.setColor(c)
          g.fillRect(0, 0, img.getWidth, img.getHeight)
        finally g.dispose()
        combineWithMask(img, maskGray, filled)
      case _ => // "No censor" or "Disable"
        img

  /** Detect candidate NSFW regions according to active category thresholds and expansion factors. */
  def detectRegions(img: BufferedImage, cfg: NsfwConfig): Option[BufferedImage] =
    val minActiveThreshold = cfg.thresholds.min
    if minActiveThreshold >= 1.0f then return None
    val det = Adetailer.detect(img, "nudenet", minActiveThreshold.toDouble)
    // a detection counts only when its own class threshold is met (thresholds of 1.0 switch a category off)
    val active = det.boxes.indices.filter(i => det.confidences(i) >= cfg.thresholds(det.classes(i)).toDouble)
    if active.isEmpty then None
    else
      val expanded = active.map { i =>
        val (x1, y1, x2, y2) = det.boxes(i)
        val cIdx = det.classes(i)
        val eh = cfg.expandHorizontal(cIdx).toDouble
        val ev = cfg.expandVertical(cIdx).toDouble
        val w = (x2 - x1) * eh
        val h = (y2 - y1) * ev
        val cx = (x1 + x2) / 2.0
        val cy = (y1 + y2) / 2.0
        (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
      }.toVector
      Some(drawMaskShape(img.getWidth, img.getHeight, expanded, cfg.maskShape, cfg.rectangleRoundRadius))

  /** Run final generation censoring; returns `(censoredImage, Vector[(suffix, auxiliaryImage)])`. */
  def censorFinal(img: BufferedImage, cfg: NsfwConfig): (BufferedImage, Vector[(String, BufferedImage)]) =
    if cfg.genFilterType == "Disable" || !cfg.nudenet then return (img, Vector.empty)
    detectRegions(img, cfg) match
      case None => (img, Vector.empty)
      case Some(rawMask) =>
        val before = if cfg.saveBeforeCensor then Some(img) else None
        val factor = scaleFactor(img.getWidth, img.getHeight)
        val blend = if cfg.genFilterType == "Variable blur" then cfg.maskBlendRadiusVariableBlur else cfg.maskBlendRadius
        val blendedMask = gaussianBlurImage(rawMask, blend * factor)
        val censored = applyFilter(
          img,
          blendedMask,
          cfg.genFilterType,
          blurRadius = cfg.blurRadius * factor,
          blurStrengthCurve = cfg.blurStrengthCurve,
          pixelationFactor = cfg.pixelationFactor,
          color = cfg.fillColor
        )
        val aux = new ArrayBuffer[(String, BufferedImage)]()
        before.foreach(b => aux += (("-before-censor", b)))
        if cfg.saveMask then aux += (("-censor-mask", blendedMask))
        (censored, aux.toVector)

// =====================================================================================================================
// 3. Aesthetic Score (`aesthetic_score` hook & `image_tool: "aesthetic_score"`)
// =====================================================================================================================

final case class AestheticScoreConfig(
    improved: Boolean = true,
    simulacra: String = "off",
    cafeAesthetic: Boolean = false,
    cafeStyle: Boolean = false,
    cafeWaifu: Boolean = false,
    pngChunks: Boolean = true,
    generationFields: Boolean = false,
    hash: Boolean = false,
    forceCpu: Boolean = false
):
  val normalizedSimulacra: String =
    val s = Option(simulacra).getOrElse("off").trim.toLowerCase(Locale.ROOT)
    if s.isEmpty || s == "none" || s == "false" then "off" else s
  require(AestheticScore.SimulacraChoices.contains(normalizedSimulacra),
    s"simulacra must be one of ${AestheticScore.SimulacraChoices.keys.mkString(", ")} (got $simulacra)")
  require(scorers.nonEmpty, "Aesthetic score: enable at least one scorer")

  def scorers: Vector[String] =
    val out = new ArrayBuffer[String]()
    if improved then out += "improved"
    AestheticScore.SimulacraChoices(normalizedSimulacra).foreach(s => out += s"simulacra_$s")
    if cafeAesthetic then out += "cafe_aesthetic"
    if cafeStyle then out += "cafe_style"
    if cafeWaifu then out += "cafe_waifu"
    out.toVector

object AestheticScore:
  val SimulacraChoices: Map[String, Option[String]] = Map(
    "off" -> None,
    "vit-l-14" -> Some("l14"),
    "vit-b-16" -> Some("b16")
  )
  val CafeLabels: Map[String, Vector[String]] = Map(
    "cafe_aesthetic" -> Vector("not_aesthetic", "aesthetic"),
    "cafe_style" -> Vector("anime", "real_life", "3d", "manga_like", "other"),
    "cafe_waifu" -> Vector("not_waifu", "waifu")
  )

  sealed trait ScoreValue
  final case class NumericScore(value: Double) extends ScoreValue
  final case class DistributionScore(probs: Vector[(String, Double)]) extends ScoreValue

  /** Runs the real scorers (see [[AestheticModels]]): `improved` = LAION improved aesthetic predictor (CLIP ViT-L/14 + MLP, rounded to 0.1),
    * `simulacra_l14` / `simulacra_b16` = Simulacra aesthetic models (rounded to 0.01), `cafe_*` = cafeai BEiT classifiers (probabilities rounded
    * to 4 places). Missing weights throw IllegalStateException ("weights not found"); nothing is invented. */
  def scoreImage(image: BufferedImage, cfg: AestheticScoreConfig): Vector[(String, ScoreValue)] =
    cfg.scorers.map {
      case "improved" =>
        "improved" -> NumericScore(math.rint(AestheticModels.improved(image) * 10.0) / 10.0)
      case name if name.startsWith("simulacra_") =>
        name -> NumericScore(math.rint(AestheticModels.simulacra(image, name.stripPrefix("simulacra_")) * 100.0) / 100.0)
      case name =>
        val probs = AestheticModels.cafe(image, name, CafeLabels(name))
        name -> DistributionScore(probs.map((l, p) => l -> math.rint(p * 10000.0) / 10000.0))
    }

  def infotextFields(scores: Vector[(String, ScoreValue)]): Vector[(String, String)] =
    scores.map {
      case ("improved", NumericScore(v)) =>
        "aesthetic_score" -> String.format(Locale.ROOT, "%.1f", Double.box(v))
      case (name, NumericScore(v)) if name.startsWith("simulacra_") =>
        val modelLabel = if name.endsWith("l14") then "ViT-L/14" else "ViT-B/16"
        s"Aesthetic Scorer $modelLabel" -> String.format(Locale.ROOT, "%.2f", Double.box(v))
      case (name, DistributionScore(probs)) =>
        val (topLabel, topP) = probs.maxBy(_._2)
        val cap = name.replace("_", " ").capitalize
        cap -> String.format(Locale.ROOT, "%s %.4f", topLabel, Double.box(topP))
      case (name, _) => name -> ""
    }

  def pngChunks(scores: Vector[(String, ScoreValue)]): Vector[(String, String)] =
    scores.map {
      case ("improved", NumericScore(v)) =>
        "aesthetic_score" -> String.format(Locale.ROOT, "%.1f", Double.box(v))
      case (name, NumericScore(v)) if name.startsWith("simulacra_") =>
        "score" -> String.format(Locale.ROOT, "%.2f", Double.box(v))
      case (name, DistributionScore(probs)) =>
        val json = probs.map { case (k, p) => s""""$k":${BigDecimal.valueOf(p).bigDecimal.stripTrailingZeros.toPlainString}""" }.mkString("{", ",", "}")
        name -> json
      case (name, _) => name -> ""
    }

  def paramsHash(
      steps: Int,
      sampler: String,
      cfgScale: Double,
      seed: Long,
      width: Int,
      height: Int,
      modelHash: String,
      prompt: String,
      negativePrompt: String
  ): String =
    def esc(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"")
    val cfgStr = if cfgScale == cfgScale.toLong.toDouble then s"${cfgScale.toLong}.0" else cfgScale.toString
    val json = s"""{"steps": $steps, "sampler": "${esc(sampler)}", "cfg_scale": $cfgStr, "seed": $seed, "width": $width, "height": $height, "model": "${esc(modelHash)}", "prompt": "${esc(prompt)}", "negative_prompt": "${esc(negativePrompt)}"}"""
    val md5 = MessageDigest.getInstance("MD5").digest(json.getBytes(StandardCharsets.UTF_8))
    md5.map(b => f"${b & 0xff}%02x").mkString

// =====================================================================================================================
// 4. AnimateDiff (`animatediff` hook & `animatediff_encode` request)
// =====================================================================================================================

final case class AnimateDiffConfig(
    model: String = "",
    videoLength: Int = 16,
    fps: Int = 8,
    loopNumber: Int = 0,
    closedLoop: String = "R+P",
    batchSize: Int = 16,
    stride: Int = 1,
    overlap: Int = -1,
    interp: String = "Off",
    interpX: Int = 10,
    latentPower: Double = 1.0,
    latentScale: Double = 32.0,
    lastFrame: String = "",
    latentPowerLast: Double = 1.0,
    latentScaleLast: Double = 32.0,
    freeInit: Boolean = false,
    freeInitFilter: String = "butterworth",
    freeInitDS: Double = 0.25,
    freeInitDT: Double = 0.25,
    freeInitN: Int = 4,
    freeInitIterations: Int = 3,
    linearBetas: Boolean = true
):
  val effectiveVideoLength: Int = if videoLength == 0 then batchSize else videoLength
  require(videoLength >= 0 && videoLength <= AnimateDiff.MaxFrames, s"AnimateDiff: video_length must be in 0..${AnimateDiff.MaxFrames}")
  require(effectiveVideoLength >= 1, "AnimateDiff: video_length must be at least 1")
  require(fps >= 1 && fps <= 120, s"AnimateDiff: fps must be in 1..120 (got $fps)")
  require(loopNumber >= 0 && loopNumber <= 1000, s"AnimateDiff: loop_number must be in 0..1000 (got $loopNumber)")
  require(batchSize >= 1 && batchSize <= AnimateDiff.MaxContext, s"AnimateDiff: batch_size must be in 1..${AnimateDiff.MaxContext} (got $batchSize)")
  require(stride >= 1 && stride <= 16, s"AnimateDiff: stride must be in 1..16 (got $stride)")
  require(overlap >= -1 && overlap <= AnimateDiff.MaxContext - 1, s"AnimateDiff: overlap must be in -1..${AnimateDiff.MaxContext - 1}")
  require(AnimateDiff.ClosedLoops.contains(closedLoop), s"AnimateDiff: closed_loop must be one of ${AnimateDiff.ClosedLoops.mkString(", ")} (got $closedLoop)")
  require(interp.isEmpty || interp.equalsIgnoreCase("off"), "AnimateDiff: frame interpolation (FILM) needs the Deforum extension and is not provided")
  require(AnimateDiff.FreeInitFilters.contains(freeInitFilter.toLowerCase(Locale.ROOT)),
    s"AnimateDiff: free_init_filter must be one of ${AnimateDiff.FreeInitFilters.mkString(", ")} (got $freeInitFilter)")
  require(freeInitDS.isFinite && freeInitDS >= 0.01 && freeInitDS <= 1.0, "AnimateDiff: free_init_d_s must be in 0.01..1")
  require(freeInitDT.isFinite && freeInitDT >= 0.01 && freeInitDT <= 1.0, "AnimateDiff: free_init_d_t must be in 0.01..1")
  require(freeInitN >= 1 && freeInitN <= 16, "AnimateDiff: free_init_n must be in 1..16")
  require(freeInitIterations >= 1 && freeInitIterations <= 10, "AnimateDiff: free_init_iterations must be in 1..10")

object AnimateDiff:
  val ClosedLoops: Vector[String] = Vector("N", "R-P", "R+P", "A")
  val VideoFormats: Vector[String] = Vector("GIF", "MP4", "WEBP", "WEBM")
  val FreeInitFilters: Vector[String] = Vector("butterworth", "gaussian", "box", "ideal")
  val MaxFrames: Int = 512
  val MaxContext: Int = 32

  private val KeyLineRegex = """^\s*(\d+)\s*:\s*(.*\S)?\s*$""".r

  private def cleanPiece(piece: String): String =
    piece.trim.stripSuffix(",").trim

  /** Prompt travel: head lines, `frame: prompt` lines in ascending order, tail lines. */
  def parseTravel(text: String, frames: Int): Option[(Vector[Int], Vector[String])] =
    val lines = Option(text).getOrElse("").linesIterator.map(_.trim).filter(_.nonEmpty).toVector
    val head = new ArrayBuffer[String]()
    val keys = new ArrayBuffer[(Int, String)]()
    val tail = new ArrayBuffer[String]()
    for line <- lines do
      line match
        case KeyLineRegex(fStr, bodyOrNull) if tail.isEmpty =>
          keys += ((fStr.toInt, Option(bodyOrNull).getOrElse("")))
        case KeyLineRegex(_, _) =>
          throw new IllegalArgumentException("AnimateDiff: prompt travel keyframe lines must follow each other; text between them belongs to the tail prompt")
        case _ if keys.nonEmpty =>
          tail += line
        case _ =>
          head += line
    if keys.isEmpty then None
    else
      val framesList = keys.map(_._1).toVector
      if framesList != framesList.distinct.sorted then
        throw new IllegalArgumentException("AnimateDiff: prompt travel frame numbers must be strictly ascending")
      if framesList.last >= frames then
        throw new IllegalArgumentException(s"AnimateDiff: prompt travel frame ${framesList.last} must be smaller than the number of frames ($frames)")
      val headText = head.map(cleanPiece).filter(_.nonEmpty).toVector
      val tailText = tail.map(cleanPiece).filter(_.nonEmpty).toVector
      val full = keys.map { case (_, p) =>
        val mid = cleanPiece(p)
        (headText ++ (if mid.nonEmpty then Vector(mid) else Vector.empty) ++ tailText).mkString(", ")
      }.toVector
      Some((framesList, full))

  /** For each frame `(keyIdxA, keyIdxB, t)` where conditioning is `lerp(a, b, t)`. */
  def travelWeights(keyFrames: Vector[Int], frames: Int, loop: Boolean): Vector[(Int, Int, Double)] =
    val last = keyFrames.size - 1
    Vector.tabulate(frames) { f =>
      if loop then
        val pos = if f >= keyFrames.head then f else f + frames
        val ends = keyFrames :+ (keyFrames.head + frames)
        val b = (1 until ends.size).find(i => ends(i) > pos).getOrElse(ends.size - 1)
        val a = b - 1
        (a, b % keyFrames.size, (pos - ends(a)).toDouble / (ends(b) - ends(a)).toDouble)
      else if f <= keyFrames.head then (0, 0, 0.0)
      else if f >= keyFrames.last then (last, last, 0.0)
      else
        val b = keyFrames.indexWhere(_ > f)
        val a = b - 1
        (a, b, (f - keyFrames(a)).toDouble / (keyFrames(b) - keyFrames(a)).toDouble)
    }

  def stridesFor(stride: Int): Vector[Int] =
    val out = new ArrayBuffer[Int]()
    out += 1
    var s = 2
    while s <= stride do
      out += s
      s *= 2
    out.toVector

  private def slide(seq: Vector[Int], length: Int, overlap: Int): Vector[Vector[Int]] =
    val n = seq.size
    if n <= length then Vector(seq)
    else
      val step = math.max(1, length - overlap)
      val starts = (0 to (n - length) by step).toVector
      val allStarts = if starts.last != n - length then starts :+ (n - length) else starts
      allStarts.map(s => seq.slice(s, s + length))

  def contextWindows(frames: Int, length: Int, overlap: Int, stride: Int, closedLoop: String): Vector[Vector[Int]] =
    if frames <= length then Vector((0 until frames).toVector)
    else
      val effOverlap = if overlap < 0 then length / 4 else math.min(overlap, length - 1)
      val out = new ArrayBuffer[Vector[Int]]()
      for st <- stridesFor(stride); offset <- 0 until st do
        val seq = (offset until frames by st).toVector
        if seq.size >= 2 then
          out ++= slide(seq, length, effOverlap)
      if closedLoop == "R-P" || closedLoop == "R+P" then
        out += Vector.tabulate(length)(k => (frames - length / 2 + k) % frames)
      else if closedLoop == "A" then
        val step = math.max(1, length - effOverlap)
        for start <- (frames - length / 2) until (frames - length / 2 + frames) by step do
          out += Vector.tabulate(length)(k => (start + k) % frames)
      out.distinct.toVector

  def isLoopPrompt(closedLoop: String): Boolean =
    closedLoop == "R+P" || closedLoop == "A"

  /** Low-pass mask over centered `(frames, height, width)` spectrum. */
  def freqFilter(frames: Int, height: Int, width: Int, kind: String, dS: Double, dT: Double, n: Int): Array[Float] =
    val out = new Array[Float](frames * height * width)
    val k = kind.toLowerCase(Locale.ROOT)
    var f = 0
    while f < frames do
      val ut = (f.toDouble * 2.0 / frames) - 1.0
      var y = 0
      while y < height do
        val uh = (y.toDouble * 2.0 / height) - 1.0
        var x = 0
        while x < width do
          val uw = (x.toDouble * 2.0 / width) - 1.0
          val d2 = math.pow(ut * (dS / dT), 2.0) + uh * uh + uw * uw
          val v = k match
            case "butterworth" => 1.0 / (1.0 + math.pow(d2 / (dS * dS), n.toDouble))
            case "gaussian" => math.exp(-d2 / (2.0 * dS * dS))
            case "ideal" => if d2 <= dS * dS then 1.0 else 0.0
            case _ => if math.abs(uh) <= dS && math.abs(uw) <= dS && math.abs(ut) <= dT then 1.0 else 0.0
          out((f * height + y) * width + x) = v.toFloat
          x += 1
        y += 1
      f += 1
    out

  def pingpong[T](frames: Vector[T]): Vector[T] =
    if frames.size > 2 then frames ++ frames.slice(1, frames.size - 1).reverse else frames

  /** Write animated GIF from a sequence of BufferedImages using javax.imageio GIF writer. */
  def writeAnimatedGif(frames: Vector[BufferedImage], outputPath: Path, fps: Int, loopNumber: Int): Unit =
    require(frames.nonEmpty, "AnimateDiff: no frames to encode")
    val delayCs = math.max(1, math.round(100.0 / math.max(1, fps).toDouble).toInt)
    val writers = ImageIO.getImageWritersBySuffix("gif")
    require(writers.hasNext, "No GIF ImageWriter available in JVM")
    val writer = writers.next()
    val ios = ImageIO.createImageOutputStream(outputPath.toFile)
    try
      writer.setOutput(ios)
      writer.prepareWriteSequence(null)
      for (frame, idx) <- frames.zipWithIndex do
        val rgbFrame =
          if frame.getType == BufferedImage.TYPE_INT_RGB then frame
          else
            val b = new BufferedImage(frame.getWidth, frame.getHeight, BufferedImage.TYPE_INT_RGB)
            val g = b.createGraphics()
            try g.drawImage(frame, 0, 0, null) finally g.dispose()
            b
        val writeParam = writer.getDefaultWriteParam
        val specifier = ImageTypeSpecifier.createFromBufferedImageType(rgbFrame.getType)
        val metadata = writer.getDefaultImageMetadata(specifier, writeParam)
        val metaFormatName = metadata.getNativeMetadataFormatName
        val root = metadata.getAsTree(metaFormatName).asInstanceOf[IIOMetadataNode]

        def getNode(parent: IIOMetadataNode, name: String): IIOMetadataNode =
          var i = 0
          while i < parent.getLength do
            if parent.item(i).getNodeName.equalsIgnoreCase(name) then
              return parent.item(i).asInstanceOf[IIOMetadataNode]
            i += 1
          val node = new IIOMetadataNode(name)
          parent.appendChild(node)
          node

        val gce = getNode(root, "GraphicControlExtension")
        gce.setAttribute("disposalMethod", "none")
        gce.setAttribute("userInputFlag", "FALSE")
        gce.setAttribute("transparentColorFlag", "FALSE")
        gce.setAttribute("delayTime", delayCs.toString)
        gce.setAttribute("transparentColorIndex", "0")

        if idx == 0 then
          val appExts = getNode(root, "ApplicationExtensions")
          val appExt = new IIOMetadataNode("ApplicationExtension")
          appExt.setAttribute("applicationID", "NETSCAPE")
          appExt.setAttribute("authenticationCode", "2.0")
          val loopBytes = Array[Byte](1, (loopNumber & 0xff).toByte, ((loopNumber >> 8) & 0xff).toByte)
          appExt.setUserObject(loopBytes)
          appExts.appendChild(appExt)

        metadata.setFromTree(metaFormatName, root)
        writer.writeToSequence(new IIOImage(rgbFrame, null, metadata), writeParam)
      writer.endWriteSequence()
    finally
      try writer.dispose() finally ios.close()
