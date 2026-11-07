package scalatorch.models.worker

import scalatorch.models.DiffusionImageInput
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.{Random, UUID}
import javax.imageio.ImageIO
import NativeJson.*

/** Pixel art image tool (`pixel_art`): cell-average downscale, optional Bayer ordered dither,
  * k-means++ palette quantization, optional Canny dark outlines, and nearest-neighbor upscale.
  */
private[worker] object NativePixelArtImageTool:
  private val Bayer4: Array[Array[Float]] =
    val m4 = Array(
      Array(0, 8, 2, 10),
      Array(12, 4, 14, 6),
      Array(3, 11, 1, 9),
      Array(15, 7, 13, 5)
    )
    m4.map(_.map(v => (v.toFloat + 0.5f) / 16.0f - 0.5f))

  def execute(p: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    require(
      p.keySet.subsetOf(Set("tool", "image_path", "output_dir", "pixel_size", "colors", "dither", "outline", "upscale", "seed")),
      "unsupported pixel_art parameters"
    )
    def string(key: String, default: String): String = p.get(key) match
      case None | Some(Null) => default
      case Some(Str(value))  => value
      case _                 => throw IllegalArgumentException(s"$key must be a string")
    def intParam(key: String, default: Int): Int = p.get(key) match
      case None | Some(Null)                    => default
      case Some(Number(v)) if v.isValidInt      => v.toInt
      case Some(Number(v)) if v.toDouble.isFinite => v.toInt
      case _                                    => throw IllegalArgumentException(s"$key must be an integer")
    def longParam(key: String, default: Long): Long = p.get(key) match
      case None | Some(Null)                    => default
      case Some(Number(v)) if v.isValidLong     => v.toLong
      case Some(Number(v)) if v.toDouble.isFinite => v.toLong
      case _                                    => throw IllegalArgumentException(s"$key must be an integer")
    def doubleParam(key: String, default: Double): Double = p.get(key) match
      case None | Some(Null)                      => default
      case Some(Number(v)) if v.toDouble.isFinite => v.toDouble
      case _                                      => throw IllegalArgumentException(s"$key must be finite and numeric")
    def boolParam(key: String, default: Boolean): Boolean = p.get(key) match
      case None | Some(Null) => default
      case Some(Bool(v))     => v
      case _                 => throw IllegalArgumentException(s"$key must be a boolean")

    val sourceName = string("image_path", "")
    require(sourceName.nonEmpty, "image_path is required")
    val image = DiffusionImageInput.readRgb(Path.of(sourceName), compositeAlpha = false)
    val output = Path.of(string("output_dir", ".")).toAbsolutePath.normalize()
    Files.createDirectories(output)

    val w = image.getWidth
    val h = image.getHeight
    val ps = math.max(1, intParam("pixel_size", 8))
    val colors = intParam("colors", 16)
    val dither = doubleParam("dither", 0.0)
    val outline = boolParam("outline", false)
    val upscale = boolParam("upscale", true)
    val seed = longParam("seed", 0L)

    val sw = math.max(1, w / ps)
    val sh = math.max(1, h / ps)
    val nPixels = sw * sh
    val small = new Array[Float](nPixels * 3)

    // 1. Area-average downscale (cv2.INTER_AREA equivalent)
    for sy <- 0 until sh do
      context.checkCancelled()
      val y0 = sy.toDouble * h / sh
      val y1 = (sy + 1).toDouble * h / sh
      val iy0 = math.floor(y0).toInt
      val iy1 = math.min(h, math.ceil(y1).toInt)
      for sx <- 0 until sw do
        val x0 = sx.toDouble * w / sw
        val x1 = (sx + 1).toDouble * w / sw
        val ix0 = math.floor(x0).toInt
        val ix1 = math.min(w, math.ceil(x1).toInt)
        var rSum = 0.0
        var gSum = 0.0
        var bSum = 0.0
        var wSum = 0.0
        var py = iy0
        while py < iy1 do
          val wy = math.min(py + 1.0, y1) - math.max(py.toDouble, y0)
          if wy > 0.0 then
            var px = ix0
            while px < ix1 do
              val wx = math.min(px + 1.0, x1) - math.max(px.toDouble, x0)
              if wx > 0.0 then
                val weight = wx * wy
                val rgb = image.getRGB(px, py)
                rSum += ((rgb >>> 16) & 255) * weight
                gSum += ((rgb >>> 8) & 255) * weight
                bSum += (rgb & 255) * weight
                wSum += weight
              px += 1
          py += 1
        val idx = (sy * sw + sx) * 3
        val inv = if wSum > 0.0 then (1.0 / wSum).toFloat else 0.0f
        small(idx) = (rSum * inv).toFloat
        small(idx + 1) = (gSum * inv).toFloat
        small(idx + 2) = (bSum * inv).toFloat

    // 2. Optional 4x4 Bayer ordered dither
    if dither > 0.0 then
      val scale = (255.0 / math.max(colors, 2) * dither * 2.0).toFloat
      for sy <- 0 until sh do
        val bRow = Bayer4(sy & 3)
        for sx <- 0 until sw do
          val delta = bRow(sx & 3) * scale
          val idx = (sy * sw + sx) * 3
          small(idx) += delta
          small(idx + 1) += delta
          small(idx + 2) += delta

    // 3. K-means++ palette quantization (k in [2, min(colors, 256, nPixels)])
    val k = math.min(math.max(2, math.min(colors, 256)), nPixels)
    val q = kmeansQuantize(small, nPixels, k, seed, context)

    // 4. Optional Canny edge dark outline
    if outline then
      val edges = cannyEdges(q, sw, sh, 40.0f, 120.0f)
      var i = 0
      while i < nPixels do
        if edges(i) then
          val idx = i * 3
          q(idx) *= 0.35f
          q(idx + 1) *= 0.35f
          q(idx + 2) *= 0.35f
        i += 1

    // 5. Render quantized small image and optionally nearest-neighbor upscale to (w, h)
    val outW = if upscale then w else sw
    val outH = if upscale then h else sh
    val result = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until outH do
      context.checkCancelled()
      val sy = if upscale then math.min(sh - 1, (y.toLong * sh / outH).toInt) else y
      for x <- 0 until outW do
        val sx = if upscale then math.min(sw - 1, (x.toLong * sw / outW).toInt) else x
        val idx = (sy * sw + sx) * 3
        val r = math.max(0, math.min(255, q(idx).toInt))
        val g = math.max(0, math.min(255, q(idx + 1).toInt))
        val b = math.max(0, math.min(255, q(idx + 2).toInt))
        result.setRGB(x, y, (r << 16) | (g << 8) | b)

    val temp = Files.createTempFile(output, ".native-pixel-art-", ".png")
    try
      require(ImageIO.write(result, "png", temp.toFile), "PNG writer is unavailable")
      context.checkCancelled()
      val destination = output.resolve(s"native-pixel-art-${UUID.randomUUID()}.png")
      context.beginPublication()
      Files.move(temp, destination)
      Map(
        "path" -> Str(destination.toString),
        "outputs" -> Obj(Map.empty),
        "info" -> Obj(Map("palette" -> Number(k)))
      )
    finally Files.deleteIfExists(temp)

  private def kmeansQuantize(
      data: Array[Float],
      n: Int,
      k: Int,
      seed: Long,
      context: NativeWorkContext
  ): Array[Float] =
    val rng = new Random(seed)
    var bestCompactness = Double.PositiveInfinity
    var bestCenters = new Array[Float](k * 3)
    var bestLabels = new Array[Int](n)

    val distSq = new Array[Float](n)
    for _ <- 0 until 3 do
      context.checkCancelled()
      val centers = new Array[Float](k * 3)
      val labels = new Array[Int](n)
      // k-means++ initialization
      val first = rng.nextInt(n)
      centers(0) = data(first * 3)
      centers(1) = data(first * 3 + 1)
      centers(2) = data(first * 3 + 2)
      var i = 0
      while i < n do
        val dr = data(i * 3) - centers(0)
        val dg = data(i * 3 + 1) - centers(1)
        val db = data(i * 3 + 2) - centers(2)
        distSq(i) = dr * dr + dg * dg + db * db
        i += 1
      var c = 1
      while c < k do
        var sum = 0.0
        i = 0
        while i < n do
          sum += distSq(i)
          i += 1
        val chosen =
          if sum <= 0.0 then rng.nextInt(n)
          else
            val target = rng.nextDouble() * sum
            var acc = 0.0
            var idx = 0
            while idx < n - 1 && acc + distSq(idx) < target do
              acc += distSq(idx)
              idx += 1
            idx
        centers(c * 3) = data(chosen * 3)
        centers(c * 3 + 1) = data(chosen * 3 + 1)
        centers(c * 3 + 2) = data(chosen * 3 + 2)
        i = 0
        while i < n do
          val dr = data(i * 3) - centers(c * 3)
          val dg = data(i * 3 + 1) - centers(c * 3 + 1)
          val db = data(i * 3 + 2) - centers(c * 3 + 2)
          val d = dr * dr + dg * dg + db * db
          if d < distSq(i) then distSq(i) = d
          i += 1
        c += 1

      // Lloyd's iterations (max 20, eps 0.5)
      val sums = new Array[Double](k * 3)
      val counts = new Array[Int](k)
      var iter = 0
      var converged = false
      while iter < 20 && !converged do
        java.util.Arrays.fill(sums, 0.0)
        java.util.Arrays.fill(counts, 0)
        i = 0
        while i < n do
          val r = data(i * 3)
          val g = data(i * 3 + 1)
          val b = data(i * 3 + 2)
          var bestC = 0
          var bestD = Float.PositiveInfinity
          var ci = 0
          while ci < k do
            val dr = r - centers(ci * 3)
            val dg = g - centers(ci * 3 + 1)
            val db = b - centers(ci * 3 + 2)
            val d = dr * dr + dg * dg + db * db
            if d < bestD then
              bestD = d
              bestC = ci
            ci += 1
          labels(i) = bestC
          sums(bestC * 3) += r
          sums(bestC * 3 + 1) += g
          sums(bestC * 3 + 2) += b
          counts(bestC) += 1
          i += 1
        var maxShiftSq = 0.0f
        var ci = 0
        while ci < k do
          if counts(ci) > 0 then
            val nr = (sums(ci * 3) / counts(ci)).toFloat
            val ng = (sums(ci * 3 + 1) / counts(ci)).toFloat
            val nb = (sums(ci * 3 + 2) / counts(ci)).toFloat
            val sr = nr - centers(ci * 3)
            val sg = ng - centers(ci * 3 + 1)
            val sb = nb - centers(ci * 3 + 2)
            val shiftSq = sr * sr + sg * sg + sb * sb
            if shiftSq > maxShiftSq then maxShiftSq = shiftSq
            centers(ci * 3) = nr
            centers(ci * 3 + 1) = ng
            centers(ci * 3 + 2) = nb
          ci += 1
        if maxShiftSq <= 0.25f then converged = true
        iter += 1

      var compactness = 0.0
      i = 0
      while i < n do
        val ci = labels(i)
        val dr = data(i * 3) - centers(ci * 3)
        val dg = data(i * 3 + 1) - centers(ci * 3 + 1)
        val db = data(i * 3 + 2) - centers(ci * 3 + 2)
        compactness += dr * dr + dg * dg + db * db
        i += 1
      if compactness < bestCompactness then
        bestCompactness = compactness
        bestCenters = centers
        bestLabels = labels

    val out = new Array[Float](n * 3)
    var i = 0
    while i < n do
      val ci = bestLabels(i) * 3
      out(i * 3) = bestCenters(ci)
      out(i * 3 + 1) = bestCenters(ci + 1)
      out(i * 3 + 2) = bestCenters(ci + 2)
      i += 1
    out

  private def cannyEdges(q: Array[Float], w: Int, h: Int, lowThresh: Float, highThresh: Float): Array[Boolean] =
    val gray = new Array[Float](w * h)
    var i = 0
    while i < w * h do
      val r = math.max(0.0f, math.min(255.0f, q(i * 3)))
      val g = math.max(0.0f, math.min(255.0f, q(i * 3 + 1)))
      val b = math.max(0.0f, math.min(255.0f, q(i * 3 + 2)))
      gray(i) = 0.299f * r + 0.587f * g + 0.114f * b
      i += 1
    val mag = new Array[Float](w * h)
    val dir = new Array[Int](w * h)
    def at(x: Int, y: Int): Float =
      val cx = math.max(0, math.min(w - 1, x))
      val cy = math.max(0, math.min(h - 1, y))
      gray(cy * w + cx)
    for y <- 0 until h; x <- 0 until w do
      val gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0f * at(x - 1, y) + 2.0f * at(x + 1, y) - at(x - 1, y + 1) + at(x + 1, y + 1)
      val gy = -at(x - 1, y - 1) - 2.0f * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2.0f * at(x, y + 1) + at(x + 1, y + 1)
      val idx = y * w + x
      mag(idx) = math.hypot(gx, gy).toFloat
      val angle = ((math.atan2(gy, gx) * 180.0 / math.Pi) + 180.0) % 180.0
      dir(idx) =
        if angle < 22.5 || angle >= 157.5 then 0
        else if angle < 67.5 then 1
        else if angle < 112.5 then 2
        else 3
    val state = new Array[Int](w * h) // 0=none, 1=weak, 2=strong
    val stack = new Array[Int](w * h)
    var top = 0
    for y <- 0 until h; x <- 0 until w do
      val idx = y * w + x
      val m = mag(idx)
      val (m1, m2) = dir(idx) match
        case 0 => (if x > 0 then mag(idx - 1) else 0f, if x + 1 < w then mag(idx + 1) else 0f)
        case 1 => (if x + 1 < w && y + 1 < h then mag(idx + w + 1) else 0f, if x > 0 && y > 0 then mag(idx - w - 1) else 0f)
        case 2 => (if y > 0 then mag(idx - w) else 0f, if y + 1 < h then mag(idx + w) else 0f)
        case _ => (if x > 0 && y + 1 < h then mag(idx + w - 1) else 0f, if x + 1 < w && y > 0 then mag(idx - w + 1) else 0f)
      if m >= m1 && m >= m2 then
        if m >= highThresh then
          state(idx) = 2
          stack(top) = idx
          top += 1
        else if m >= lowThresh then state(idx) = 1
    while top > 0 do
      top -= 1
      val idx = stack(top)
      val cx = idx % w
      val cy = idx / w
      for dy <- -1 to 1; dx <- -1 to 1 if dx != 0 || dy != 0 do
        val nx = cx + dx
        val ny = cy + dy
        if nx >= 0 && nx < w && ny >= 0 && ny < h then
          val nIdx = ny * w + nx
          if state(nIdx) == 1 then
            state(nIdx) = 2
            stack(top) = nIdx
            top += 1
    state.map(_ == 2)
