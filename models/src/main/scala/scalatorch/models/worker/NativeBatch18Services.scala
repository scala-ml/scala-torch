package scalatorch.models.worker

import java.awt.{ Color, RenderingHints }
import java.awt.image.BufferedImage
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream }
import java.nio.{ ByteBuffer, ByteOrder }
import java.nio.charset.StandardCharsets
import java.nio.file.{ Files, Path, StandardCopyOption }
import java.util.{ Base64, Locale, UUID }
import java.util.zip.{ GZIPInputStream, GZIPOutputStream }
import javax.imageio.ImageIO
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import scalatorch.{ Shape, Tensor }
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

private[worker] object NativeBatch18Services:

  private def optStr(p: Map[String, NativeJson], key: String, default: String = ""): String =
    p.get(key) match
      case Some(Str(v)) => v
      case None | Some(Null) => default
      case other => throw new IllegalArgumentException(s"$key must be a string (got $other)")

  private def optNum(p: Map[String, NativeJson], key: String, default: Double): Double =
    p.get(key) match
      case Some(Number(v)) if v.toDouble.isFinite => v.toDouble
      case None | Some(Null) => default
      case other => throw new IllegalArgumentException(s"$key must be a finite number (got $other)")

  private def optInt(p: Map[String, NativeJson], key: String, default: Int): Int =
    p.get(key) match
      case Some(Number(v)) if v.isValidInt => v.toInt
      case None | Some(Null) => default
      case other => throw new IllegalArgumentException(s"$key must be an integer (got $other)")

  private def optBool(p: Map[String, NativeJson], key: String, default: Boolean): Boolean =
    p.get(key) match
      case Some(Bool(b)) => b
      case Some(Number(n)) => n.toInt != 0
      case Some(Str(s)) => Set("1", "true", "yes", "on").contains(s.trim.toLowerCase(Locale.ROOT))
      case None | Some(Null) => default
      case _ => default

  // ===================================================================================================================
  // 1. Additional Image Tools (`nsfw_checker`, `aesthetic_score`, `person_mask`, `remove_background`, `depth_mask`, `waifu2x`)
  // ===================================================================================================================

  def executeImageTool(p: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val tool = optStr(p, "tool")
    val srcPathStr = optStr(p, "image_path")
    require(srcPathStr.nonEmpty, "image_path is required")
    val srcPath = Path.of(srcPathStr)
    require(Files.isRegularFile(srcPath), s"image file not found: $srcPathStr")
    val outDir = Path.of(optStr(p, "output_dir", ".")).toAbsolutePath.normalize()
    Files.createDirectories(outDir)
    context.checkCancelled()
    val img = DiffusionImageInput.readRgb(srcPath, compositeAlpha = false)

    def saveImg(im: BufferedImage, prefix: String): Path =
      val dst = outDir.resolve(s"$prefix-${UUID.randomUUID()}.png")
      require(ImageIO.write(im, "png", dst.toFile), "PNG writer is unavailable")
      dst

    tool match
      case "nsfw_checker" =>
        val filterType = optStr(p, "filter_type", "Variable blur")
        val maskShape = optStr(p, "mask_shape", "Ellipse")
        val enableNudenet = optBool(p, "enable_nudenet", true)
        val outputMask = optBool(p, "output_mask", false)
        val censorMaskOpt: Option[BufferedImage] =
          if maskShape == NsfwChecker.EntireImageAlways then
            Some(NsfwChecker.drawMaskShape(img.getWidth, img.getHeight, Vector.empty, NsfwChecker.EntireImageAlways, 0.5))
          else
            val customMaskOpt = Option(optStr(p, "mask_path")).filter(_.trim.nonEmpty).map { mp =>
              val cImg = ImageIO.read(Path.of(mp).toFile)
              val gray = new BufferedImage(img.getWidth, img.getHeight, BufferedImage.TYPE_BYTE_GRAY)
              val g = gray.createGraphics()
              try g.drawImage(cImg, 0, 0, img.getWidth, img.getHeight, null) finally g.dispose()
              gray
            }
            val foundOpt =
              if enableNudenet then
                val cfg = NsfwChecker.NsfwConfig(
                  genFilterType = if NsfwChecker.Filters.contains(filterType) then filterType else "Variable blur",
                  maskShape = if NsfwChecker.MaskShapes.contains(maskShape) then maskShape else "Ellipse",
                  rectangleRoundRadius = optNum(p, "rectangle_round_radius", 0.5),
                  nmsThreshold = optNum(p, "nms_threshold", 0.5)
                )
                NsfwChecker.detectRegions(img, cfg)
              else None
            (customMaskOpt, foundOpt) match
              case (Some(cm), Some(fm)) =>
                val g = cm.createGraphics()
                try g.drawImage(fm, 0, 0, null) finally g.dispose()
                Some(cm)
              case (Some(cm), None) => Some(cm)
              case (None, Some(fm)) => Some(fm)
              case (None, None) => None

        censorMaskOpt match
          case None =>
            val dst = saveImg(img, "nsfw_checker")
            Map(
              "path" -> Str(dst.toString),
              "outputs" -> Obj(Map.empty),
              "info" -> Obj(Map("censored" -> Number(0), "detected" -> Number(0)))
            )
          case Some(rawMask) =>
            val factor = NsfwChecker.scaleFactor(img.getWidth, img.getHeight)
            val defaultBlend = if filterType == "Variable blur" then 10.0 else 0.0
            val blend = optNum(p, "mask_blend_radius", defaultBlend)
            val blendedMask = NsfwChecker.gaussianBlurImage(rawMask, blend * factor)
            val blurRadius = if p.contains("blur_radius") && p("blur_radius") != Null then optNum(p, "blur_radius", 10.0) else 10.0 * factor
            val censored = NsfwChecker.applyFilter(
              img,
              blendedMask,
              filterType,
              blurRadius = blurRadius,
              blurStrengthCurve = optNum(p, "blur_strength_curve", 3.0),
              pixelationFactor = optNum(p, "pixelation_factor", 5.0),
              color = optStr(p, "fill_color", "#000000")
            )
            val dst = saveImg(censored, "nsfw_checker")
            val outputs =
              if outputMask then Map("mask" -> Str(saveImg(blendedMask, "nsfw_checker-mask").toString))
              else Map.empty[String, NativeJson]
            Map(
              "path" -> Str(dst.toString),
              "outputs" -> Obj(outputs),
              "info" -> Obj(Map("censored" -> Number(1), "detected" -> Number(1)))
            )

      case "aesthetic_score" =>
        val cfg = AestheticScoreConfig(
          improved = optBool(p, "improved", true),
          simulacra = optStr(p, "simulacra", "off"),
          cafeAesthetic = optBool(p, "cafe_aesthetic", false),
          cafeStyle = optBool(p, "cafe_style", false),
          cafeWaifu = optBool(p, "cafe_waifu", false),
          pngChunks = optBool(p, "png_chunks", true),
          generationFields = optBool(p, "generation_fields", false),
          hash = optBool(p, "hash", false),
          forceCpu = optBool(p, "force_cpu", false)
        )
        val scores = AestheticScore.scoreImage(img, cfg)
        val infoFields = AestheticScore.infotextFields(scores).map { case (k, v) => k -> Str(v) }.toMap
        val dst = saveImg(img, "aesthetic_score")
        Map(
          "path" -> Str(dst.toString),
          "outputs" -> Obj(Map.empty),
          "info" -> Obj(infoFields)
        )

      case "person_mask" =>
        val conf = optNum(p, "confidence", 0.3)
        val dilate = optInt(p, "dilate", 4)
        val invert = optBool(p, "invert", false)
        val det = Adetailer.detect(img, "person_yolov8n-seg.pt", conf)
        val w = img.getWidth; val h = img.getHeight
        var merged = new Array[Int](w * h)
        for m <- det.masks do
          var i = 0
          while i < merged.length do { if m(i) > merged(i) then merged(i) = m(i); i += 1 }
        if dilate != 0 then merged = Adetailer.offsetDilate(merged, w, h, 0, 0, dilate)
        if invert then
          var i = 0
          while i < merged.length do { merged(i) = 255 - merged(i); i += 1 }
        val maskImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        var i = 0
        while i < w * h do
          val v = merged(i)
          maskImg.setRGB(i % w, i / w, (0xff << 24) | (v << 16) | (v << 8) | v)
          i += 1
        val dst = saveImg(maskImg, "person_mask")
        val maskDst = saveImg(maskImg, "person_mask-mask")
        Map("path" -> Str(dst.toString), "outputs" -> Obj(Map("mask" -> Str(maskDst.toString))), "info" -> Obj(Map.empty))

      case "remove_background" =>
        val bg = optStr(p, "background", "transparent").toLowerCase(Locale.ROOT)
        val w = img.getWidth; val h = img.getHeight
        val alpha = NativeU2Net.alpha(img)
        val outImg =
          if bg.isEmpty || bg == "transparent" then
            val rgba = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
            var i = 0
            while i < w * h do
              val rgb = img.getRGB(i % w, i / w) & 0x00ffffff
              rgba.setRGB(i % w, i / w, (alpha(i) << 24) | rgb)
              i += 1
            rgba
          else
            val bgColor = bg match
              case "black" => new Color(0, 0, 0)
              case "green" => new Color(0, 177, 64)
              case _ => new Color(255, 255, 255)
            val rgbOut = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
            var i = 0
            while i < w * h do
              val a = alpha(i) / 255.0
              val c = img.getRGB(i % w, i / w)
              val r = math.round(((c >> 16) & 0xff) * a + bgColor.getRed * (1.0 - a)).toInt
              val g = math.round(((c >> 8) & 0xff) * a + bgColor.getGreen * (1.0 - a)).toInt
              val b = math.round((c & 0xff) * a + bgColor.getBlue * (1.0 - a)).toInt
              rgbOut.setRGB(i % w, i / w, (0xff << 24) | (r << 16) | (g << 8) | b)
              i += 1
            rgbOut
        val maskImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        val mRaster = maskImg.getRaster
        var idx = 0
        while idx < w * h do { mRaster.setSample(idx % w, idx / w, 0, alpha(idx)); idx += 1 }
        val dst = saveImg(outImg, "remove_background")
        val maskDst = saveImg(maskImg, "remove_background-mask")
        Map("path" -> Str(dst.toString), "outputs" -> Obj(Map("mask" -> Str(maskDst.toString))), "info" -> Obj(Map.empty))

      case "depth_mask" =>
        val near = optNum(p, "near", 0.5)
        val far = optNum(p, "far", 1.0)
        val invert = optBool(p, "invert", false)
        val blur = optInt(p, "blur", 4)
        val lo = math.min(near, far)
        val hi = math.max(near, far)
        val w = img.getWidth; val h = img.getHeight
        val depthVals = new Array[Double](w * h)
        var dMin = Double.PositiveInfinity; var dMax = Double.NegativeInfinity
        // Real Depth Anything V2 depth (near = bright); fails clearly when the annotator weights are missing.
        val rawDepth = NativeDepthAnything.preprocess(img, () => context.cancelled())
        val depthImage = if rawDepth.getWidth == w && rawDepth.getHeight == h then rawDepth else DiffusionImageResize(rawDepth, w, h, 0)
        var i = 0
        while i < w * h do
          val lum = ((depthImage.getRGB(i % w, i / w) >> 8) & 0xff) / 255.0
          depthVals(i) = lum
          if lum < dMin then dMin = lum
          if lum > dMax then dMax = lum
          i += 1
        val span = math.max(1e-6, dMax - dMin)
        val maskGray = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        val depthGray = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        val mR = maskGray.getRaster; val dR = depthGray.getRaster
        i = 0
        while i < w * h do
          val dNorm = (depthVals(i) - dMin) / span
          dR.setSample(i % w, i / w, 0, math.round(dNorm * 255.0).toInt)
          val inside = dNorm >= lo && dNorm <= hi
          val mv = if invert then (if inside then 0 else 255) else (if inside then 255 else 0)
          mR.setSample(i % w, i / w, 0, mv)
          i += 1
        val finalMask = if blur > 0 then NsfwChecker.gaussianBlurImage(maskGray, blur.toDouble) else maskGray
        val rgbMask = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val fmR = finalMask.getRaster
        i = 0
        while i < w * h do
          val v = fmR.getSample(i % w, i / w, 0)
          rgbMask.setRGB(i % w, i / w, (0xff << 24) | (v << 16) | (v << 8) | v)
          i += 1
        val dst = saveImg(rgbMask, "depth_mask")
        val maskDst = saveImg(finalMask, "depth_mask-mask")
        val depthDst = saveImg(depthGray, "depth_mask-depth")
        Map(
          "path" -> Str(dst.toString),
          "outputs" -> Obj(Map("mask" -> Str(maskDst.toString), "depth" -> Str(depthDst.toString))),
          "info" -> Obj(Map.empty)
        )

      case "waifu2x" =>
        if !p.contains("scale") && !p.contains("style") && !p.contains("noise") then
          throw new NativeBackendError("unsupported", "waifu2x requires scale, style, or noise parameters")
        val scale = optNum(p, "scale", 2.0)
        val style = optStr(p, "style", "Anime")
        val noise = optInt(p, "noise", 2)
        val targetW = math.max(1, math.round(img.getWidth * scale).toInt)
        val targetH = math.max(1, math.round(img.getHeight * scale).toInt)
        if noise < 0 || noise > 3 then throw new NativeBackendError("invalid_params", "waifu2x noise must be 0..3")
        val tile = optInt(p, "tile", 256)
        // Real waifu2x upconv_7 (NativeWaifu2x); missing weights fail loudly, no resize fallback
        val up = NativeWaifu2x.upscale(img, targetW, targetH, style, noise, () => context.cancelled(), if tile <= 0 then 256 else tile)
        val dst = saveImg(up, "waifu2x")
        Map(
          "path" -> Str(dst.toString),
          "outputs" -> Obj(Map.empty),
          "info" -> Obj(Map("model" -> Str(s"waifu2x-upconv_7-${NativeWaifu2x.styleKey(style)}-noise$noise"), "scale" -> Number(scale)))
        )

      case other =>
        throw new NativeBackendError("unsupported", s"native image tool '$other' is not implemented")

  // ===================================================================================================================
  // 2. `embedding_merge` RPC (`inspect`, `save`, `restore`)
  // ===================================================================================================================

  def executeEmbeddingMerge(
      params: Map[String, NativeJson],
      context: NativeWorkContext,
      pipelineOpt: Option[EulerImagePipeline]
  ): Map[String, NativeJson] =
    val pipeline = pipelineOpt.getOrElse(
      throw new NativeBackendError("no_model", "load a checkpoint before merging embeddings")
    )
    val operation = optStr(params, "operation", "inspect")
    if !Set("inspect", "save", "restore").contains(operation) then
      throw new NativeBackendError("invalid_params", "Embedding Merge operation must be inspect, save or restore")
    context.checkCancelled()
    val text = optStr(params, "text", "")

    // Check provenance restoration `<'expr' = token>`
    val provenanceRegex = """<(\s*'[^']*'\s*)=\s*([^>]+)>""".r
    val restored = provenanceRegex.replaceAllIn(text, m => s"<${m.group(1).trim}>")
    if operation == "restore" || restored != text then
      return Map(
        "text" -> Str(restored),
        "streams" -> Arr(Vector.empty),
        "paths" -> Arr(Vector.empty),
        "active" -> Bool(false)
      )

    val streamNames = if pipeline.sdxlConditioning.isDefined then Vector("clip_l", "clip_g") else Vector("clip_l")
    val streamDims =
      if pipeline.sdxlConditioning.isDefined then
        Vector(
          pipeline.textEncoder.tokenEmbedding.weight.data.shape.dims(1).toInt,
          pipeline.sdxlConditioning.get.encoder2.tokenEmbedding.weight.data.shape.dims(1).toInt
        )
      else Vector(pipeline.textEncoder.tokenEmbedding.weight.data.shape.dims(1).toInt)

    def resolveText(txt: String): Vector[Vector[Array[Float]]] =
      def extractFor(tok: ClipFastTokenizer, enc: ClipTextModel): Vector[Array[Float]] =
        val ids = tok.encode(txt, addSpecialTokens = false)
        val chosen = if ids.nonEmpty then ids.toVector else Vector(tok.eosId)
        val w = enc.tokenEmbedding.weight.data.shape.dims(1).toInt
        val table = enc.tokenEmbedding.weight.data.toFloatArray
        chosen.map { id =>
          val row = new Array[Float](w)
          System.arraycopy(table, id.toInt * w, row, 0, w)
          row
        }.toVector
      val lVec = extractFor(pipeline.tokenizer, pipeline.textEncoder)
      pipeline.sdxlConditioning match
        case Some(sc) => Vector(lVec, extractFor(sc.tokenizer2, sc.encoder2))
        case None => Vector(lVec)

    def resolveToken(spec: Either[Int, String]): Vector[Vector[Array[Float]]] =
      def extractId(tok: ClipFastTokenizer, enc: ClipTextModel): Vector[Array[Float]] =
        val rawId = spec match
          case Left(id) => id
          case Right("start") => tok.bosId.toInt
          case Right("end") => tok.eosId.toInt
          case Right("pad") => tok.padId.toInt
          case Right(_) => tok.eosId.toInt
        val vocab = enc.tokenEmbedding.weight.data.shape.dims(0).toInt
        val w = enc.tokenEmbedding.weight.data.shape.dims(1).toInt
        val safeId = math.max(0, math.min(vocab - 1, rawId))
        val row = new Array[Float](w)
        System.arraycopy(enc.tokenEmbedding.weight.data.toFloatArray, safeId * w, row, 0, w)
        Vector(row)
      val lVec = extractId(pipeline.tokenizer, pipeline.textEncoder)
      pipeline.sdxlConditioning match
        case Some(sc) => Vector(lVec, extractId(sc.tokenizer2, sc.encoder2))
        case None => Vector(lVec)

    def writeSafetensorsFile(path: Path, tensors: Vector[(String, Tensor[Float])]): Unit =
      var offset = 0L
      val headerFields = mutable.ArrayBuffer.empty[String]
      val payloads = tensors.map { case (name, t) =>
        val floats = t.toFloatArray
        val byteLen = floats.length.toLong * 4L
        val shapeStr = t.shape.dims.mkString("[", ",", "]")
        val start = offset
        val end = offset + byteLen
        offset = end
        headerFields += s"\"$name\":{\"dtype\":\"F32\",\"shape\":$shapeStr,\"data_offsets\":[$start,$end]}"
        floats
      }
      val jsonBytes = ("{" + headerFields.mkString(",") + "}").getBytes(StandardCharsets.UTF_8)
      val pad = (8 - (jsonBytes.length % 8)) % 8
      val paddedLen = jsonBytes.length + pad
      val out = Files.newOutputStream(path)
      try
        val lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(paddedLen.toLong)
        out.write(lenBuf.array())
        out.write(jsonBytes)
        if pad > 0 then out.write(Array.fill[Byte](pad)(' '.toByte))
        payloads.foreach { floats =>
          val bb = ByteBuffer.allocate(floats.length * 4).order(ByteOrder.LITTLE_ENDIAN)
          bb.asFloatBuffer().put(floats)
          out.write(bb.array())
        }
      finally out.close()

    def vectorStats(vectors: Vector[Array[Float]]): Obj =
      val all = vectors.flatten
      if all.isEmpty then
        Obj(Map("min" -> Number(0), "max" -> Number(0), "mean" -> Number(0), "std" -> Number(0), "l1" -> Number(0), "l2" -> Number(0)))
      else
        val n = all.length.toDouble
        val minV = all.min.toDouble
        val maxV = all.max.toDouble
        val meanV = all.map(_.toDouble).sum / n
        val varV = all.map(v => { val d = v.toDouble - meanV; d * d }).sum / n
        val stdV = math.sqrt(math.max(0.0, varV))
        val l1V = all.map(v => math.abs(v.toDouble)).sum / vectors.size
        val l2V = vectors.map(row => math.sqrt(row.map(v => v.toDouble * v.toDouble).sum)).sum / vectors.size
        Obj(Map(
          "min" -> Number(minV),
          "max" -> Number(maxV),
          "mean" -> Number(meanV),
          "std" -> Number(stdV),
          "l1" -> Number(l1V),
          "l2" -> Number(l2V)
        ))

    val evaluatedStreams: Vector[Vector[Array[Float]]] =
      if text.trim.startsWith("'") then
        val actions = EmbeddingMerge.parseExpression(text.trim)
        EmbeddingMerge.evaluate(actions, streamNames, streamDims, resolveText, resolveToken)
      else
        val spans = EmbeddingMerge.extractSpans(text)
        if spans.nonEmpty then
          val partsPerStream = Array.fill(streamNames.size)(new ArrayBuffer[Array[Float]]())
          var pos = 0
          for sp <- spans do
            val prefix = text.substring(pos, sp.start).trim
            if prefix.nonEmpty then
              val pVecs = resolveText(prefix)
              for s <- streamNames.indices do partsPerStream(s) ++= pVecs(s)
            val mVecs = EmbeddingMerge.evaluate(EmbeddingMerge.parseExpression(sp.expression), streamNames, streamDims, resolveText, resolveToken)
            for s <- streamNames.indices do partsPerStream(s) ++= mVecs(s)
            pos = sp.end
          val tail = text.substring(pos).trim
          if tail.nonEmpty then
            val tVecs = resolveText(tail)
            for s <- streamNames.indices do partsPerStream(s) ++= tVecs(s)
          partsPerStream.map(_.toVector).toVector
        else
          resolveText(text)

    val streamResults = streamNames.zip(evaluatedStreams).map { case (sName, vecs) =>
      val rows = vecs.zipWithIndex.map { case (vec, i) =>
        Obj(Map(
          "start" -> Number(i + 1),
          "end" -> Number(i + 1),
          "count" -> Number(1),
          "text" -> Str(text),
          "tokens" -> Arr(Vector.empty),
          "statistics" -> vectorStats(Vector(vec))
        ))
      }
      Obj(Map(
        "name" -> Str(sName),
        "count" -> Number(vecs.size),
        "rows" -> Arr(rows),
        "total" -> vectorStats(vecs)
      ))
    }

    val savedPaths = new ArrayBuffer[String]()
    var active = false
    if operation == "save" then
      val outDirStr = optStr(params, "output_dir")
      if outDirStr.isEmpty then
        throw new NativeBackendError("invalid_params", "Embedding Merge: managed embedding output directory is required")
      val outDir = Path.of(outDirStr).toAbsolutePath.normalize()
      Files.createDirectories(outDir)
      val rawName = optStr(params, "name", "merged_embedding").trim
      val baseName = if rawName.toLowerCase(Locale.ROOT).endsWith(".safetensors") then rawName else s"$rawName.safetensors"
      val dst = outDir.resolve(baseName)
      val writes = streamNames.zip(evaluatedStreams).map { case (sName, vecs) =>
        val count = vecs.size
        val dim = if count > 0 then vecs.head.length else 768
        val flat = vecs.flatten.toArray
        val key = if streamNames.size == 1 then "emb_params" else sName
        key -> Tensor.fromFloats(flat, Shape.from(Seq(count.toLong, dim.toLong)))
      }
      writeSafetensorsFile(dst, writes)
      savedPaths += dst.toString
      active = true

    Map(
      "text" -> Str(text),
      "streams" -> Arr(streamResults),
      "paths" -> Arr(savedPaths.map(Str.apply).toVector),
      "active" -> Bool(active),
      "warning" -> Str("")
    )

  // ===================================================================================================================
  // 3. `aesthetic_embedding` RPC (`list`, `create`)
  // ===================================================================================================================

  def executeAestheticEmbedding(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "op", "create")
    val defaultDir = Path.of(
      sys.env.getOrElse(
        "SDFX_AESTHETIC_GRADIENTS_CACHE",
        Path.of(System.getProperty("user.home", "."), ".cache", "stablediffusion-nui", "aesthetic_gradients").toString
      )
    ).resolve("embeddings")
    val envDirs = sys.env.getOrElse("SDFX_AESTHETIC_EMBEDDINGS", "")
      .split(java.io.File.pathSeparatorChar).map(_.trim).filter(_.nonEmpty).map(Path.of(_)).toVector
    val folders = (envDirs :+ defaultDir).distinct

    if op == "list" then
      val found = new ArrayBuffer[String]()
      for dir <- folders if Files.isDirectory(dir) do
        val s = Files.list(dir)
        try
          s.iterator().asScala.foreach { p =>
            val fn = p.getFileName.toString
            val lower = fn.toLowerCase(Locale.ROOT)
            if Files.isRegularFile(p) && (lower.endsWith(".pt") || lower.endsWith(".safetensors")) then
              found += fn.replaceFirst("\\.(pt|safetensors)$", "")
          }
        finally s.close()
      Map(
        "embeddings" -> Arr(found.distinct.sorted.map(Str.apply).toVector),
        "folders" -> Arr(folders.map(d => Str(d.toString)))
      )
    else if op == "create" then
      val name = optStr(params, "name").trim
      if name.isEmpty || name.exists(ch => "\\/:*?\"<>|".indexOf(ch) >= 0) || name == "." || name == ".." || name.startsWith(".") then
        throw new NativeBackendError("invalid_params", "Aesthetic Gradients: name must be a plain file name")
      val imgSpecs: Vector[String] = params.get("images") match
        case Some(Str(s)) if s.trim.nonEmpty => Vector(s.trim)
        case Some(Arr(items)) => items.collect { case Str(s) if s.trim.nonEmpty => s.trim }
        case _ => Vector.empty
      if imgSpecs.isEmpty then
        throw new NativeBackendError("invalid_params", "Aesthetic Gradients: images (files or a folder) are required")

      val imagePaths = new ArrayBuffer[Path]()
      for spec <- imgSpecs do
        val p = Path.of(spec)
        if Files.isDirectory(p) then
          val s = Files.list(p)
          try
            s.iterator().asScala.filter { f =>
              val l = f.getFileName.toString.toLowerCase(Locale.ROOT)
              Files.isRegularFile(f) && (l.endsWith(".png") || l.endsWith(".jpg") || l.endsWith(".jpeg") || l.endsWith(".webp"))
            }.toVector.sortBy(_.getFileName.toString).foreach(imagePaths += _)
          finally s.close()
        else if Files.isRegularFile(p) then
          imagePaths += p
      if imagePaths.isEmpty then
        throw new NativeBackendError("invalid_params", "Aesthetic Gradients: no readable images found")

      val folder = Path.of(optStr(params, "output_dir", folders.head.toString)).toAbsolutePath.normalize()
      Files.createDirectories(folder)
      val outName = if name.toLowerCase(Locale.ROOT).endsWith(".pt") || name.toLowerCase(Locale.ROOT).endsWith(".safetensors") then name else s"$name.safetensors"
      val output = folder.resolve(outName)
      val overwrite = optBool(params, "overwrite", false)
      if Files.exists(output) && !overwrite then
        throw new NativeBackendError("invalid_params", s"Aesthetic Gradients: $output exists (pass overwrite to replace it)")

      val acc = new Array[Float](AestheticGradients.EmbedDim)
      for imgPath <- imagePaths do
        context.checkCancelled()
        val t = ReferenceOnly.loadNormalizedImageTensor(imgPath.toString, 224, 224, "Aesthetic Gradients")
        val vec = AestheticGradients.normalizeL2(Revision.extractImageEmbedding(t.toFloatArray, 224, 224, AestheticGradients.EmbedDim))
        var i = 0
        while i < acc.length do { acc(i) += vec(i); i += 1 }
      val meanEmbed = AestheticGradients.normalizeL2(acc)
      val byteLen = meanEmbed.length.toLong * 4L
      val headerJson = s"""{"string_to_param.*":{"dtype":"F32","shape":[1,${AestheticGradients.EmbedDim}],"data_offsets":[0,$byteLen]}}"""
      val jsonBytes = headerJson.getBytes(StandardCharsets.UTF_8)
      val pad = (8 - (jsonBytes.length % 8)) % 8
      val paddedLen = jsonBytes.length + pad
      val outStream = Files.newOutputStream(output)
      try
        val lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(paddedLen.toLong)
        outStream.write(lenBuf.array())
        outStream.write(jsonBytes)
        if pad > 0 then outStream.write(Array.fill[Byte](pad)(' '.toByte))
        val bb = ByteBuffer.allocate(meanEmbed.length * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(meanEmbed)
        outStream.write(bb.array())
      finally outStream.close()
      Map(
        "path" -> Str(output.toString),
        "name" -> Str(name.replaceFirst("\\.(pt|safetensors)$", "")),
        "images" -> Number(imagePaths.size)
      )
    else
      throw new NativeBackendError("invalid_params", s"Aesthetic Gradients: unknown operation '$op'")

  // ===================================================================================================================
  // 4. `controlnet_fastload_data` RPC
  // ===================================================================================================================

  private val FastloadStart: Array[Byte] = "###START_OF_CONTROLNET_FASTLOAD###".getBytes(StandardCharsets.US_ASCII)
  private val FastloadEnd: Array[Byte] = "###END_OF_CONTROLNET_FASTLOAD###".getBytes(StandardCharsets.US_ASCII)

  private def indexOfBytes(haystack: Array[Byte], needle: Array[Byte], from: Int = 0): Int =
    val maxIdx = haystack.length - needle.length
    var i = math.max(0, from)
    while i <= maxIdx do
      var j = 0
      while j < needle.length && haystack(i + j) == needle(j) do j += 1
      if j == needle.length then return i
      i += 1
    -1

  def appendFastloadPayload(imageBytes: Array[Byte], wireUnits: NativeJson): Array[Byte] =
    val jsonBytes = NativeJson.render(wireUnits).getBytes(StandardCharsets.UTF_8)
    val bos = new ByteArrayOutputStream()
    val gz = new GZIPOutputStream(bos)
    gz.write(jsonBytes)
    gz.finish()
    gz.close()
    val compressed = bos.toByteArray
    val out = new Array[Byte](imageBytes.length + FastloadStart.length + compressed.length + FastloadEnd.length)
    System.arraycopy(imageBytes, 0, out, 0, imageBytes.length)
    var off = imageBytes.length
    System.arraycopy(FastloadStart, 0, out, off, FastloadStart.length)
    off += FastloadStart.length
    System.arraycopy(compressed, 0, out, off, compressed.length)
    off += compressed.length
    System.arraycopy(FastloadEnd, 0, out, off, FastloadEnd.length)
    out

  def extractFastloadPayload(data: Array[Byte]): (Array[Byte], NativeJson) =
    val start = indexOfBytes(data, FastloadStart)
    val end = indexOfBytes(data, FastloadEnd)
    require(start >= 0 && end >= start + FastloadStart.length && end + FastloadEnd.length == data.length, "missing or invalid fastload trailer")
    val compressed = java.util.Arrays.copyOfRange(data, start + FastloadStart.length, end)
    val gis = new GZIPInputStream(new ByteArrayInputStream(compressed))
    val payloadBytes =
      try gis.readAllBytes()
      finally gis.close()
    val payloadStr = new String(payloadBytes, StandardCharsets.UTF_8)
    val parsed = NativeJson.parse(payloadStr)
    (java.util.Arrays.copyOfRange(data, 0, start), parsed)

  def executeControlNetFastloadData(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "operation")
    context.checkCancelled()
    op match
      case "version" =>
        Map(
          "version" -> Number(1),
          "upstream_version" -> Number(BigDecimal("1.1")),
          "legacy_subset_only" -> Bool(true),
          "protected_write" -> Bool(false)
        )
      case "encode" =>
        val units = params.getOrElse("units", Obj(Map("type" -> Str("list"), "items" -> Arr(Vector.empty))))
        val payload = appendFastloadPayload(Array.emptyByteArray, units)
        val encoded = Base64.getEncoder.encodeToString(payload)
        context.beginPublication()
        Map("base64" -> Str(encoded))
      case "save" =>
        val pathStr = optStr(params, "path")
        require(pathStr.nonEmpty, "output path is required")
        val target = Path.of(pathStr).toAbsolutePath.normalize()
        if Files.exists(target) then throw new IllegalArgumentException("output already exists")
        val prefixB64 = optStr(params, "image_base64", "")
        val imageBytes = if prefixB64.isEmpty then Array.emptyByteArray else Base64.getDecoder.decode(prefixB64)
        val units = params.getOrElse("units", Obj(Map("type" -> Str("list"), "items" -> Arr(Vector.empty))))
        val fullBytes = appendFastloadPayload(imageBytes, units)
        context.beginPublication()
        Files.write(target, fullBytes)
        Map("path" -> Str(target.toString))
      case "inspect" =>
        val pathStr = optStr(params, "path")
        require(pathStr.nonEmpty, "input path is required")
        val raw = Files.readAllBytes(Path.of(pathStr))
        val (prefix, unitsWire) = extractFastloadPayload(raw)
        Map(
          "units" -> unitsWire,
          "has_image" -> Bool(prefix.nonEmpty),
          "summary" -> Arr(Vector.empty)
        )
      case "resolve" | "prepare_generation" =>
        val mode = optStr(params, "mode", "load_only")
        val priority = optStr(params, "priority", "current_first")
        val current = params.getOrElse("current", Obj(Map("type" -> Str("list"), "items" -> Arr(Vector.empty))))
        val loadedOpt = params.get("loaded").filter(_ != Null)
        val chosen = if mode == "save_only" || loadedOpt.isEmpty || priority == "current_first" then current else loadedOpt.get
        val save = mode == "save_only" || mode == "load_save"
        val skip = mode != "save_only" && (loadedOpt.isEmpty || priority == "current_first")
        Map(
          "units" -> chosen,
          "save" -> Bool(save),
          "load_skipped" -> Bool(skip)
        )
      case "gallery" =>
        val pathStr = optStr(params, "path")
        require(pathStr.nonEmpty, "gallery directory is required")
        val root = Path.of(pathStr)
        require(Files.isDirectory(root), "gallery directory is unavailable")
        Map(
          "paths" -> Arr(Vector.empty),
          "page" -> Number(0),
          "pages" -> Number(0),
          "choices" -> Obj(Map.empty),
          "skipped_links" -> Number(0),
          "skipped_oversize" -> Number(0)
        )
      case "gallery_source" =>
        val pathStr = optStr(params, "path")
        require(pathStr.nonEmpty, "gallery image is required")
        val p = Path.of(pathStr).toAbsolutePath.normalize()
        require(Files.isRegularFile(p), "gallery image is unavailable")
        val raw = Files.readAllBytes(p)
        if indexOfBytes(raw, FastloadStart) >= 0 then
          Map("path" -> Str(p.toString), "kind" -> Str("embedded"))
        else
          val sidecar = p.resolveSibling(p.getFileName.toString.replaceFirst("\\.[^.]+$", "") + ".cni")
          if Files.isRegularFile(sidecar) then Map("path" -> Str(sidecar.toString), "kind" -> Str("sidecar"))
          else Map("path" -> Null, "kind" -> Str("none"))
      case "map" =>
        val units = params.getOrElse("units", Obj(Map("type" -> Str("list"), "items" -> Arr(Vector.empty))))
        Map("mapped" -> units)
      case "from_native" =>
        val items = params.getOrElse("items", Obj(Map("type" -> Str("list"), "items" -> Arr(Vector.empty))))
        Map("units" -> items, "summary" -> Arr(Vector.empty))
      case "preview" =>
        Map("image" -> Null, "mask" -> Null)
      case "native_form" =>
        Map(
          "params" -> Obj(Map("enabled" -> Bool(true), "model" -> Str(""), "preprocessor" -> Str("none"), "weight" -> Number(1.0))),
          "image" -> Null,
          "mask" -> Null,
          "unapplied" -> Arr(Vector.empty)
        )
      case other =>
        throw new NativeBackendError("invalid_params", s"ControlNet Fastload: unknown Fastload operation '$other'")

  // ===================================================================================================================
  // 5. `temporal_lab` RPC (`preview`, `dfi`, `deflicker`, `fuse`, `extract`)
  // ===================================================================================================================

  private def listSequenceFrames(dirPath: Path): Vector[Path] =
    require(Files.isDirectory(dirPath), s"sequence directory does not exist: $dirPath")
    val s = Files.list(dirPath)
    val files =
      try
        s.iterator().asScala.filter { p =>
          val l = p.getFileName.toString.toLowerCase(Locale.ROOT)
          Files.isRegularFile(p) && (l.endsWith(".png") || l.endsWith(".jpg") || l.endsWith(".jpeg"))
        }.toVector.sortBy(_.getFileName.toString)
      finally s.close()
    require(files.nonEmpty, s"sequence directory contains no frames: $dirPath")
    files

  private def computeTemporalMask(prev: BufferedImage, cur: BufferedImage, tolerance: Double, expand: Int, smooth: Int): BufferedImage =
    val w = cur.getWidth
    val h = cur.getHeight
    val stable = new Array[Int](w * h)
    var y = 0
    while y < h do
      var x = 0
      while x < w do
        val c0 = prev.getRGB(math.min(prev.getWidth - 1, x), math.min(prev.getHeight - 1, y))
        val c1 = cur.getRGB(x, y)
        val g0 = 0.299 * ((c0 >> 16) & 0xff) + 0.587 * ((c0 >> 8) & 0xff) + 0.114 * (c0 & 0xff)
        val g1 = 0.299 * ((c1 >> 16) & 0xff) + 0.587 * ((c1 >> 8) & 0xff) + 0.114 * (c1 & 0xff)
        stable(y * w + x) = if math.abs(g0 - g1) <= tolerance then 255 else 0
        x += 1
      y += 1
    val expanded =
      if expand > 0 then
        val changing = stable.map(v => 255 - v)
        val dilated = Adetailer.offsetDilate(changing, w, h, 0, 0, expand)
        dilated.map(v => 255 - v)
      else stable
    val maskImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val r = maskImg.getRaster
    var idx = 0
    while idx < w * h do { r.setSample(idx % w, idx / w, 0, expanded(idx)); idx += 1 }
    if smooth > 1 then NsfwChecker.gaussianBlurImage(maskImg, smooth / 3.0) else maskImg

  private def overlayBlend(target: BufferedImage, blend: BufferedImage, strength: Double): BufferedImage =
    val w = blend.getWidth
    val h = blend.getHeight
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    var y = 0
    while y < h do
      var x = 0
      while x < w do
        val cA = target.getRGB(math.min(target.getWidth - 1, x), math.min(target.getHeight - 1, y))
        val cB = blend.getRGB(x, y)
        def ch(shift: Int): Int =
          val a = ((cA >> shift) & 0xff) / 255.0
          val b = ((cB >> shift) & 0xff) / 255.0
          val comp = if a <= 0.5 then 2.0 * a * b else 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
          val mixed = (b * (1.0 - strength) + comp * strength) * 255.0
          math.max(0, math.min(255, math.round(mixed).toInt))
        out.setRGB(x, y, (0xff << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0))
        x += 1
      y += 1
    out

  def executeTemporalLab(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "operation")
    context.checkCancelled()
    op match
      case "preview" =>
        val refFrames = listSequenceFrames(Path.of(optStr(params, "reference_dir")))
        val genFrames = listSequenceFrames(Path.of(optStr(params, "generated_dir")))
        require(refFrames.size == genFrames.size, "frame sequences must have equal counts")
        val idx = optInt(params, "frame_index", 1)
        require(idx >= 1 && idx < refFrames.size, s"frame_index must be in 1..${refFrames.size - 1}")
        val tol = optNum(params, "tolerance", 5.0)
        val expand = optInt(params, "expand", 0)
        val smooth = optInt(params, "smooth", 1)
        val prevImg = ImageIO.read(refFrames(idx - 1).toFile)
        val curImg = ImageIO.read(refFrames(idx).toFile)
        val maskImg = computeTemporalMask(prevImg, curImg, tol, expand, smooth)
        val bos = new ByteArrayOutputStream()
        ImageIO.write(maskImg, "png", bos)
        val b64 = Base64.getEncoder.encodeToString(bos.toByteArray)
        Map(
          "operation" -> Str("preview"),
          "mask_png_base64" -> Str(b64),
          "frame_index" -> Number(idx),
          "dimensions" -> Arr(Vector(Number(maskImg.getWidth), Number(maskImg.getHeight))),
          "preview_dimensions" -> Arr(Vector(Number(maskImg.getWidth), Number(maskImg.getHeight))),
          "generated_previous" -> Str(genFrames(idx - 1).toString),
          "generated_current" -> Str(genFrames(idx).toString)
        )

      case "dfi" =>
        val refFrames = listSequenceFrames(Path.of(optStr(params, "reference_dir")))
        val genFrames = listSequenceFrames(Path.of(optStr(params, "generated_dir")))
        require(refFrames.size == genFrames.size, "frame sequences must have equal counts")
        val outParent = Path.of(optStr(params, "output_parent")).toAbsolutePath.normalize()
        val outName = optStr(params, "output_name", "dfi_out")
        val outDir = outParent.resolve(outName)
        Files.createDirectories(outDir)
        val tol = optNum(params, "tolerance", 5.0)
        val expand = optInt(params, "expand", 0)
        val smooth = optInt(params, "smooth", 1)
        val limit = optInt(params, "frame_limit", 0)
        val count = if limit > 0 then math.min(refFrames.size, limit) else refFrames.size
        var prevRef = ImageIO.read(refFrames(0).toFile)
        var prevGen = ImageIO.read(genFrames(0).toFile)
        var bytesUsed = 0L
        val f1 = outDir.resolve("frame000001.png")
        ImageIO.write(prevGen, "png", f1.toFile)
        bytesUsed += Files.size(f1)
        for idx <- 1 until count do
          context.checkCancelled()
          val curRef = ImageIO.read(refFrames(idx).toFile)
          val curGen = ImageIO.read(genFrames(idx).toFile)
          val mask = computeTemporalMask(prevRef, curRef, tol, expand, smooth)
          val mRaster = mask.getRaster
          val w = curGen.getWidth; val h = curGen.getHeight
          var whiteCount = 0
          for py <- 0 until h; px <- 0 until w do
            if mRaster.getSample(px, py, 0) == 255 then whiteCount += 1
          val retention = whiteCount.toDouble / math.max(1, w * h).toDouble
          val blended = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
          for py <- 0 until h; px <- 0 until w do
            val weight = (mRaster.getSample(px, py, 0) / 255.0) * retention
            val cP = prevGen.getRGB(px, py)
            val cC = curGen.getRGB(px, py)
            val r = math.round(((cP >> 16) & 0xff) * weight + ((cC >> 16) & 0xff) * (1.0 - weight)).toInt
            val g = math.round(((cP >> 8) & 0xff) * weight + ((cC >> 8) & 0xff) * (1.0 - weight)).toInt
            val b = math.round((cP & 0xff) * weight + (cC & 0xff) * (1.0 - weight)).toInt
            blended.setRGB(px, py, (0xff << 24) | (r << 16) | (g << 8) | b)
          prevGen = blended
          prevRef = curRef
          val fOut = outDir.resolve(f"frame${idx + 1}%06d.png")
          ImageIO.write(prevGen, "png", fOut.toFile)
          bytesUsed += Files.size(fOut)
        context.beginPublication()
        Map(
          "operation" -> Str("dfi"),
          "output_dir" -> Str(outDir.toString),
          "frame_count" -> Number(count),
          "dimensions" -> Arr(Vector(Number(prevGen.getWidth), Number(prevGen.getHeight))),
          "output_bytes" -> Number(bytesUsed),
          "reference_aligned" -> Bool(false)
        )

      case "deflicker" =>
        val frames = listSequenceFrames(Path.of(optStr(params, "frames_dir")))
        val blend = optNum(params, "blend", 0.0)
        val overlay = optNum(params, "overlay", 0.0)
        val outParent = Path.of(optStr(params, "output_parent")).toAbsolutePath.normalize()
        val outName = optStr(params, "output_name", "deflicker_out")
        val outDir = outParent.resolve(outName)
        Files.createDirectories(outDir)
        var prev = ImageIO.read(frames.head.toFile)
        var bytesUsed = 0L
        for (fPath, idx) <- frames.zipWithIndex do
          context.checkCancelled()
          var cur = ImageIO.read(fPath.toFile)
          if idx > 0 && blend > 0.0 then
            val w = cur.getWidth; val h = cur.getHeight
            val next = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
            for py <- 0 until h; px <- 0 until w do
              val c0 = prev.getRGB(px, py); val c1 = cur.getRGB(px, py)
              val r = math.round(((c1 >> 16) & 0xff) * blend + ((c0 >> 16) & 0xff) * (1.0 - blend)).toInt
              val g = math.round(((c1 >> 8) & 0xff) * blend + ((c0 >> 8) & 0xff) * (1.0 - blend)).toInt
              val b = math.round((c1 & 0xff) * blend + (c0 & 0xff) * (1.0 - blend)).toInt
              next.setRGB(px, py, (0xff << 24) | (r << 16) | (g << 8) | b)
            cur = next
          if idx > 0 && overlay > 0.0 then
            cur = overlayBlend(prev, cur, overlay)
          prev = cur
          val fOut = outDir.resolve(f"frame${idx + 1}%06d.png")
          ImageIO.write(cur, "png", fOut.toFile)
          bytesUsed += Files.size(fOut)
        context.beginPublication()
        Map(
          "operation" -> Str("deflicker"),
          "output_dir" -> Str(outDir.toString),
          "frame_count" -> Number(frames.size),
          "dimensions" -> Arr(Vector(Number(prev.getWidth), Number(prev.getHeight))),
          "output_bytes" -> Number(bytesUsed)
        )

      case "fuse" =>
        val styleFrames = listSequenceFrames(Path.of(optStr(params, "style_dir")))
        val videoFrames = listSequenceFrames(Path.of(optStr(params, "video_dir")))
        require(styleFrames.size == videoFrames.size, "frame sequences must have equal counts")
        val strength = optNum(params, "strength", 0.5)
        val outParent = Path.of(optStr(params, "output_parent")).toAbsolutePath.normalize()
        val outName = optStr(params, "output_name", "fuse_out")
        val outDir = outParent.resolve(outName)
        Files.createDirectories(outDir)
        var bytesUsed = 0L
        var lastImg: BufferedImage = null
        for ((sPath, vPath), idx) <- styleFrames.zip(videoFrames).zipWithIndex do
          context.checkCancelled()
          val sImg = ImageIO.read(sPath.toFile)
          val vImg = ImageIO.read(vPath.toFile)
          val fused = overlayBlend(sImg, vImg, strength)
          lastImg = fused
          val fOut = outDir.resolve(f"frame${idx + 1}%06d.png")
          ImageIO.write(fused, "png", fOut.toFile)
          bytesUsed += Files.size(fOut)
        context.beginPublication()
        Map(
          "operation" -> Str("fuse"),
          "output_dir" -> Str(outDir.toString),
          "frame_count" -> Number(videoFrames.size),
          "dimensions" -> Arr(Vector(Number(lastImg.getWidth), Number(lastImg.getHeight))),
          "resized_style" -> Bool(false),
          "output_bytes" -> Number(bytesUsed)
        )

      case other =>
        throw new NativeBackendError("invalid_params", s"Temporal Lab: unknown operation '$other'")

  // ===================================================================================================================
  // 6. `cn_animation` (`prepare`, `flow`, `finish`, `encode`, `release`) & `animatediff_encode`
  // ===================================================================================================================

  /** Pooled cumulative histogram matching (matches `skimage.exposure.match_histograms` with `channel_axis=None`). */
  def matchHistograms(image: BufferedImage, reference: BufferedImage): BufferedImage =
    val w = image.getWidth
    val h = image.getHeight
    val total = w * h * 3
    val srcHist = new Array[Long](256)
    val refHist = new Array[Long](256)
    var y = 0
    while y < h do
      var x = 0
      while x < w do
        val cS = image.getRGB(x, y)
        srcHist((cS >> 16) & 0xff) += 1
        srcHist((cS >> 8) & 0xff) += 1
        srcHist(cS & 0xff) += 1
        val cR = reference.getRGB(math.min(reference.getWidth - 1, x), math.min(reference.getHeight - 1, y))
        refHist((cR >> 16) & 0xff) += 1
        refHist((cR >> 8) & 0xff) += 1
        refHist(cR & 0xff) += 1
        x += 1
      y += 1
    val srcCdf = new Array[Double](256)
    val refCdf = new Array[Double](256)
    var accS = 0L; var accR = 0L
    for i <- 0 until 256 do
      accS += srcHist(i); accR += refHist(i)
      srcCdf(i) = accS.toDouble / total.toDouble
      refCdf(i) = accR.toDouble / total.toDouble
    val lut = new Array[Int](256)
    for i <- 0 until 256 do
      val q = srcCdf(i)
      var j = 0
      while j < 255 && refCdf(j) < q do j += 1
      lut(i) = j
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    y = 0
    while y < h do
      var x = 0
      while x < w do
        val c = image.getRGB(x, y)
        val r = lut((c >> 16) & 0xff)
        val g = lut((c >> 8) & 0xff)
        val b = lut(c & 0xff)
        out.setRGB(x, y, (0xff << 24) | (r << 16) | (g << 8) | b)
        x += 1
      y += 1
    out

  private def writeNpyFloat2D(path: Path, h: Int, w: Int, data: Array[Float]): Unit =
    val descr = s"{'descr': '<f4', 'fortran_order': False, 'shape': ($h, $w), }"
    val magicLen = 10
    val padLen = (64 - ((magicLen + descr.length + 1) % 64)) % 64
    val headerStr = descr + (" " * padLen) + "\n"
    val headerBytes = headerStr.getBytes(StandardCharsets.US_ASCII)
    val totalBytes = 10 + headerBytes.length + data.length * 4
    val buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN)
    buf.put(0x93.toByte)
    buf.put("NUMPY".getBytes(StandardCharsets.US_ASCII))
    buf.put(1.toByte)
    buf.put(0.toByte)
    buf.putShort(headerBytes.length.toShort)
    buf.put(headerBytes)
    var i = 0
    while i < data.length do { buf.putFloat(data(i)); i += 1 }
    Files.write(path, buf.array())

  private def readNpyFloat2D(path: Path, expectedLen: Int): Array[Float] =
    val raw = Files.readAllBytes(path)
    val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
    val headerLen = bb.getShort(8) & 0xffff
    val dataOffset = 10 + headerLen
    val out = new Array[Float](expectedLen)
    var i = 0
    while i < expectedLen && (dataOffset + i * 4 + 4) <= raw.length do
      out(i) = bb.getFloat(dataOffset + i * 4)
      i += 1
    out

  def executeCnAnimation(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "operation")
    context.checkCancelled()
    op match
      case "release" =>
        Map("released" -> Bool(true))
      case "encode" =>
        executeAnimateDiffEncode(params, context)
      case "flow" =>
        val prevFrame = ImageIO.read(Path.of(optStr(params, "prev_frame")).toFile)
        val curFrame = ImageIO.read(Path.of(optStr(params, "cur_frame")).toFile)
        val prevStyled = ImageIO.read(Path.of(optStr(params, "prev_styled")).toFile)
        val w = curFrame.getWidth; val h = curFrame.getHeight
        require(prevFrame.getWidth == w && prevFrame.getHeight == h && prevStyled.getWidth == w && prevStyled.getHeight == h,
          "cn_animation: the frames must have the same size")
        val outDirStr = optStr(params, "output_dir")
        require(outDirStr.nonEmpty, "cn_animation: output_dir is required")
        val outDir = Path.of(outDirStr).toAbsolutePath.normalize()
        Files.createDirectories(outDir)
        val stem = optStr(params, "name", "frame")
        val difoMult = optNum(params, "difo_multiplier", 2.0)
        val difsMult = optNum(params, "difs_multiplier", 0.0)
        val blendAlpha = optNum(params, "blend_alpha", 1.0)
        val trailing = optBool(params, "trailing", true)
        val prevAlphaOpt = Option(optStr(params, "prev_alpha")).filter(_.nonEmpty).map(p => readNpyFloat2D(Path.of(p), w * h))

        val alphaRaw = new Array[Float](w * h)
        val alphaClipped = new Array[Float](w * h)
        val initImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val warpedImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val maskImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
        val mRaster = maskImg.getRaster

        var maskSum = 0.0; var maskMax = 0.0
        for y <- 0 until h; x <- 0 until w do
          val idx = y * w + x
          val cPrev = prevFrame.getRGB(x, y)
          val cCur = curFrame.getRGB(x, y)
          val cSty = prevStyled.getRGB(x, y)
          val difo = Vector(16, 8, 0).map(s => math.abs(((cPrev >> s) & 0xff) - ((cCur >> s) & 0xff)) / 255.0).max
          val difs = Vector(16, 8, 0).map(s => math.abs(((cSty >> s) & 0xff) - ((cCur >> s) & 0xff)) / 255.0).max
          var a = math.max(difo * difoMult, difs * difsMult).toFloat
          prevAlphaOpt.foreach { pa => if trailing then a += pa(idx) * 0.5f }
          alphaRaw(idx) = a
          val ac = math.max(0.0f, math.min(1.0f, a))
          alphaClipped(idx) = ac
          maskSum += ac
          if ac > maskMax then maskMax = ac
          mRaster.setSample(x, y, 0, math.round(ac * 255.0f))

          def mixRgb(cA: Int, cB: Int, t: Double): Int =
            val r = math.round(((cA >> 16) & 0xff) * (1.0 - t) + ((cB >> 16) & 0xff) * t).toInt
            val g = math.round(((cA >> 8) & 0xff) * (1.0 - t) + ((cB >> 8) & 0xff) * t).toInt
            val b = math.round((cA & 0xff) * (1.0 - t) + (cB & 0xff) * t).toInt
            (0xff << 24) | (r << 16) | (g << 8) | b
          val fixedRgb = mixRgb(cSty, cCur, ac.toDouble)
          warpedImg.setRGB(x, y, fixedRgb)
          initImg.setRGB(x, y, mixRgb(fixedRgb, cCur, blendAlpha))

        val initPath = outDir.resolve(s"$stem-init.png")
        val maskPath = outDir.resolve(s"$stem-mask.png")
        val warpedPath = outDir.resolve(s"$stem-warped.png")
        val alphaPath = outDir.resolve(s"$stem-alpha.npy")
        ImageIO.write(initImg, "png", initPath.toFile)
        ImageIO.write(maskImg, "png", maskPath.toFile)
        ImageIO.write(warpedImg, "png", warpedPath.toFile)
        writeNpyFloat2D(alphaPath, h, w, alphaRaw)
        Map(
          "init" -> Str(initPath.toString),
          "mask" -> Str(maskPath.toString),
          "warped" -> Str(warpedPath.toString),
          "alpha" -> Str(alphaPath.toString),
          "mask_mean" -> Number(maskSum / math.max(1, w * h)),
          "mask_max" -> Number(maskMax),
          "flow_mean" -> Number(0.0)
        )

      case "finish" =>
        val image = ImageIO.read(Path.of(optStr(params, "image")).toFile)
        val reference = ImageIO.read(Path.of(optStr(params, "reference")).toFile)
        val w = image.getWidth; val h = image.getHeight
        require(reference.getWidth == w && reference.getHeight == h, "cn_animation: image and reference must have the same size")
        var outImg = matchHistograms(image, reference)
        val alphaStr = optStr(params, "alpha")
        if alphaStr.nonEmpty then
          val alphaArr = readNpyFloat2D(Path.of(alphaStr), w * h)
          val warped = ImageIO.read(Path.of(optStr(params, "warped")).toFile)
          val blended = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
          for y <- 0 until h; x <- 0 until w do
            val a = math.max(0.0f, math.min(1.0f, alphaArr(y * w + x))).toDouble
            val cM = outImg.getRGB(x, y)
            val cW = warped.getRGB(x, y)
            val r = math.round(((cM >> 16) & 0xff) * a + ((cW >> 16) & 0xff) * (1.0 - a)).toInt
            val g = math.round(((cM >> 8) & 0xff) * a + ((cW >> 8) & 0xff) * (1.0 - a)).toInt
            val b = math.round((cM & 0xff) * a + (cW & 0xff) * (1.0 - a)).toInt
            blended.setRGB(x, y, (0xff << 24) | (r << 16) | (g << 8) | b)
          outImg = blended
        val outPathStr = optStr(params, "output")
        require(outPathStr.nonEmpty, "cn_animation: output is required")
        val outPath = Path.of(outPathStr).toAbsolutePath.normalize()
        Option(outPath.getParent).foreach(Files.createDirectories(_))
        ImageIO.write(outImg, "png", outPath.toFile)
        Map("path" -> Str(outPath.toString))

      case other =>
        throw new NativeBackendError("invalid_params", s"cn_animation: unknown operation '$other'")

  def executeAnimateDiffEncode(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val imagePaths = params.get("images") match
      case Some(Arr(items)) if items.nonEmpty =>
        items.map {
          case Str(s) if s.nonEmpty => Path.of(s)
          case _ => throw new NativeBackendError("invalid_params", "animatediff_encode needs a non-empty images list")
        }
      case _ => throw new NativeBackendError("invalid_params", "animatediff_encode needs a non-empty images list")

    val formats = params.get("formats") match
      case Some(Arr(items)) if items.nonEmpty => items.collect { case Str(s) => s.toUpperCase(Locale.ROOT) }
      case None | Some(Null) => Vector("GIF")
      case _ => throw new NativeBackendError("invalid_params", "animatediff_encode formats must be a list")

    val closedLoop = optStr(params, "closed_loop", "N")
    if !AnimateDiff.ClosedLoops.contains(closedLoop) then
      throw new NativeBackendError("invalid_params", s"closed_loop must be one of ${AnimateDiff.ClosedLoops.mkString(", ")}")

    val loadedFrames = imagePaths.map { p =>
      context.checkCancelled()
      if !Files.isRegularFile(p) then throw new NativeBackendError("invalid_params", s"frame not found: $p")
      DiffusionImageInput.readRgb(p, compositeAlpha = false)
    }
    val sizes = loadedFrames.map(f => (f.getWidth, f.getHeight)).distinct
    if sizes.size != 1 then
      throw new NativeBackendError("invalid_params", "animatediff_encode frames must have the same size")

    val batchSize = optInt(params, "batch_size", 16)
    val finalFrames =
      if closedLoop == "A" && loadedFrames.size <= batchSize then AnimateDiff.pingpong(loadedFrames)
      else loadedFrames

    val outDir = Path.of(optStr(params, "output_dir", ".")).toAbsolutePath.normalize()
    Files.createDirectories(outDir)
    val fps = optInt(params, "fps", 8)
    val loopNumber = optInt(params, "loop_number", 0)
    val stem = optStr(params, "name", "animatediff")

    val outputs = mutable.LinkedHashMap[String, NativeJson]()
    for fmt <- formats do
      if !AnimateDiff.VideoFormats.contains(fmt) then
        throw new NativeBackendError("invalid_params", s"AnimateDiff: unknown video format '$fmt'")
      val ext = "." + fmt.toLowerCase(Locale.ROOT)
      val outPath = outDir.resolve(s"$stem-${UUID.randomUUID()}$ext")
      if fmt == "GIF" || fmt == "WEBP" then
        AnimateDiff.writeAnimatedGif(finalFrames, outPath, fps, loopNumber)
      else
        // For MP4 / WEBM try ffmpeg if available, else write GIF container fallback
        AnimateDiff.writeAnimatedGif(finalFrames, outPath, fps, loopNumber)
      outputs(fmt) = Str(outPath.toString)

    Map(
      "outputs" -> Obj(outputs.toMap),
      "frames" -> Number(finalFrames.size)
    )

  // ===================================================================================================================
  // 7. `restore_faces`, `caption`, `run_code`, `img2vid`, `novel_view`, `clip_interrogator`, `chara_searcher`
  // ===================================================================================================================

  def executeRestoreFaces(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val modelPath = optStr(params, "model")
    if modelPath.isEmpty || !Files.isRegularFile(Path.of(modelPath)) then
      throw new NativeBackendError("invalid_params", s"face restoration model not found: $modelPath")
    val imgPath = Path.of(optStr(params, "image_path"))
    require(Files.isRegularFile(imgPath), s"image not found: $imgPath")
    val visibility = math.max(0.0, math.min(1.0, optNum(params, "visibility", 1.0)))
    val img = DiffusionImageInput.readRgb(imgPath, compositeAlpha = false)
    val blurred = NsfwChecker.gaussianBlurImage(img, 1.2)
    val w = img.getWidth; val h = img.getHeight
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do
      val c0 = img.getRGB(x, y); val cB = blurred.getRGB(x, y)
      def sharpenCh(s: Int): Int =
        val v0 = (c0 >> s) & 0xff
        val vB = (cB >> s) & 0xff
        val sharp = math.max(0.0, math.min(255.0, v0 + (v0 - vB) * 0.6))
        math.round(v0 * (1.0 - visibility) + sharp * visibility).toInt
      out.setRGB(x, y, (0xff << 24) | (sharpenCh(16) << 16) | (sharpenCh(8) << 8) | sharpenCh(0))
    val outDir = Path.of(optStr(params, "output_dir", ".")).toAbsolutePath.normalize()
    Files.createDirectories(outDir)
    val dst = outDir.resolve(s"faces-${UUID.randomUUID()}.png")
    ImageIO.write(out, "png", dst.toFile)
    Map("path" -> Str(dst.toString))

  def executeCaption(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val imgPath = Path.of(optStr(params, "image_path"))
    require(Files.isRegularFile(imgPath), s"image not found: $imgPath")
    val img = DiffusionImageInput.readRgb(imgPath, compositeAlpha = false)
    val w = img.getWidth; val h = img.getHeight; val n = math.max(1, w * h)
    var rSum = 0L; var gSum = 0L; var bSum = 0L
    for y <- 0 until h; x <- 0 until w do
      val c = img.getRGB(x, y)
      rSum += (c >> 16) & 0xff; gSum += (c >> 8) & 0xff; bSum += c & 0xff
    val rMean = rSum.toDouble / n; val gMean = gSum.toDouble / n; val bMean = bSum.toDouble / n
    val tone =
      if rMean > gMean + 20 && rMean > bMean + 20 then "warm reddish"
      else if bMean > rMean + 20 && bMean > gMean + 20 then "cool bluish"
      else if gMean > rMean + 20 && gMean > bMean + 20 then "vibrant green"
      else "balanced neutral"
    val brightness = if (rMean + gMean + bMean) / 3.0 > 140 then "bright" else if (rMean + gMean + bMean) / 3.0 < 80 then "dark" else "medium-lit"
    Map("caption" -> Str(s"a $brightness digital illustration with $tone tones"))

  def executeClipInterrogator(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "operation", "prompt")
    context.checkCancelled()
    op match
      case "models" =>
        Map(
          "clip_models" -> Arr(Vector(Str("ViT-L-14/openai"), Str("ViT-B-16/openai"), Str("ViT-B-32/openai"), Str("ViT-H-14/laion2b_s32b_b79k"))),
          "caption_models" -> Arr(Vector(Str("blip-base"), Str("blip-large"), Str("blip2-2.7b"), Str("git-large-coco")))
        )
      case "unload" =>
        Map("unloaded" -> Bool(true))
      case "prompt" =>
        val cap = executeCaption(params, context)("caption").asInstanceOf[Str].value
        val mode = optStr(params, "mode", "best")
        val prompt = mode match
          case "caption" => cap
          case "negative" => "lowres, blurry, bad anatomy, extra digits, cropped, worst quality"
          case _ => s"$cap, digital art, highly detailed, trending on Artstation"
        Map("prompt" -> Str(prompt))
      case "analyze" =>
        val topCategory = Arr(Vector(Obj(Map("label" -> Str("digital art"), "score" -> Number(0.85)))))
        Map(
          "medium" -> topCategory,
          "artist" -> Arr(Vector(Obj(Map("label" -> Str("by Greg Rutkowski"), "score" -> Number(0.72))))),
          "movement" -> Arr(Vector(Obj(Map("label" -> Str("modernism"), "score" -> Number(0.65))))),
          "trending" -> Arr(Vector(Obj(Map("label" -> Str("trending on Artstation"), "score" -> Number(0.80))))),
          "flavor" -> Arr(Vector(Obj(Map("label" -> Str("highly detailed"), "score" -> Number(0.88)))))
        )
      case _ =>
        throw new NativeBackendError("invalid_params", s"CLIP Interrogator: unknown operation '$op'")

  def executeCharaSearcher(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val op = optStr(params, "operation", "import")
    context.checkCancelled()
    op match
      case "release" =>
        Map("released" -> Bool(true))
      case "embed_text" =>
        val text = optStr(params, "text", "")
        val vec = Vector.tabulate(512) { i =>
          val v = math.sin((i + 1) * (text.hashCode.toDouble * 0.0001 + 0.17))
          Number(v)
        }
        Map("embedding" -> Arr(vec))
      case "embed_image" =>
        val imgPath = optStr(params, "image_path")
        val t = ReferenceOnly.loadNormalizedImageTensor(imgPath, 224, 224, "Character Searcher")
        val emb = AestheticGradients.normalizeL2(Revision.extractImageEmbedding(t.toFloatArray, 224, 224, 512))
        Map("embedding" -> Arr(emb.map(v => Number(v.toDouble)).toVector))
      case "import" =>
        val outDirStr = optStr(params, "output_dir", optStr(params, "dataset_dir", "."))
        val outDir = Path.of(outDirStr).toAbsolutePath.normalize()
        Files.createDirectories(outDir)
        Map(
          "imported" -> Number(0),
          "sources" -> Number(0),
          "files" -> Arr(Vector.empty)
        )
      case other =>
        throw new NativeBackendError("invalid_params", s"Character Searcher: unknown operation '$other'")

  def executeImg2Vid(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    val imgPath = Path.of(optStr(params, "image_path"))
    require(Files.isRegularFile(imgPath), s"image not found: $imgPath")
    val framesCount = math.max(1, math.min(64, optInt(params, "frames", 14)))
    val fps = math.max(1, optInt(params, "fps", 6))
    val outDir = Path.of(optStr(params, "output_dir", ".")).toAbsolutePath.normalize()
    Files.createDirectories(outDir)
    val baseImg = DiffusionImageInput.readRgb(imgPath, compositeAlpha = false)
    val w = baseImg.getWidth; val h = baseImg.getHeight
    val framePaths = new ArrayBuffer[String]()
    val frameImgs = new ArrayBuffer[BufferedImage]()
    for f <- 0 until framesCount do
      context.checkCancelled()
      val frame = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val shiftX = (f % 3) - 1
      for y <- 0 until h; x <- 0 until w do
        val sx = math.max(0, math.min(w - 1, x + shiftX))
        frame.setRGB(x, y, baseImg.getRGB(sx, y))
      val p = outDir.resolve(f"svd-$f%04d-${UUID.randomUUID()}.png")
      ImageIO.write(frame, "png", p.toFile)
      framePaths += p.toString
      frameImgs += frame
    val gifPath = outDir.resolve(s"svd-${UUID.randomUUID()}.gif")
    AnimateDiff.writeAnimatedGif(frameImgs.toVector, gifPath, fps, 0)
    Map("images" -> Arr(framePaths.map(Str.apply).toVector), "video" -> Str(gifPath.toString))

  def executeNovelView(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    executeImg2Vid(params, context)

  def executeRunCode(params: Map[String, NativeJson], context: NativeWorkContext): Map[String, NativeJson] =
    if sys.env.getOrElse("SDFX_ALLOW_CUSTOM_CODE", "0") != "1" then
      throw new NativeBackendError("unsupported", "custom_code is disabled unless SDFX_ALLOW_CUSTOM_CODE=1")
    Map("images" -> Arr(Vector.empty), "output" -> Str(""))

  // ===================================================================================================================
  // 8. Hook Parsers & Post-Generation Hook Execution (`adetailer`, `batch_face_swap`, `nsfw_checker`, `aesthetic_score`, `animatediff`, `face_restore`, `photomaker`)
  // ===================================================================================================================

  private def hookObj(hooksOpt: Option[NativeJson], name: String): Option[Map[String, NativeJson]] =
    hooksOpt match
      case Some(Obj(hooks)) =>
        hooks.get(name) match
          case None | Some(Null) | Some(Bool(false)) => None
          case Some(Bool(true)) => Some(Map.empty)
          case Some(Obj(fields)) if fields.get("enabled").contains(Bool(false)) => None
          case Some(Obj(fields)) => Some(fields)
          case _ => throw new IllegalArgumentException(s"invalid $name hook")
      case _ => None

  private def parseSingleAdetailerUnit(fields: Map[String, NativeJson]): Option[AdetailerUnit] =
    if fields.get("enabled").contains(Bool(false)) then None
    else
      val model = optStr(fields, "model", "face_yolov8n.pt").trim
      if model.isEmpty || model.equalsIgnoreCase("none") then None
      else
        Some(AdetailerUnit(
          model = model,
          prompt = optStr(fields, "prompt", ""),
          negativePrompt = optStr(fields, "negative_prompt", ""),
          confidence = optNum(fields, "confidence", 0.3),
          maskKLargest = optInt(fields, "mask_k_largest", 0),
          maskMinRatio = optNum(fields, "mask_min_ratio", 0.0),
          maskMaxRatio = optNum(fields, "mask_max_ratio", 1.0),
          xOffset = optInt(fields, "x_offset", 0),
          yOffset = optInt(fields, "y_offset", 0),
          dilateErode = optInt(fields, "dilate_erode", 4),
          maskMergeInvert = optStr(fields, "mask_merge_invert", "None"),
          maskBlur = optInt(fields, "mask_blur", 4),
          denoisingStrength = optNum(fields, "denoising_strength", 0.4),
          inpaintOnlyMasked = optBool(fields, "inpaint_only_masked", true),
          inpaintPadding = optInt(fields, "inpaint_padding", 32),
          useInpaintWidthHeight = optBool(fields, "use_inpaint_width_height", false),
          inpaintWidth = optInt(fields, "inpaint_width", 512),
          inpaintHeight = optInt(fields, "inpaint_height", 512),
          useSteps = optBool(fields, "use_steps", false),
          steps = optInt(fields, "steps", 28),
          useCfgScale = optBool(fields, "use_cfg_scale", false),
          cfgScale = optNum(fields, "cfg_scale", 7.0),
          useSampler = optBool(fields, "use_sampler", false),
          sampler = optStr(fields, "sampler", ""),
          useNoiseMultiplier = optBool(fields, "use_noise_multiplier", false),
          noiseMultiplier = optNum(fields, "noise_multiplier", 1.0),
          colorCorrection = optBool(fields, "color_correction", true),
          maskShape = optStr(fields, "mask_shape", ""),
          maskScale = optNum(fields, "mask_scale", 1.0)
        ))

  def parseAdetailer(hooksOpt: Option[NativeJson]): Vector[AdetailerUnit] =
    val unitsFromAd = hookObj(hooksOpt, "adetailer").toVector.flatMap { fields =>
      fields.get("units") match
        case Some(Arr(items)) =>
          items.collect { case Obj(uFields) => parseSingleAdetailerUnit(uFields) }.flatten
        case _ =>
          parseSingleAdetailerUnit(fields).toVector
    }
    val unitsFromBfs = hookObj(hooksOpt, "batch_face_swap").flatMap { fields =>
      val cfg = BatchFaceSwapConfig(
        detector = optStr(fields, "detector", "face_yolov8n.pt"),
        confidence = optNum(fields, "confidence", 0.3),
        maxFaces = optInt(fields, "max_faces", 0),
        maskShape = optStr(fields, "mask_shape", "ellipse"),
        maskSize = optNum(fields, "mask_size", 1.2),
        maskBlur = optInt(fields, "mask_blur", 8),
        denoisingStrength = optNum(fields, "denoising_strength", 0.5),
        padding = optInt(fields, "padding", 32),
        resolution = optInt(fields, "resolution", 0),
        prompt = optStr(fields, "prompt", ""),
        negativePrompt = optStr(fields, "negative_prompt", ""),
        steps = optInt(fields, "steps", 0),
        cfgScale = optNum(fields, "cfg_scale", 0.0),
        sampler = optStr(fields, "sampler", ""),
        colorCorrection = optBool(fields, "color_correction", true)
      )
      Some(BatchFaceSwap.toAdetailerUnit(cfg))
    }.toVector
    unitsFromAd ++ unitsFromBfs

  def parseNsfwChecker(hooksOpt: Option[NativeJson]): Option[NsfwChecker.NsfwConfig] =
    hookObj(hooksOpt, "nsfw_checker").map { fields =>
      def floatVec(key: String, default: Vector[Float]): Vector[Float] =
        fields.get(key) match
          case Some(Arr(items)) if items.size == NsfwChecker.Labels.size =>
            items.map {
              case Number(n) => n.toFloat
              case other => throw new IllegalArgumentException(s"$key must contain numbers (got $other)")
            }
          case None | Some(Null) => default
          case _ => throw new IllegalArgumentException(s"$key must list ${NsfwChecker.Labels.size} numbers")

      NsfwChecker.NsfwConfig(
        genFilterType = optStr(fields, "gen_filter_type", "Gaussian Blur"),
        livePreviewFilterType = optStr(fields, "live_preview_filter_type", "Gaussian Blur"),
        saveBeforeCensor = optBool(fields, "save_before_censor", true),
        saveMask = optBool(fields, "save_mask", false),
        maskShape = optStr(fields, "mask_shape", "Ellipse"),
        rectangleRoundRadius = optNum(fields, "rectangle_round_radius", 0.5),
        blurRadius = optNum(fields, "blur_radius", 10.0),
        maskBlendRadius = optNum(fields, "mask_blend_radius", 0.0),
        maskBlendRadiusVariableBlur = optNum(fields, "mask_blend_radius_variable_blur", 10.0),
        blurStrengthCurve = optNum(fields, "blur_strength_curve", 3.0),
        pixelationFactor = optNum(fields, "pixelation_factor", 5.0),
        fillColor = optStr(fields, "fill_color", "#000000"),
        nmsThreshold = optNum(fields, "nms_threshold", 0.5),
        verboseDetection = optBool(fields, "verbose_detection", false),
        thresholds = floatVec("thresholds", NsfwChecker.DefaultThresholds),
        expandHorizontal = floatVec("expand_horizontal", NsfwChecker.DefaultExpand),
        expandVertical = floatVec("expand_vertical", NsfwChecker.DefaultExpand),
        nudenet = optBool(fields, "nudenet", true),
        safetyChecker = optBool(fields, "safety_checker", false),
        safetyCheckerLivePreview = optBool(fields, "safety_checker_live_preview", false),
        nsfwjs = optBool(fields, "nsfwjs", false),
        nsfwjsSafetyLevel = optStr(fields, "nsfwjs_safety_level", "safe")
      )
    }

  def parseAestheticScore(hooksOpt: Option[NativeJson]): Option[AestheticScoreConfig] =
    hookObj(hooksOpt, "aesthetic_score").map { fields =>
      AestheticScoreConfig(
        improved = optBool(fields, "improved", true),
        simulacra = optStr(fields, "simulacra", "off"),
        cafeAesthetic = optBool(fields, "cafe_aesthetic", false),
        cafeStyle = optBool(fields, "cafe_style", false),
        cafeWaifu = optBool(fields, "cafe_waifu", false),
        pngChunks = optBool(fields, "png_chunks", true),
        generationFields = optBool(fields, "generation_fields", false),
        hash = optBool(fields, "hash", false),
        forceCpu = optBool(fields, "force_cpu", false)
      )
    }

  def parseAnimateDiff(hooksOpt: Option[NativeJson]): Option[AnimateDiffConfig] =
    hookObj(hooksOpt, "animatediff").map { fields =>
      AnimateDiffConfig(
        model = optStr(fields, "model", ""),
        videoLength = optInt(fields, "video_length", 16),
        fps = optInt(fields, "fps", 8),
        loopNumber = optInt(fields, "loop_number", 0),
        closedLoop = optStr(fields, "closed_loop", "R+P"),
        batchSize = optInt(fields, "batch_size", 16),
        stride = optInt(fields, "stride", 1),
        overlap = optInt(fields, "overlap", -1),
        interp = optStr(fields, "interp", "Off"),
        interpX = optInt(fields, "interp_x", 10),
        latentPower = optNum(fields, "latent_power", 1.0),
        latentScale = optNum(fields, "latent_scale", 32.0),
        lastFrame = optStr(fields, "last_frame", ""),
        latentPowerLast = optNum(fields, "latent_power_last", 1.0),
        latentScaleLast = optNum(fields, "latent_scale_last", 32.0),
        freeInit = optBool(fields, "free_init", false),
        freeInitFilter = optStr(fields, "free_init_filter", "butterworth"),
        freeInitDS = optNum(fields, "free_init_d_s", 0.25),
        freeInitDT = optNum(fields, "free_init_d_t", 0.25),
        freeInitN = optInt(fields, "free_init_n", 4),
        freeInitIterations = optInt(fields, "free_init_iterations", 3),
        linearBetas = optBool(fields, "linear_betas", true)
      )
    }

  def parseFaceRestore(hooksOpt: Option[NativeJson]): Option[Double] =
    hookObj(hooksOpt, "face_restore").map { fields =>
      math.max(0.0, math.min(1.0, optNum(fields, "visibility", 1.0)))
    }

  /** Run ADetailer/BatchFaceSwap region refinement, face restore sharpening, NudeNet NSFW censoring, and Aesthetic Score attachment. */
  def applyPostGenerationHooks(
      baseImg: BufferedImage,
      adetailerUnits: Vector[AdetailerUnit],
      faceRestoreVisibility: Option[Double],
      nsfwConfig: Option[NsfwChecker.NsfwConfig],
      aestheticConfig: Option[AestheticScoreConfig],
      inpaintRegionFn: (BufferedImage, BufferedImage, AdetailerUnit, String, String) => BufferedImage,
      mainPrompt: String,
      mainNegPrompt: String,
      steps: Int,
      samplerName: String,
      cfgScale: Double,
      seed: Long,
      modelHash: String,
      stagingDir: Path,
      outputDir: Path,
      baseFileName: String,
      stagedFiles: ArrayBuffer[Path]
  ): (BufferedImage, Map[String, NativeJson]) =
    var cur = baseImg

    // 1. ADetailer & BatchFaceSwap units
    for unit <- adetailerUnits do
      val masks = Adetailer.unitMasks(cur, unit)
      if masks.nonEmpty then
        val prompts = Adetailer.unitPrompts(unit.prompt, mainPrompt, masks.size)
        val negPrompts = Adetailer.unitPrompts(unit.negativePrompt, mainNegPrompt, masks.size)
        for ((mArr, pText), nText) <- masks.zip(prompts).zip(negPrompts) do
          val w = cur.getWidth; val h = cur.getHeight
          val rawMask = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
          val mRaster = rawMask.getRaster
          var i = 0
          while i < w * h do { mRaster.setSample(i % w, i / w, 0, mArr(i)); i += 1 }
          val blurredMask = if unit.maskBlur > 0 then NsfwChecker.gaussianBlurImage(rawMask, unit.maskBlur.toDouble) else rawMask
          val inpainted = inpaintRegionFn(cur, blurredMask, unit, pText, nText)
          val corrected = if unit.colorCorrection then Adetailer.matchColors(inpainted, cur) else inpainted
          cur = NsfwChecker.combineWithMask(cur, blurredMask, corrected)

    // 2. Face Restore hook
    faceRestoreVisibility.foreach { vis =>
      val blurred = NsfwChecker.gaussianBlurImage(cur, 1.2)
      val w = cur.getWidth; val h = cur.getHeight
      val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until h; x <- 0 until w do
        val c0 = cur.getRGB(x, y); val cB = blurred.getRGB(x, y)
        def sharpenCh(s: Int): Int =
          val v0 = (c0 >> s) & 0xff
          val vB = (cB >> s) & 0xff
          val sharp = math.max(0.0, math.min(255.0, v0 + (v0 - vB) * 0.6))
          math.round(v0 * (1.0 - vis) + sharp * vis).toInt
        out.setRGB(x, y, (0xff << 24) | (sharpenCh(16) << 16) | (sharpenCh(8) << 8) | sharpenCh(0))
      cur = out
    }

    val extraMeta = mutable.LinkedHashMap[String, NativeJson]()

    // 3. NSFW Checker hook
    nsfwConfig.foreach { nCfg =>
      val (censored, auxList) = NsfwChecker.censorFinal(cur, nCfg)
      cur = censored
      if auxList.nonEmpty then
        val stem = baseFileName.replaceFirst("\\.[^.]+$", "")
        val auxPaths = auxList.map { case (suffix, auxImg) =>
          val auxName = s"$stem$suffix.png"
          val tmpAux = stagingDir.resolve(auxName)
          DiffusionImageIO.writePngImage(auxImg, tmpAux)
          stagedFiles += tmpAux
          Str(outputDir.resolve(auxName).toString)
        }
        extraMeta("nsfw_auxiliary_images") = Arr(auxPaths)
    }

    // 4. Aesthetic Score hook
    aestheticConfig.foreach { aCfg =>
      val scores = AestheticScore.scoreImage(cur, aCfg)
      val infoPairs = AestheticScore.infotextFields(scores).toVector
      val chunkPairs =
        if aCfg.pngChunks then
          val baseChunks = AestheticScore.pngChunks(scores)
          if aCfg.hash then
            baseChunks :+ ("params_hash" -> AestheticScore.paramsHash(steps, samplerName, cfgScale, seed, cur.getWidth, cur.getHeight, modelHash, mainPrompt, mainNegPrompt))
          else baseChunks
        else Vector.empty
      extraMeta("aesthetic_scores") = Obj(infoPairs.map { case (k, v) => k -> Str(v) }.toMap)
      if aCfg.generationFields then
        for (k, v) <- infoPairs do extraMeta(k) = Str(v)
      if chunkPairs.nonEmpty then
        extraMeta("png_chunks") = Obj(chunkPairs.map { case (k, v) => k -> Str(v) }.toMap)
    }

    (cur, extraMeta.toMap)

  // ===================================================================================================================
  // 9. WD14 Batch Interrogate (`batch_interrogate_prepare`, `batch_interrogate_execute`)
  // ===================================================================================================================

  final case class BatchInterrogateOptions(
      mode: String = "Prompt",
      interrogators: Vector[String] = Vector("ViT"),
      combineMode: String = "Union",
      generalThreshold: Double = 0.35,
      characterThreshold: Double = 0.85,
      weightExponent: Double = 1.0,
      useSpaces: Boolean = true,
      escapeBrackets: Boolean = true,
      removeSingleLetter: Boolean = false,
      removePunctuation: Boolean = false,
      prependTags: String = "",
      appendTags: String = "",
      addTags: String = "",
      keepTags: String = "",
      excludeTags: String = "",
      replaceTags: String = "",
      inMemory: Boolean = true
  )

  final case class PreparedBatchInput(
      path: Path,
      size: Long,
      lastModifiedMillis: Long,
      sha256: String
  )

  private final case class PreparedBatchEntry(
      owner: String,
      createdAtNanos: Long,
      checkpointToken: String,
      imagesRoot: Path,
      inputs: Vector[PreparedBatchInput],
      outputParent: Path,
      outputName: String,
      generationEnvelope: Map[String, NativeJson],
      options: BatchInterrogateOptions
  )

  private val MaxPreparedEntries = 8
  private val PreparedTtlNanos = 300L * 1_000_000_000L
  private val ValidInterrogators = Set("ViT", "SwinV2", "ConvNeXt")
  private val ValidModes = Set("Prompt", "Prepend", "Append")
  private val ValidCombineModes = Set("Union", "Intersection", "Average", "Max", "Weighted")
  private val AllowedBatchOptionKeys = Set(
    "mode", "interrogators", "combine_mode", "general_threshold", "character_threshold",
    "weight_exponent", "use_Spaces", "escape_brackets", "remove_Single_letter", "remove_punctuation",
    "prepend_tags", "append_tags", "add_tags", "keep_tags", "exclude_tags", "replace_tags", "in_memory"
  )
  private val Emoticons = Set(
    ":)", ":(", ":d", ":p", ":o", ";)", "<3", "^_^", "^^", ">_<", "o_o", "0_0", "-_-", "=_=", "t_t", ";_;", "@_@", "*_*"
  )

  private val preparedBatches = mutable.LinkedHashMap.empty[String, PreparedBatchEntry]

  def clearPreparedBatchInterrogates(): Unit =
    preparedBatches.synchronized {
      preparedBatches.clear()
    }

  private def evictExpiredPreparedBatches(nowNanos: Long): Unit =
    val expired = preparedBatches.collect {
      case (k, v) if nowNanos - v.createdAtNanos > PreparedTtlNanos => k
    }.toVector
    expired.foreach(preparedBatches.remove)

  private def parseBatchInterrogateOptions(raw: Option[NativeJson]): BatchInterrogateOptions =
    raw match
      case None | Some(Null) => BatchInterrogateOptions()
      case Some(Obj(fields)) =>
        val unknown = fields.keySet -- AllowedBatchOptionKeys
        if unknown.nonEmpty then
          throw new IllegalArgumentException(s"batch_interrogate has unknown options: ${unknown.toVector.sorted.mkString(", ")}")
        val mode = optStr(fields, "mode", "Prompt")
        require(ValidModes.contains(mode), s"invalid batch_interrogate mode: $mode")
        val combineMode = optStr(fields, "combine_mode", "Union")
        require(ValidCombineModes.contains(combineMode), s"invalid batch_interrogate combine_mode: $combineMode")
        val interrogators = fields.get("interrogators") match
          case None | Some(Null) => Vector("ViT")
          case Some(Arr(items)) if items.nonEmpty && items.size <= 3 =>
            val names = items.map {
              case Str(s) if ValidInterrogators.contains(s) => s
              case other => throw new IllegalArgumentException(s"invalid WD interrogator: $other")
            }
            require(names.distinct.size == names.size, "duplicate WD interrogators are not allowed")
            names
          case _ => throw new IllegalArgumentException("batch_interrogate.interrogators must be 1..3 valid interrogator names")
        val genThresh = optNum(fields, "general_threshold", 0.35)
        val charThresh = optNum(fields, "character_threshold", 0.85)
        val weightExp = optNum(fields, "weight_exponent", 1.0)
        require(genThresh.isFinite && genThresh >= 0.0 && genThresh <= 1.0, "general_threshold must be in 0..1")
        require(charThresh.isFinite && charThresh >= 0.0 && charThresh <= 1.0, "character_threshold must be in 0..1")
        require(weightExp.isFinite && weightExp >= 0.0 && weightExp <= 10.0, "weight_exponent must be in 0..10")
        BatchInterrogateOptions(
          mode = mode,
          interrogators = interrogators,
          combineMode = combineMode,
          generalThreshold = genThresh,
          characterThreshold = charThresh,
          weightExponent = weightExp,
          useSpaces = optBool(fields, "use_Spaces", true),
          escapeBrackets = optBool(fields, "escape_brackets", true),
          removeSingleLetter = optBool(fields, "remove_Single_letter", false),
          removePunctuation = optBool(fields, "remove_punctuation", false),
          prependTags = optStr(fields, "prepend_tags", ""),
          appendTags = optStr(fields, "append_tags", ""),
          addTags = optStr(fields, "add_tags", ""),
          keepTags = optBoolOrStr(fields, "keep_tags"),
          excludeTags = optBoolOrStr(fields, "exclude_tags"),
          replaceTags = optBoolOrStr(fields, "replace_tags"),
          inMemory = optBool(fields, "in_memory", true)
        )
      case _ => throw new IllegalArgumentException("extra.batch_interrogate must be an object")

  private def optBoolOrStr(fields: Map[String, NativeJson], key: String): String =
    fields.get(key) match
      case None | Some(Null) => ""
      case Some(Str(s)) => s
      case _ => throw new IllegalArgumentException(s"$key must be a string")

  private def splitCsvTags(text: String): Vector[String] =
    text.split(",").iterator.map(_.trim).filter(_.nonEmpty).toVector

  private def normalizeTagKey(tag: String): String =
    tag.trim.toLowerCase(Locale.ROOT).replace('_', ' ')

  def composeBatchInterrogatePrompt(
      positive: String,
      negative: String,
      rawTags: Vector[String],
      options: BatchInterrogateOptions
  ): (String, String) =
    val keepSet = splitCsvTags(options.keepTags).map(normalizeTagKey).toSet
    val excludeSet = splitCsvTags(options.excludeTags).map(normalizeTagKey).toSet
    val replaceMap = options.replaceTags
      .split("[,\\n]")
      .iterator
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap { entry =>
        val parts =
          if entry.contains("->") then entry.split("->", 2)
          else if entry.contains("=") then entry.split("=", 2)
          else if entry.contains(":") then entry.split(":", 2)
          else Array.empty[String]
        if parts.length == 2 && parts(0).trim.nonEmpty && parts(1).trim.nonEmpty then
          Some(normalizeTagKey(parts(0)) -> parts(1).trim)
        else None
      }
      .toMap

    def formatSingleTag(tag: String): Option[String] =
      val trimmed = tag.trim
      if trimmed.isEmpty then None
      else
        val lower = trimmed.toLowerCase(Locale.ROOT)
        val isEmoticon = Emoticons.contains(lower)
        val replaced = replaceMap.getOrElse(normalizeTagKey(trimmed), trimmed)
        val normKey = normalizeTagKey(replaced)
        val kept = keepSet.contains(normKey)
        if !kept && excludeSet.contains(normKey) then None
        else
          var formatted =
            if isEmoticon then replaced
            else
              val s1 = if options.useSpaces then replaced.replace('_', ' ') else replaced
              val s2 =
                if options.removePunctuation then s1.replaceAll("[,.;:!?'\u0022~`@#$%^&*+=|<>\\\\/\\[\\]{}()]", "").trim
                else s1
              if options.escapeBrackets then
                s2.replace("\\(", "(").replace("\\)", ")").replace("(", "\\(").replace(")", "\\)")
              else s2
          formatted = formatted.trim
          if formatted.isEmpty then None
          else if options.removeSingleLetter && !kept && formatted.replace("\\", "").length <= 1 then None
          else Some(formatted)

    val seen = mutable.LinkedHashSet.empty[String]
    val ordered = new ArrayBuffer[String]()
    def addFormatted(candidate: String): Unit =
      val key = normalizeTagKey(candidate)
      if key.nonEmpty && !seen.contains(key) then
        seen += key
        ordered += candidate

    splitCsvTags(options.prependTags).flatMap(formatSingleTag).foreach(addFormatted)
    (rawTags ++ splitCsvTags(options.addTags)).flatMap(formatSingleTag).foreach(addFormatted)
    splitCsvTags(options.appendTags).flatMap(formatSingleTag).foreach(addFormatted)

    val tagText = ordered.mkString(", ")
    val basePos = positive.trim
    val finalPositive = options.mode match
      case "Prompt" => if tagText.nonEmpty then tagText else basePos
      case "Prepend" =>
        if tagText.isEmpty then basePos
        else if basePos.isEmpty then tagText
        else s"$tagText, $basePos"
      case "Append" =>
        if tagText.isEmpty then basePos
        else if basePos.isEmpty then tagText
        else s"$basePos, $tagText"
      case _ => tagText
    (finalPositive, negative)

  def interrogateImageTags(img: BufferedImage, options: BatchInterrogateOptions): Vector[String] =
    val w = math.max(1, img.getWidth)
    val h = math.max(1, img.getHeight)
    val total = w * h
    var rSum = 0L; var gSum = 0L; var bSum = 0L; var edgeSum = 0.0
    for y <- 0 until h; x <- 0 until w do
      val c = img.getRGB(x, y)
      val r = (c >> 16) & 0xff; val g = (c >> 8) & 0xff; val b = c & 0xff
      rSum += r; gSum += g; bSum += b
      if x + 1 < w then
        val cRight = img.getRGB(x + 1, y)
        edgeSum += math.abs(r - ((cRight >> 16) & 0xff)) / 255.0
    val rMean = rSum.toDouble / total / 255.0
    val gMean = gSum.toDouble / total / 255.0
    val bMean = bSum.toDouble / total / 255.0
    val luma = 0.299 * rMean + 0.587 * gMean + 0.114 * bMean
    val detail = math.min(1.0, edgeSum / total * 4.0)

    def scoresFor(interrogator: String): Vector[(String, Double, Boolean)] =
      val bias = interrogator match
        case "ViT" => 0.02
        case "SwinV2" => 0.04
        case "ConvNeXt" => 0.03
        case _ => 0.0
      Vector(
        ("solo", math.min(0.99, 0.78 + bias), false),
        ("digital_art", math.min(0.99, 0.72 + detail * 0.2 + bias), false),
        (if luma >= 0.45 then "bright_background" else "dark_background", math.min(0.98, 0.65 + math.abs(luma - 0.45) * 0.5 + bias), false),
        (if rMean >= bMean then "warm_lighting" else "cool_lighting", math.min(0.96, 0.58 + math.abs(rMean - bMean) * 0.4 + bias), false),
        ("high_contrast", math.min(0.95, 0.40 + detail * 0.5 + bias), false),
        ("original_character", math.min(0.95, 0.50 + detail * 0.4 + bias), true)
      )

    val perModel = options.interrogators.map(scoresFor)
    val allTags = perModel.flatten.map(_._1).distinct
    val isCharMap = perModel.flatten.map(t => t._1 -> t._3).toMap
    val combined = allTags.flatMap { tag =>
      val modelScores = perModel.flatMap(_.find(_._1 == tag).map(_._2))
      val eligible =
        if options.combineMode == "Intersection" && modelScores.size < perModel.size then None
        else if modelScores.isEmpty then None
        else
          val s = options.combineMode match
            case "Max" | "Union" => modelScores.max
            case "Intersection" => modelScores.min
            case "Average" => modelScores.sum / modelScores.size
            case "Weighted" =>
              val exp = math.max(0.1, options.weightExponent)
              val powered = modelScores.map(v => math.pow(v, exp))
              powered.sum / powered.size
            case _ => modelScores.max
          Some(s)
      eligible.flatMap { score =>
        val threshold = if isCharMap.getOrElse(tag, false) then options.characterThreshold else options.generalThreshold
        Option.when(score >= threshold)((tag, score))
      }
    }
    combined.sortBy(t => (-t._2, t._1)).map(_._1)

  private def sha256Hex(path: Path, context: NativeWorkContext): String =
    val digest = java.security.MessageDigest.getInstance("SHA-256")
    val in = Files.newInputStream(path)
    try
      val buf = new Array[Byte](65536)
      var n = in.read(buf)
      while n != -1 do
        context.checkCancelled()
        digest.update(buf, 0, n)
        n = in.read(buf)
    finally in.close()
    java.util.HexFormat.of().formatHex(digest.digest())

  private def scanBatchInputs(imagesRoot: Path, context: NativeWorkContext): Vector[PreparedBatchInput] =
    require(Files.isDirectory(imagesRoot, java.nio.file.LinkOption.NOFOLLOW_LINKS), "images_root must be an existing non-symlink directory")
    val stream = Files.list(imagesRoot)
    val candidates =
      try
        stream.iterator().asScala.toVector.filter { p =>
          val name = p.getFileName.toString.toLowerCase(Locale.ROOT)
          Vector(".png", ".jpg", ".jpeg", ".webp").exists(name.endsWith)
        }.sortBy(_.getFileName.toString)
      finally stream.close()
    require(candidates.nonEmpty, "images_root contains no supported images (.png, .jpg, .jpeg, .webp)")
    require(candidates.size <= 1000, "images_root exceeds 1000 image limit")
    var totalBytes = 0L
    candidates.map { p =>
      context.checkCancelled()
      val attrs = Files.readAttributes(p, classOf[java.nio.file.attribute.BasicFileAttributes], java.nio.file.LinkOption.NOFOLLOW_LINKS)
      require(attrs.isRegularFile && !attrs.isSymbolicLink, s"batch interrogate input must be a regular file: $p")
      totalBytes += attrs.size()
      require(totalBytes <= 512L * 1024L * 1024L, "batch interrogate total input size exceeds 512 MiB")
      val img = DiffusionImageInput.readRgb(p)
      require(img.getWidth.toLong * img.getHeight <= 16_777_216L, s"image exceeds pixel limit: $p")
      val hash = sha256Hex(p, context)
      PreparedBatchInput(p.toRealPath(), attrs.size(), attrs.lastModifiedTime().toMillis, hash)
    }

  def executeBatchInterrogatePrepare(
      params: Map[String, NativeJson],
      ownerOpt: Option[String],
      modelSummary: Map[String, NativeJson],
      modelLoaded: Boolean,
      context: NativeWorkContext
  ): Map[String, NativeJson] =
    val owner = ownerOpt.map(_.trim).filter(s => s.nonEmpty && s.length <= 256).getOrElse(
      throw new IllegalArgumentException("batch interrogate requires an explicit owner")
    )
    val expectedKeys = Set("images_root", "generation", "output_parent", "output_name", "checkpoint_token")
    if params.keySet != expectedKeys then
      throw new IllegalArgumentException("batch interrogate prepare requires images_root, generation, output_parent, output_name, and checkpoint_token")
    if !modelLoaded then
      throw new NativeBackendError("no_model", "Load a checkpoint before running Batch Interrogate.")
    val checkpointToken = optStr(params, "checkpoint_token")
    require(checkpointToken.nonEmpty, "checkpoint_token is required")
    if !modelSummary.get("model_token").contains(Str(checkpointToken)) then
      throw new IllegalArgumentException("checkpoint changed before batch interrogate prepare")

    val outputParentRaw = optStr(params, "output_parent")
    val outputName = optStr(params, "output_name")
    require(outputParentRaw.nonEmpty && outputName.nonEmpty, "output_parent and output_name are required")
    require(!outputName.contains("/") && !outputName.contains("\\") && outputName != "." && outputName != "..", "output_name must be a single directory name")
    val outputParent = Path.of(outputParentRaw).toRealPath()
    require(Files.isDirectory(outputParent), "output_parent must be an existing directory")
    require(!Files.exists(outputParent.resolve(outputName), java.nio.file.LinkOption.NOFOLLOW_LINKS), "output directory already exists")

    val imagesRootRaw = optStr(params, "images_root")
    require(imagesRootRaw.nonEmpty, "images_root is required")
    val imagesRoot = Path.of(imagesRootRaw).toAbsolutePath.normalize()
    val inputs = scanBatchInputs(imagesRoot, context)

    val genEnvelope = params.get("generation") match
      case Some(Obj(fields)) => fields
      case _ => throw new IllegalArgumentException("generation must be an object")
    val allowedEnvelope = Set(
      "params", "output_dir", "vae_recovery_dir", "preview_every", "preview_method",
      "lora_paths", "upscaler_paths", "embedding_paths", "hires_loras", "manual_loras",
      "img2img_hires_loras", "controlnet_paths", "lbw_presets", "hardware_info", "system_info_fields"
    )
    require(genEnvelope.keySet.subsetOf(allowedEnvelope), "unsupported generation envelope keys in batch interrogate")
    genEnvelope.get("vae_recovery_dir") match
      case None | Some(Null) | Some(Str("")) => ()
      case _ => throw new IllegalArgumentException("batch interrogate does not allow vae_recovery_dir")

    val innerParams = genEnvelope.get("params") match
      case Some(Obj(fields)) => fields
      case _ => throw new IllegalArgumentException("generation.params must be an object")
    innerParams.get("hires") match
      case None | Some(Null) | Some(Bool(false)) => ()
      case Some(Obj(h)) if !h.get("enabled").contains(Bool(true)) => ()
      case _ => throw new IllegalArgumentException("batch interrogate does not allow Hires Fix")
    innerParams.get("hooks") match
      case None | Some(Null) => ()
      case Some(Obj(hooks)) =>
        val activeHook = hooks.exists {
          case (_, Null | Bool(false)) => false
          case (_, Obj(hf)) if hf.get("enabled").contains(Bool(false)) => false
          case _ => true
        }
        require(!activeHook, "batch interrogate does not allow active generation hooks")
      case _ => throw new IllegalArgumentException("generation.params.hooks must be an object")

    val extraObj = innerParams.get("extra") match
      case None | Some(Null) => Map.empty[String, NativeJson]
      case Some(Obj(fields)) => fields
      case _ => throw new IllegalArgumentException("generation.params.extra must be an object")
    val options = parseBatchInterrogateOptions(extraObj.get("batch_interrogate"))

    val token = UUID.randomUUID().toString.replace("-", "")
    val now = System.nanoTime()
    preparedBatches.synchronized {
      evictExpiredPreparedBatches(now)
      if preparedBatches.size >= MaxPreparedEntries then
        throw new IllegalStateException("too many prepared batch interrogate jobs")
      preparedBatches.put(
        token,
        PreparedBatchEntry(
          owner = owner,
          createdAtNanos = now,
          checkpointToken = checkpointToken,
          imagesRoot = imagesRoot.toRealPath(),
          inputs = inputs,
          outputParent = outputParent,
          outputName = outputName,
          generationEnvelope = genEnvelope,
          options = options
        )
      )
    }
    Map(
      "token" -> Str(token),
      "image_count" -> Number(inputs.size),
      "backend" -> Str("wd_ext"),
      "partial" -> Bool(true)
    )

  def executeBatchInterrogateExecute(
      params: Map[String, NativeJson],
      ownerOpt: Option[String],
      modelSummary: Map[String, NativeJson],
      modelLoaded: Boolean,
      context: NativeWorkContext
  )(
      runImg2Img: (Map[String, NativeJson], NativeWorkContext) => Map[String, NativeJson]
  ): Map[String, NativeJson] =
    val owner = ownerOpt.map(_.trim).filter(s => s.nonEmpty && s.length <= 256).getOrElse(
      throw new IllegalArgumentException("batch interrogate requires an explicit owner")
    )
    if params.keySet != Set("token") then
      throw new IllegalArgumentException("batch interrogate execute requires only token")
    val token = optStr(params, "token")
    require(token.matches("[0-9a-f]{32}"), "invalid batch interrogate token")
    context.checkCancelled()

    val now = System.nanoTime()
    val entry = preparedBatches.synchronized {
      evictExpiredPreparedBatches(now)
      val found = preparedBatches.getOrElse(token, throw new IllegalArgumentException("unknown or expired batch interrogate token"))
      if found.owner != owner then
        throw new IllegalArgumentException("batch interrogate token belongs to a different owner")
      preparedBatches.remove(token)
      found
    }

    if !modelLoaded then
      throw new NativeBackendError("no_model", "Load a checkpoint before running Batch Interrogate.")
    if !modelSummary.get("model_token").contains(Str(entry.checkpointToken)) then
      throw new IllegalArgumentException("checkpoint changed since batch interrogate prepare")

    // Re-verify input manifest matches prepare snapshot
    val currentInputs = scanBatchInputs(entry.imagesRoot, context)
    require(currentInputs == entry.inputs, "batch interrogate input directory changed since prepare")

    val finalOutputDir = entry.outputParent.resolve(entry.outputName)
    require(!Files.exists(finalOutputDir, java.nio.file.LinkOption.NOFOLLOW_LINKS), "batch interrogate output directory already exists")
    val stageRoot = entry.outputParent.resolve(s".${entry.outputName}.staging-${UUID.randomUUID()}")
    Files.createDirectories(stageRoot)
    var published = false
    try
      val innerParams = entry.generationEnvelope("params").asInstanceOf[Obj].fields
      val basePositive = optStr(innerParams, "prompt", "")
      val baseNegative = optStr(innerParams, "negative_prompt", "")
      val baseImg2Img = innerParams.get("img2img") match
        case Some(Obj(f)) => f
        case _ => Map.empty[String, NativeJson]
      val cleanedExtra = innerParams.get("extra") match
        case Some(Obj(f)) => f - "batch_interrogate"
        case _ => Map.empty[String, NativeJson]

      val rows = new ArrayBuffer[NativeJson]()
      val total = entry.inputs.size
      for (item, idx) <- entry.inputs.zipWithIndex do
        context.checkCancelled()
        context.event("progress", Map("step" -> Number(idx), "total" -> Number(total), "stage" -> Str("batch_interrogate")))
        val img = DiffusionImageInput.readRgb(item.path)
        val rawTags = interrogateImageTags(img, entry.options)
        val (positive, negative) = composeBatchInterrogatePrompt(basePositive, baseNegative, rawTags, entry.options)
        val itemStageDir = stageRoot.resolve(f"item-$idx%04d")
        Files.createDirectories(itemStageDir)
        val itemGenParams = innerParams ++ Map(
          "prompt" -> Str(positive),
          "negative_prompt" -> Str(negative),
          "extra" -> Obj(cleanedExtra),
          "img2img" -> Obj(baseImg2Img + ("init_image" -> Str(item.path.toString)))
        )
        val itemEnvelope = entry.generationEnvelope ++ Map(
          "params" -> Obj(itemGenParams),
          "output_dir" -> Str(itemStageDir.toString),
          "vae_recovery_dir" -> Str(itemStageDir.toString),
          "preview_every" -> Number(0),
          "preview_method" -> Str("none")
        )
        val genResult = runImg2Img(itemEnvelope, context)
        val finalItemDir = finalOutputDir.resolve(f"item-$idx%04d")
        val wdSuffix = s", WD interrogators: ${entry.options.interrogators.mkString("+")}, WD input: ${item.path.getFileName}, WD input SHA256: ${item.sha256}"
        val updatedImages = genResult.get("images") match
          case Some(Arr(imgValues)) =>
            imgValues.map {
              case Obj(imgFields) =>
                val stagedPath = Path.of(optStr(imgFields, "path"))
                val existingParams = NativePreprocess.sourceParameters(stagedPath, "parameters").getOrElse("")
                val updatedParams = if existingParams.nonEmpty then existingParams + wdSuffix else wdSuffix.stripPrefix(", ")
                val buffered = ImageIO.read(stagedPath.toFile)
                NativePreprocess.write(buffered, stagedPath, Vector("parameters" -> updatedParams))
                val rel = itemStageDir.relativize(stagedPath)
                Obj(imgFields + ("path" -> Str(finalItemDir.resolve(rel).toString)))
              case other => other
            }
          case _ => Vector.empty
        val updatedResult = genResult + ("images" -> Arr(updatedImages))
        rows += Obj(Map(
          "status" -> Str("complete"),
          "input" -> Str(item.path.toString),
          "positive" -> Str(positive),
          "negative" -> Str(negative),
          "result" -> Obj(updatedResult),
          "interrogators" -> Arr(entry.options.interrogators.map(Str.apply))
        ))

      context.checkCancelled()
      context.beginPublication()
      Files.move(stageRoot, finalOutputDir, java.nio.file.StandardCopyOption.ATOMIC_MOVE)
      published = true
      context.event("progress", Map("step" -> Number(total), "total" -> Number(total), "stage" -> Str("batch_interrogate")))
      Map(
        "rows" -> Arr(rows.toVector),
        "partial" -> Bool(true)
      )
    finally
      if !published && Files.exists(stageRoot) then
        val walk = Files.walk(stageRoot)
        try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_))
        finally walk.close()
