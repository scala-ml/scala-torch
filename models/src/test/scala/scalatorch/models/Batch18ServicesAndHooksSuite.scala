package scalatorch.models.worker

import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.file.{ Files, Path }
import javax.imageio.ImageIO
import munit.FunSuite
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

final class Batch18ServicesAndHooksSuite extends FunSuite:

  private def tinyModelRoot: Path =
    Seq(
      Path.of(".superpowers/models/tiny-sd-pipe"),
      Path.of("../.superpowers/models/tiny-sd-pipe")
    ).find(p => Files.isDirectory(p.resolve("text_encoder"))).getOrElse {
      assume(false,"tiny-sd-pipe fixture not present (.superpowers/models/tiny-sd-pipe)")
      Path.of(".")
    }.toRealPath()

  private def makeBackend(): NativeDiffusionBackend =
    val root = tinyModelRoot
    new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))

  private def makeWorkContext(): NativeWorkContext =
    new NativeWorkContext(() => false, () => true, (_, _) => ())

  private def createSalientTestImage(w: Int = 64, h: Int = 64): BufferedImage =
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = img.createGraphics()
    try
      g.setColor(new Color(30, 30, 35))
      g.fillRect(0, 0, w, h)
      // Draw a warm skin-colored / high-contrast face-like patch in the center
      g.setColor(new Color(220, 165, 135))
      g.fillOval(w / 4, h / 4, w / 2, h / 2)
    finally g.dispose()
    img

  test("Adetailer and BatchFaceSwap detect, filter, mask, matchColors, and expand prompts") {
    val img = createSalientTestImage(64, 64)
    val det = Adetailer.detect(img, "face_yolov8n.pt", 0.3)
    assert(det.boxes.nonEmpty, "expected at least one salient region detected")

    val filtered = Adetailer.filterByRatio(det, 0.01, 0.95)
    assertEquals(filtered.boxes.size, det.boxes.size)

    val top1 = Adetailer.filterKLargest(det, 1)
    assertEquals(top1.boxes.size, 1)

    val dilated = Adetailer.offsetDilate(top1.masks.head, 64, 64, 2, -2, 4)
    assert(dilated.count(_ > 0) > top1.masks.head.count(_ > 0))

    val inverted = Adetailer.mergeInvert(Vector(dilated), "Merge and Invert")
    assertEquals(inverted.size, 1)
    assertEquals(inverted.head.count(_ == 0), dilated.count(_ == 255))

    val ellipseMasks = Adetailer.shapedMasks(top1, "ellipse", 1.2)
    assertEquals(ellipseMasks.size, 1)
    assert(ellipseMasks.head.exists(_ == 255))

    val prompts = Adetailer.unitPrompts("closeup of [PROMPT] [SEP] smiling [PROMPT]", "1girl", 3)
    assertEquals(prompts, Vector("closeup of 1girl", "smiling 1girl", "smiling 1girl"))

    val refImg = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 64; x <- 0 until 64 do refImg.setRGB(x, y, new Color(180, 120, 90).getRGB)
    val matched = Adetailer.matchColors(img, refImg)
    assertEquals(matched.getWidth, 64)
    assertEquals(matched.getHeight, 64)

    val bfsCfg = BatchFaceSwapConfig(detector = "face_yolov8n.pt", confidence = 0.35, maxFaces = 2, maskShape = "ellipse", maskSize = 1.2, resolution = 512)
    val unit = BatchFaceSwap.toAdetailerUnit(bfsCfg)
    assertEquals(unit.model, "face_yolov8n.pt")
    assertEquals(unit.maskKLargest, 2)
    assertEquals(unit.useInpaintWidthHeight, true)
    assertEquals(unit.inpaintWidth, 512)
  }

  test("NsfwChecker scaleFactor, blurRadii, filters, censorFinal, and image_tool execute accurately") {
    assert(math.abs(NsfwChecker.scaleFactor(512, 512) - 1.0) < 1e-6)
    val radii = NsfwChecker.blurRadii(Array(0, 128, 255), 10.0, 3.0)
    assertEquals(radii(0), 0.0)
    assert(math.abs(radii(2) - 10.0) < 1e-6)

    val img = createSalientTestImage(64, 64)
    val mask = NsfwChecker.drawMaskShape(64, 64, Vector((16.0, 16.0, 48.0, 48.0)), "Ellipse", 0.5)
    for filter <- Vector("Variable blur", "Gaussian Blur", "Pixelate", "Fill color", "No censor") do
      val out = NsfwChecker.applyFilter(img, mask, filter, blurRadius = 6.0, pixelationFactor = 4.0, color = "#112233")
      assertEquals(out.getWidth, 64)
      assertEquals(out.getHeight, 64)

    val cfg = NsfwChecker.NsfwConfig(genFilterType = "Fill color", fillColor = "#000000", saveBeforeCensor = true, saveMask = true)
    val (censored, aux) = NsfwChecker.censorFinal(img, cfg)
    assertEquals(censored.getWidth, 64)
    assertEquals(aux.map(_._1), Vector("-before-censor", "-censor-mask"))

    val tmpDir = Files.createTempDirectory("batch18-nsfw-")
    val srcFile = tmpDir.resolve("input.png")
    ImageIO.write(img, "png", srcFile.toFile)
    val backend = makeBackend()
    val res = backend.execute(
      NativeRequest("1", "image_tool", Map(
        "tool" -> Str("nsfw_checker"),
        "image_path" -> Str(srcFile.toString),
        "output_dir" -> Str(tmpDir.toString),
        "filter_type" -> Str("Pixelate"),
        "output_mask" -> Bool(true)
      ), None),
      makeWorkContext()
    )
    assert(Files.isRegularFile(Path.of(res("path").asInstanceOf[Str].value)))
    assert(res("outputs").asInstanceOf[Obj].fields.contains("mask"))
  }

  test("AestheticScore and additional image tools (person_mask, remove_background, depth_mask, waifu2x) work") {
    val img = createSalientTestImage(64, 64)
    val aCfg = AestheticScoreConfig(
      improved = true,
      simulacra = "vit-l-14",
      cafeAesthetic = true,
      cafeStyle = true,
      cafeWaifu = true,
      pngChunks = true,
      generationFields = true,
      hash = true
    )
    val scores = AestheticScore.scoreImage(img, aCfg)
    assertEquals(scores.size, 5)
    val infoFields = AestheticScore.infotextFields(scores).toMap
    assert(infoFields.contains("aesthetic_score"))
    assert(infoFields.contains("Aesthetic Scorer ViT-L/14"))
    assert(infoFields.contains("Cafe aesthetic"))
    val chunks = AestheticScore.pngChunks(scores).toMap
    assert(chunks.contains("aesthetic_score"))
    assert(chunks.contains("score"))
    assert(chunks.contains("cafe_style"))
    val h = AestheticScore.paramsHash(20, "Euler", 7.0, 42L, 512, 512, "deadbeef", "1girl", "bad")
    assertEquals(h.length, 32)

    val tmpDir = Files.createTempDirectory("batch18-imgtools-")
    val srcFile = tmpDir.resolve("input.png")
    ImageIO.write(img, "png", srcFile.toFile)
    val backend = makeBackend()
    val ctx = makeWorkContext()

    for toolName <- Vector("aesthetic_score", "person_mask", "remove_background", "depth_mask", "waifu2x") do
      val extra = if toolName == "waifu2x" then Map("scale" -> Number(2.0)) else Map.empty[String, NativeJson]
      val r = backend.execute(
        NativeRequest(s"tool-$toolName", "image_tool", Map(
          "tool" -> Str(toolName),
          "image_path" -> Str(srcFile.toString),
          "output_dir" -> Str(tmpDir.toString)
        ) ++ extra, None),
        ctx
      )
      val outPath = Path.of(r("path").asInstanceOf[Str].value)
      assert(Files.isRegularFile(outPath), s"expected output file for $toolName")
  }

  test("AnimateDiff prompt travel, context windows, frequency filter, GIF encoding, and animatediff_encode RPC") {
    val travelText =
      """masterpiece, best quality
        |0: closed eyes
        |8: open eyes, smiling
        |highly detailed""".stripMargin
    val parsed = AnimateDiff.parseTravel(travelText, 16)
    assert(parsed.isDefined)
    val (kf, prompts) = parsed.get
    assertEquals(kf, Vector(0, 8))
    assertEquals(prompts, Vector(
      "masterpiece, best quality, closed eyes, highly detailed",
      "masterpiece, best quality, open eyes, smiling, highly detailed"
    ))
    val weights = AnimateDiff.travelWeights(kf, 16, loop = true)
    assertEquals(weights.size, 16)
    assertEquals(weights(0), (0, 1, 0.0))
    assertEquals(weights(4), (0, 1, 0.5))
    assertEquals(weights(8), (1, 0, 0.0))

    val windows = AnimateDiff.contextWindows(frames = 24, length = 16, overlap = 4, stride = 2, closedLoop = "R+P")
    assert(windows.nonEmpty)
    assert(windows.forall(_.size <= 16))

    val filt = AnimateDiff.freqFilter(8, 8, 8, "butterworth", 0.25, 0.25, 4)
    assertEquals(filt.length, 512)
    assert(filt.forall(v => v >= 0.0f && v <= 1.0f))

    assertEquals(AnimateDiff.pingpong(Vector(1, 2, 3, 4)), Vector(1, 2, 3, 4, 3, 2))

    val tmpDir = Files.createTempDirectory("batch18-anim-")
    val f0 = tmpDir.resolve("f0.png")
    val f1 = tmpDir.resolve("f1.png")
    val f2 = tmpDir.resolve("f2.png")
    ImageIO.write(createSalientTestImage(32, 32), "png", f0.toFile)
    ImageIO.write(createSalientTestImage(32, 32), "png", f1.toFile)
    ImageIO.write(createSalientTestImage(32, 32), "png", f2.toFile)

    val backend = makeBackend()
    val encRes = backend.execute(
      NativeRequest("anim-1", "animatediff_encode", Map(
        "images" -> Arr(Vector(Str(f0.toString), Str(f1.toString), Str(f2.toString))),
        "formats" -> Arr(Vector(Str("GIF"))),
        "closed_loop" -> Str("A"),
        "fps" -> Number(8),
        "output_dir" -> Str(tmpDir.toString)
      ), None),
      makeWorkContext()
    )
    assertEquals(encRes("frames").asInstanceOf[Number].value.toInt, 4)
    val gifPath = Path.of(encRes("outputs").asInstanceOf[Obj].fields("GIF").asInstanceOf[Str].value)
    assert(Files.isRegularFile(gifPath) && Files.size(gifPath) > 0L)
  }

  test("All Batch 18 RPC endpoints and end-to-end txt2img post-generation hooks execute cleanly") {
    val backend = makeBackend()
    val ctx = makeWorkContext()
    backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)

    // Verify all Batch 18 features are advertised
    val advertised = backend.features.toSet
    for feat <- Vector(
      "adetailer", "batch_face_swap", "face_restore", "nsfw_checker", "aesthetic_score",
      "animatediff", "cn_animation", "controlnet_fastload_data", "photomaker", "caption",
      "img2vid", "novel_view", "custom_code", "ldsr", "marigold", "controlnet_aux",
      "temporal_lab", "clip_interrogator", "chara_searcher"
    ) do assert(advertised.contains(feat), s"missing feature advertisement: $feat")

    val tmpDir = Files.createTempDirectory("batch18-rpcs-")

    // 1. embedding_merge
    val emInspect = backend.execute(
      NativeRequest("em-1", "embedding_merge", Map(
        "operation" -> Str("inspect"),
        "text" -> Str("<'cat' + 'dog'>")
      ), None),
      ctx
    )
    assert(emInspect("streams").asInstanceOf[Arr].values.nonEmpty)
    val emSave = backend.execute(
      NativeRequest("em-2", "embedding_merge", Map(
        "operation" -> Str("save"),
        "text" -> Str("'cat' + 'dog'"),
        "name" -> Str("catdog"),
        "output_dir" -> Str(tmpDir.toString)
      ), None),
      ctx
    )
    assertEquals(emSave("active"), Bool(true))
    assert(Files.isRegularFile(Path.of(emSave("paths").asInstanceOf[Arr].values.head.asInstanceOf[Str].value)))

    // 2. aesthetic_embedding
    val sampleImg = tmpDir.resolve("sample.png")
    ImageIO.write(createSalientTestImage(32, 32), "png", sampleImg.toFile)
    val aeCreate = backend.execute(
      NativeRequest("ae-1", "aesthetic_embedding", Map(
        "op" -> Str("create"),
        "name" -> Str("test_aes"),
        "images" -> Arr(Vector(Str(sampleImg.toString))),
        "output_dir" -> Str(tmpDir.toString),
        "overwrite" -> Bool(true)
      ), None),
      ctx
    )
    assert(Files.isRegularFile(Path.of(aeCreate("path").asInstanceOf[Str].value)))

    // 3. controlnet_fastload_data
    val cnfSavePath = tmpDir.resolve("fastload.cni")
    val unitsWire = Obj(Map("type" -> Str("list"), "items" -> Arr(Vector(Obj(Map("model" -> Str("canny")))))))
    backend.execute(
      NativeRequest("cnf-1", "controlnet_fastload_data", Map(
        "operation" -> Str("save"),
        "path" -> Str(cnfSavePath.toString),
        "units" -> unitsWire
      ), None),
      ctx
    )
    val cnfInspect = backend.execute(
      NativeRequest("cnf-2", "controlnet_fastload_data", Map(
        "operation" -> Str("inspect"),
        "path" -> Str(cnfSavePath.toString)
      ), None),
      ctx
    )
    assertEquals(cnfInspect("units"), unitsWire)

    // 4. temporal_lab
    val seqDir = tmpDir.resolve("seq")
    Files.createDirectories(seqDir)
    ImageIO.write(createSalientTestImage(32, 32), "png", seqDir.resolve("001.png").toFile)
    ImageIO.write(createSalientTestImage(32, 32), "png", seqDir.resolve("002.png").toFile)
    val tlRes = backend.execute(
      NativeRequest("tl-1", "temporal_lab", Map(
        "operation" -> Str("deflicker"),
        "frames_dir" -> Str(seqDir.toString),
        "blend" -> Number(0.5),
        "output_parent" -> Str(tmpDir.toString),
        "output_name" -> Str("deflickered")
      ), None),
      ctx
    )
    assertEquals(tlRes("frame_count"), Number(2))

    // 5. cn_animation
    val cnFlow = backend.execute(
      NativeRequest("cna-1", "cn_animation", Map(
        "operation" -> Str("flow"),
        "prev_frame" -> Str(seqDir.resolve("001.png").toString),
        "cur_frame" -> Str(seqDir.resolve("002.png").toString),
        "prev_styled" -> Str(seqDir.resolve("001.png").toString),
        "output_dir" -> Str(tmpDir.toString),
        "name" -> Str("step1")
      ), None),
      ctx
    )
    val cnFinishOut = tmpDir.resolve("step1-final.png")
    val cnFinish = backend.execute(
      NativeRequest("cna-2", "cn_animation", Map(
        "operation" -> Str("finish"),
        "image" -> Str(seqDir.resolve("002.png").toString),
        "reference" -> Str(seqDir.resolve("001.png").toString),
        "alpha" -> cnFlow("alpha"),
        "warped" -> cnFlow("warped"),
        "output" -> Str(cnFinishOut.toString)
      ), None),
      ctx
    )
    assert(Files.isRegularFile(Path.of(cnFinish("path").asInstanceOf[Str].value)))

    // 6. restore_faces, caption, clip_interrogator, chara_searcher, img2vid, novel_view
    val dummyFaceModel = tmpDir.resolve("CodeFormer.pth")
    Files.write(dummyFaceModel, Array[Byte](1, 2, 3, 4))
    val rfRes = backend.execute(
      NativeRequest("rf-1", "restore_faces", Map(
        "model" -> Str(dummyFaceModel.toString),
        "image_path" -> Str(sampleImg.toString),
        "visibility" -> Number(0.8),
        "output_dir" -> Str(tmpDir.toString)
      ), None),
      ctx
    )
    assert(Files.isRegularFile(Path.of(rfRes("path").asInstanceOf[Str].value)))

    val capRes = backend.execute(NativeRequest("cap-1", "caption", Map("image_path" -> Str(sampleImg.toString)), None), ctx)
    assert(capRes("caption").asInstanceOf[Str].value.nonEmpty)

    val ciRes = backend.execute(NativeRequest("ci-1", "clip_interrogator", Map("operation" -> Str("prompt"), "image_path" -> Str(sampleImg.toString)), None), ctx)
    assert(ciRes("prompt").asInstanceOf[Str].value.nonEmpty)

    val csRes = backend.execute(NativeRequest("cs-1", "chara_searcher", Map("operation" -> Str("embed_image"), "image_path" -> Str(sampleImg.toString)), None), ctx)
    assertEquals(csRes("embedding").asInstanceOf[Arr].values.size, 512)

    val i2vRes = backend.execute(NativeRequest("i2v-1", "img2vid", Map("image_path" -> Str(sampleImg.toString), "frames" -> Number(3), "output_dir" -> Str(tmpDir.toString)), None), ctx)
    assertEquals(i2vRes("images").asInstanceOf[Arr].values.size, 3)

    // 7. End-to-end txt2img with adetailer, batch_face_swap, face_restore, nsfw_checker, and aesthetic_score hooks
    val outDir = tmpDir.resolve("gen")
    Files.createDirectories(outDir)
    val genRes = backend.execute(
      NativeRequest("gen-1", "txt2img", Map(
        "output_dir" -> Str(outDir.toString),
        "preview_every" -> Number(0),
        "preview_method" -> Str("none"),
        "params" -> Obj(Map(
          "prompt" -> Str("a portrait"),
          "negative_prompt" -> Str(""),
          "width" -> Number(8),
          "height" -> Number(8),
          "steps" -> Number(2),
          "seed" -> Number(42),
          "cfg_scale" -> Number(7.0),
          "sampler_name" -> Str("Euler"),
          "hooks" -> Obj(Map(
            "adetailer" -> Obj(Map("enabled" -> Bool(true), "model" -> Str("face_yolov8n.pt"), "confidence" -> Number(0.1), "denoising_strength" -> Number(0.2))),
            "face_restore" -> Obj(Map("enabled" -> Bool(true), "visibility" -> Number(0.5))),
            "nsfw_checker" -> Obj(Map("enabled" -> Bool(true), "gen_filter_type" -> Str("Gaussian Blur"))),
            "aesthetic_score" -> Obj(Map("enabled" -> Bool(true), "improved" -> Bool(true), "generation_fields" -> Bool(true), "hash" -> Bool(true)))
          ))
        ))
      ), None),
      ctx
    )
    val firstImgObj = genRes("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
    assert(Files.isRegularFile(Path.of(firstImgObj("path").asInstanceOf[Str].value)))
    assert(firstImgObj.contains("aesthetic_scores"))
    assert(firstImgObj.contains("aesthetic_score"))
    assert(firstImgObj.contains("png_chunks"))

    // 8. batch_interrogate_prepare & batch_interrogate_execute + extra preprocessors & upscalers
    assert(backend.features.contains("batch_interrogate_wd_v3_partial"))
    assertEquals(backend.supported.size, 42)
    val modelToken = backend.systemStatus("model").asInstanceOf[Obj].fields("model_token").asInstanceOf[Str].value
    val batchInputDir = tmpDir.resolve("wd-inputs")
    val batchOutParent = tmpDir.resolve("wd-outputs")
    Files.createDirectories(batchInputDir)
    Files.createDirectories(batchOutParent)
    ImageIO.write(createSalientTestImage(16, 16), "png", batchInputDir.resolve("01.png").toFile)
    ImageIO.write(createSalientTestImage(16, 16), "png", batchInputDir.resolve("02.png").toFile)

    val prepRes = backend.execute(
      NativeRequest("wd-prep", "batch_interrogate_prepare", Map(
        "images_root" -> Str(batchInputDir.toString),
        "output_parent" -> Str(batchOutParent.toString),
        "output_name" -> Str("run1"),
        "checkpoint_token" -> Str(modelToken),
        "generation" -> Obj(Map(
          "params" -> Obj(Map(
            "prompt" -> Str("masterpiece"),
            "negative_prompt" -> Str("lowres"),
            "width" -> Number(16),
            "height" -> Number(16),
            "steps" -> Number(1),
            "seed" -> Number(42),
            "cfg_scale" -> Number(7.0),
            "sampler_name" -> Str("Euler"),
            "img2img" -> Obj(Map("denoising_strength" -> Number(0.5))),
            "extra" -> Obj(Map(
              "batch_interrogate" -> Obj(Map(
                "mode" -> Str("Append"),
                "interrogators" -> Arr(Vector(Str("ViT"), Str("SwinV2"))),
                "combine_mode" -> Str("Average"),
                "general_threshold" -> Number(0.35),
                "character_threshold" -> Number(0.5),
                "prepend_tags" -> Str("best quality"),
                "add_tags" -> Str("^_^, old_tag"),
                "replace_tags" -> Str("old_tag -> replaced_tag"),
                "remove_punctuation" -> Bool(true)
              ))
            ))
          ))
        ))
      ), Some("owner-1")),
      ctx
    )
    assertEquals(prepRes("image_count"), Number(2))
    assertEquals(prepRes("backend"), Str("wd_ext"))
    assertEquals(prepRes("partial"), Bool(true))
    val wdToken = prepRes("token").asInstanceOf[Str].value
    assert(wdToken.matches("[0-9a-f]{32}"))

    val execRes = backend.execute(
      NativeRequest("wd-exec", "batch_interrogate_execute", Map("token" -> Str(wdToken)), Some("owner-1")),
      ctx
    )
    assertEquals(execRes("partial"), Bool(true))
    val rows = execRes("rows").asInstanceOf[Arr].values
    assertEquals(rows.size, 2)
    val row0 = rows.head.asInstanceOf[Obj].fields
    assertEquals(row0("status"), Str("complete"))
    assert(row0("positive").asInstanceOf[Str].value.startsWith("masterpiece, best quality"))
    assert(row0("positive").asInstanceOf[Str].value.contains("^_^"))
    assert(row0("positive").asInstanceOf[Str].value.contains("replaced tag"))
    val row0ImgPath = Path.of(row0("result").asInstanceOf[Obj].fields("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
    assert(Files.isRegularFile(row0ImgPath))
    val pngMeta = NativePreprocess.sourceParameters(row0ImgPath).getOrElse("")
    assert(pngMeta.contains("WD interrogators: ViT+SwinV2"))
    assert(pngMeta.contains("WD input: 01.png"))
    assert(pngMeta.contains("WD input SHA256:"))

    // Single-use token cannot be reused
    intercept[IllegalArgumentException] {
      backend.execute(NativeRequest("wd-exec-2", "batch_interrogate_execute", Map("token" -> Str(wdToken)), Some("owner-1")), ctx)
    }

    // Extra ControlNet preprocessors & LDSR/Waifu2x upscalers
    for pre <- Vector("depth", "depth_marigold", "normal_marigold", "openpose", "lineart", "lineart_anime", "softedge_hed", "mlsd", "normal_bae") do
      val preRes = backend.execute(
        NativeRequest(s"pre-$pre", "preprocess", Map(
          "preprocessor" -> Str(pre),
          "image_path" -> Str(sampleImg.toString),
          "output_dir" -> Str(tmpDir.toString)
        ), None),
        ctx
      )
      assert(Files.isRegularFile(Path.of(preRes("path").asInstanceOf[Str].value)))

    for up <- Vector("LDSR", "Waifu2x") do
      val upRes = backend.execute(
        NativeRequest(s"up-$up", "upscale", Map(
          "upscaler" -> Str(up),
          "scale" -> Number(1.5),
          "image_path" -> Str(sampleImg.toString),
          "output_dir" -> Str(tmpDir.toString)
        ), None),
        ctx
      )
      assert(Files.isRegularFile(Path.of(upRes("path").asInstanceOf[Str].value)))
  }
