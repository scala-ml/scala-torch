package scalatorch.models.worker

import munit.FunSuite
import scalatorch.*
import scalatorch.models.*
import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class SaveIntermediateStepsAnimImg2ImgAltCdTunerSuite extends FunSuite:
  import NativeJson.*

  private def deleteTree(path: java.nio.file.Path): Unit =
    if Files.exists(path) then
      import scala.jdk.CollectionConverters.*
      val stream = Files.walk(path)
      try stream.iterator().asScala.toSeq.reverse.foreach(Files.deleteIfExists)
      finally stream.close()

  test("SaveIntermediateConfig and StepsAnimationConfig validation and Session recording") {
    val cfg = SaveIntermediateConfig.validate(
      SaveIntermediateConfig(everyN = 2, startAt = 3, stopAt = 7, mode = "Video", ffmpegBat = true, batOnly = true, finalSave = true)
    )
    assertEquals(cfg.startAt, 2)
    assertEquals(cfg.stopAt, 6)
    assert(!cfg.isSelected(1))
    assert(cfg.isSelected(2))
    assert(cfg.isSelected(4))
    assert(cfg.isSelected(6))
    assert(!cfg.isSelected(8))

    intercept[IllegalArgumentException] {
      SaveIntermediateConfig.validate(SaveIntermediateConfig(smooth = true, fps = 1.0))
    }
    intercept[IllegalArgumentException] {
      StepsAnimationConfig.validate(StepsAnimationConfig(framesDir = "../escape"))
    }
    intercept[IllegalArgumentException] {
      StepsAnimationConfig.validate(StepsAnimationConfig(outputDir = "C:/abs/path"))
    }

    val dir = Files.createTempDirectory("scala-torch-inter-sess-")
    try
      val sess = new NativeIntermediateRecorder.Session(
        dir,
        Some(cfg),
        None,
        Map("prompt" -> Str("a cat"), "negative_prompt" -> Str("blur"), "steps" -> Number(6), "sampler" -> Str("Euler"), "cfg_scale" -> Number(7)),
        () => false
      )
      val dummyLat = Tensor.fromFloats(Array.fill(4 * 2 * 2)(0.1f), Shape.from(Seq(1L, 4L, 2L, 2L)))
      def decodeImg(lat: Tensor[Float]): BufferedImage =
        new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB)
      for step <- 1 to 6 do
        sess.observeStep("base", step, 0, 42L, 0, 0, dummyLat, dummyLat, dummyLat, decodeImg, _ => "")
      sess.recordFinal(decodeImg(dummyLat), 42L, 0, 0)
      val artifacts = sess.finish()
      // 3 step images (steps 2, 4, 6) + 1 final image + 1 ffmpeg_recipe + 1 ffmpeg_playlist + 1 manifest = 7 artifacts
      assertEquals(artifacts.size, 7)
      val kinds = artifacts.map(_.asInstanceOf[Obj].fields("type").asInstanceOf[Str].value)
      assertEquals(kinds.count(_ == "image"), 4)
      assert(kinds.contains("ffmpeg_recipe"))
      assert(kinds.contains("ffmpeg_playlist"))
      assert(kinds.contains("manifest"))
    finally deleteTree(dir)
  }

  test("Img2ImgAlt DDIM inversion and randomness mixing") {
    val schedule = DiffusionNoiseSchedule.fromBetas(
      Vector.tabulate(100)(i => 0.00085 + (0.012 - 0.00085) * i / 99.0)
    )
    val x0 = Tensor.fromFloats(Array.tabulate(4 * 2 * 2)(i => (i - 8) * 0.05f), Shape.from(Seq(1L, 4L, 2L, 2L)))
    var evalCount = 0
    val inverted = Img2ImgAlt.invertLatents(
      x0,
      steps = 4,
      cfgScale = 1.5,
      schedule = schedule,
      predictEps = (x, t, isCond) => {
        evalCount += 1
        val scale = if isCond then 0.2f else 0.1f
        Tensor.fromFloats(x.toFloatArray.map(_ * scale), x.shape)
      }
    )
    assertEquals(evalCount, 8) // 4 steps * 2 branches (guided)
    assertEquals(inverted.shape, x0.shape)
    assert(inverted.toFloatArray.forall(_.isFinite))
    assert(inverted.toFloatArray.toSeq != x0.toFloatArray.toSeq)

    val noise = Tensor.fromFloats(Array.fill(16)(1.0f), x0.shape)
    val mixed0 = Img2ImgAlt.mixRandomness(inverted, noise, 0.0)
    assertEquals(mixed0.toFloatArray.toSeq, inverted.toFloatArray.toSeq)
    val mixed1 = Img2ImgAlt.mixRandomness(inverted, noise, 1.0)
    for (a, b) <- mixed1.toFloatArray.zip(noise.toFloatArray) do
      assertEqualsDouble(a.toDouble, b.toDouble, 1e-6)
  }

  test("CdTuner step shift, regional color map, and saturation") {
    val cfg = CdTuner.validate(
      CdTunerConfig(
        d1 = 2.0,
        d2 = 1.0,
        con1 = 1.5,
        con2 = 2.0,
        bri = 1.0,
        col1 = -1.0,
        col2 = 0.5,
        col3 = 0.5,
        sat = 1.0,
        sat2 = 1.0,
        splitRatio = "1,1",
        splitBy = "Horizontal",
        colors = "1, 0, -1 ; -1, 1, 0",
        mapStopStep = 2,
        mapStrength = 1.0
      )
    )
    val base = Tensor.fromFloats(Array.tabulate(4 * 4 * 4)(i => ((i % 7) - 3) * 0.2f), Shape.from(Seq(1L, 4L, 4L, 4L)))
    // Step 0 applies detail/con1 and regional color map
    val s0 = CdTuner.applyStepShift(base, cfg, sdxl = false, stepIndex = 0, totalSteps = 4, isHires = false)
    assert(s0.toFloatArray.forall(_.isFinite))
    assert(s0.toFloatArray.toSeq != base.toFloatArray.toSeq)
    // Step 2 (totalSteps - 2 - dis = 4 - 2 - 0 = 2) applies con2/bri/col1..3
    val s2 = CdTuner.applyStepShift(base, cfg, sdxl = false, stepIndex = 2, totalSteps = 4, isHires = false)
    assert(s2.toFloatArray.forall(_.isFinite))
    assert(s2.toFloatArray.toSeq != base.toFloatArray.toSeq)
    // Step 3 (totalSteps - 1) applies saturation
    val s3 = CdTuner.applyStepShift(base, cfg, sdxl = false, stepIndex = 3, totalSteps = 4, isHires = false)
    assert(s3.toFloatArray.forall(_.isFinite))
    assert(s3.toFloatArray.toSeq != base.toFloatArray.toSeq)

    intercept[IllegalArgumentException] {
      CdTuner.validate(CdTunerConfig(splitRatio = "1,1,1", colors = "1,0,0 ; 0,1,0"))
    }
  }

  test("NativeDiffusionBackend advertises new features and executes save_intermediate, steps_animation, img2imgalt, and cd_tuner") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val modelRoot = java.nio.file.Path.of(configured.get).getParent
    val dir = Files.createTempDirectory("scala-torch-batch14-backend-")
    try
      val outDir = dir.resolve("out")
      Files.createDirectories(outDir)
      val backend = new NativeDiffusionBackend(modelRoot, modelRoot.resolve("euler-demo.json"))
      val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())

      for feat <- Seq("controlnet", "controlnet_preprocess_extras", "system_info", "save_intermediate", "steps_animation", "img2imgalt", "cd_tuner") do
        assert(backend.features.contains(feat), s"missing advertised feature: $feat")

      backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)

      // 1. txt2img with save_intermediate and cd_tuner
      val resSave = backend.execute(
        NativeRequest(
          "gen-save",
          "txt2img",
          Map(
            "output_dir" -> Str(outDir.toString),
            "preview_every" -> Number(0),
            "preview_method" -> Str("none"),
            "params" -> Obj(
              Map(
                "prompt" -> Str("tiny test"),
                "width" -> Number(16),
                "height" -> Number(16),
                "steps" -> Number(2),
                "seed" -> Number(42),
                "hooks" -> Obj(
                  Map(
                    "save_intermediate" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "every_n" -> Number(1),
                        "final_save" -> Bool(true)
                      )
                    ),
                    "cd_tuner" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "d1" -> Number(1.5),
                        "bri" -> Number(0.5),
                        "sat" -> Number(0.5)
                      )
                    )
                  )
                )
              )
            )
          ),
          None
        ),
        ctx
      )
      val saveArtifacts = resSave("intermediate_artifacts").asInstanceOf[Arr].values
      // 2 step images + 1 final image + 1 manifest = 4
      assertEquals(saveArtifacts.size, 4)

      // 2. txt2img with steps_animation
      val resAnim = backend.execute(
        NativeRequest(
          "gen-anim",
          "txt2img",
          Map(
            "output_dir" -> Str(outDir.toString),
            "preview_every" -> Number(0),
            "preview_method" -> Str("none"),
            "params" -> Obj(
              Map(
                "prompt" -> Str("tiny anim"),
                "width" -> Number(16),
                "height" -> Number(16),
                "steps" -> Number(2),
                "seed" -> Number(43),
                "hooks" -> Obj(
                  Map(
                    "steps_animation" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "skip_initial" -> Number(0),
                        "create_output" -> Bool(false)
                      )
                    )
                  )
                )
              )
            )
          ),
          None
        ),
        ctx
      )
      val animArtifacts = resAnim("intermediate_artifacts").asInstanceOf[Arr].values
      // 2 step images + 1 manifest = 3
      assertEquals(animArtifacts.size, 3)

      // 3. img2img with img2imgalt
      val initImgPath = dir.resolve("init.png")
      val initImg = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB)
      ImageIO.write(initImg, "png", initImgPath.toFile)
      val resAlt = backend.execute(
        NativeRequest(
          "gen-alt",
          "img2img",
          Map(
            "output_dir" -> Str(outDir.toString),
            "preview_every" -> Number(0),
            "preview_method" -> Str("none"),
            "params" -> Obj(
              Map(
                "prompt" -> Str("tiny alt"),
                "width" -> Number(16),
                "height" -> Number(16),
                "steps" -> Number(2),
                "seed" -> Number(44),
                "img2img" -> Obj(
                  Map(
                    "init_image" -> Str(initImgPath.toString),
                    "denoising_strength" -> Number(BigDecimal("0.5"))
                  )
                ),
                "hooks" -> Obj(
                  Map(
                    "img2imgalt" -> Obj(
                      Map(
                        "enabled" -> Bool(true),
                        "override_prompt" -> Str("override alt"),
                        "decode_cfg_scale" -> Number(BigDecimal("1.2")),
                        "randomness" -> Number(BigDecimal("0.2")),
                        "inversion_steps" -> Number(2)
                      )
                    )
                  )
                )
              )
            )
          ),
          None
        ),
        ctx
      )
      val altImages = resAlt("images").asInstanceOf[Arr].values
      assertEquals(altImages.size, 1)
    finally deleteTree(dir)
  }
