package scalatorch.models.worker

import munit.FunSuite
import scalatorch.{Shape, Tensor}
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO

class SemanticGuidanceSonarHiresTweaksSuite extends FunSuite:

  private def noopContext(): NativeWorkContext =
    new NativeWorkContext(() => false, () => true, (_, _) => ())

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        stream.sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
      finally stream.close()

  test("SemanticGuidance parses concepts, computes zScore/editDirection, and shifts SD1/SDXL conditioning") {
    val parsed = SemanticGuidance.parseConcepts("sunglasses, (smiling:2.0)", "(hat:1.5)")
    assertEquals(parsed, Vector(("sunglasses", 1.0), ("smiling", 2.0), ("hat", -1.5)))

    val z = SemanticGuidance.zScore(0.05)
    assert(math.abs(z - 1.6448536) < 1e-4, s"zScore(0.05) should be ~1.64485, got $z")

    val cond = Tensor.fromFloats(Array(1f, 2f, 3f, 4f), Shape.from(Seq(1L, 2L, 2L)))
    val uncond = Tensor.fromFloats(Array(0f, 0f, 0f, 0f), Shape.from(Seq(1L, 2L, 2L)))
    val empty = Tensor.fromFloats(Array(0.1f, 0.2f), Shape.from(Seq(1L, 1L, 2L)))
    val concept = Tensor.fromFloats(Array(0.5f, 1.5f, -0.5f, 0.5f), Shape.from(Seq(1L, 2L, 2L)))

    val sgCfg = SemanticGuidanceConfig(prompt = "(smiling:2.0)", warmup = 0, editGuidanceScale = 5.0, tailPercentageThreshold = 0.5)
    val (shifted, shiftedAdded) = SemanticGuidance.applyToConditioning(
      cond, uncond, empty,
      Vector(EncodedSemanticConcept(concept, None, 2.0)),
      sgCfg, None, None, 256
    )
    assertEquals(shifted.shape, cond.shape)
    assert(shiftedAdded.isEmpty)
    val diff = shifted.toFloatArray.zip(cond.toFloatArray).map((a, b) => math.abs(a - b)).max
    assert(diff > 1e-4f, s"SemanticGuidance should shift conditioning, max diff=$diff")
  }

  test("Sonar computes momentum step, reference guidance, and upscale target") {
    val refLatent = Sonar.normalizeReference(Tensor.fromFloats(Array.fill(256)(0.5f), Shape.from(Seq(1L, 4L, 8L, 8L))))
    assertEquals(refLatent.shape, Shape.from(Seq(1L, 4L, 8L, 8L)))
    assert(refLatent.toFloatArray.forall(_.isFinite))

    val cfg = SonarConfig(
      sampler = "Naive", momentum = 0.8, momentumHist = 0.5,
      momentumHistInit = "zero", momentumSign = "pos",
      refMeth = "linear", refHgf = 0.2, refMinStep = 0.0, refMaxStep = 1.0,
      upscale = true, upscaler = "Lanczos", upscaleRatio = 1.5
    )
    val stepper = new SonarStepper(cfg, steps = 4, seed = 42L, ancestral = false, reference = Some(refLatent))
    val x = Tensor.fromFloats(Array.fill(256)(1f), Shape.from(Seq(1L, 4L, 8L, 8L)))
    val eps = Tensor.fromFloats(Array.fill(256)(0.1f), Shape.from(Seq(1L, 4L, 8L, 8L)))
    val x1 = stepper.step(x, eps, 10.0, 5.0, DiffusionPrediction.Epsilon)
    assertEquals(x1.shape, x.shape)
    assert(x1.toFloatArray.forall(_.isFinite))

    assertEquals(Sonar.upscaleTarget(64, 64, cfg), Some((96, 96, false)))
  }

  test("HiresFixTweaks computes seedPlan and resizedNoise") {
    val plan = HiresFixTweaks.seedPlan(
      batchCount = 2, seedEnabled = true, seed = 105L, extras = true,
      subseed = 99L, strength = 0.25, resizeWidth = 64, resizeHeight = 64,
      baseSeeds = Vector(100L), baseSubseeds = Vector(10L), baseStrength = 0.0
    )
    assertEquals(plan.seedFor(100L), 105L)
    assertEquals(plan.subseedFor(10L), 99L)
    assertEquals(plan.subseedStrength(0.0), 0.25)
    assertEquals(plan.repeat, 2)
    assertEquals(plan.resizeFrom, Some((64, 64)))

    val noise = HiresFixTweaks.resizedNoise(
      targetShape = Shape.from(Seq(1L, 4L, 8L, 8L)),
      seed = 42L, subseed = Some(99L), strength = 0.2,
      resizeWidth = 32, resizeHeight = 32, factor = 8,
      draw = (s, sh) => Tensor.fromFloats(Array.fill(sh.numel.toInt)(s.toFloat * 0.01f), sh)
    )
    assertEquals(noise.shape, Shape.from(Seq(1L, 4L, 8L, 8L)))
    assert(noise.toFloatArray.forall(_.isFinite))
  }

  test("NativeDiffusionBackend advertises and executes semantic_guidance, sonar, hires_fix_tweaks, pnginfo_injection, and prevent_interrupt") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("native-sg-sonar-hr-out-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      for feat <- Seq("semantic_guidance", "sonar", "hires_fix_tweaks", "pnginfo_injection", "prevent_interrupt") do
        assert(backend.features.contains(feat), s"missing advertised feature: $feat")

      backend.execute(NativeRequest("load", "load_model", Map.empty, None), noopContext())

      val refPath = outDir.resolve("ref.png")
      val refImg = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 16; x <- 0 until 16 do refImg.setRGB(x, y, 0x4080C0)
      ImageIO.write(refImg, "png", refPath.toFile)

      // 1. semantic_guidance + sonar (Naive + ref_img + upscale)
      val res1 = backend.execute(NativeRequest("gen-sg-sonar", "txt2img", Map(
        "output_dir" -> Str(outDir.toString),
        "preview_every" -> Number(0),
        "preview_method" -> Str("none"),
        "params" -> Obj(Map(
          "prompt" -> Str("tiny test"),
          "negative_prompt" -> Str(""),
          "width" -> Number(16),
          "height" -> Number(16),
          "steps" -> Number(1),
          "seed" -> Number(42),
          "cfg_scale" -> Number(4.0),
          "sampler_name" -> Str("Euler"),
          "scheduler" -> Str("Uniform"),
          "hooks" -> Obj(Map(
            "semantic_guidance" -> Obj(Map(
              "enabled" -> Bool(true),
              "prompt" -> Str("(bright:1.5)"),
              "negative_prompt" -> Str("(dark:1.0)"),
              "edit_guidance_scale" -> Number(4.0),
              "tail_percentage_threshold" -> Number(BigDecimal("0.2")),
              "warmup" -> Number(0)
            )),
            "sonar" -> Obj(Map(
              "enabled" -> Bool(true),
              "sampler" -> Str("Naive"),
              "momentum" -> Number(BigDecimal("0.9")),
              "momentum_hist" -> Number(BigDecimal("0.75")),
              "momentum_sign" -> Str("pos"),
              "momentum_hist_init" -> Str("zero"),
              "ref_img" -> Str(refPath.toString),
              "ref_hgf" -> Number(BigDecimal("0.2")),
              "upscale" -> Bool(true),
              "upscaler" -> Str("Nearest"),
              "upscale_ratio" -> Number(BigDecimal("1.5"))
            )),
            "pnginfo_injection" -> Obj(Map("enabled" -> Bool(true), "mode" -> Str("all"))),
            "prevent_interrupt" -> Obj(Map("enabled" -> Bool(true), "mode" -> Str("confirm")))
          ))
        ))
      ), None), noopContext())

      val img1 = res1("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals(img1("width"), Number(24))
      assertEquals(img1("height"), Number(24))

      // 2. hires + hires_fix_tweaks
      val res2 = backend.execute(NativeRequest("gen-hr-tweaks", "txt2img", Map(
        "output_dir" -> Str(outDir.toString),
        "preview_every" -> Number(0),
        "preview_method" -> Str("none"),
        "params" -> Obj(Map(
          "prompt" -> Str("tiny test"),
          "negative_prompt" -> Str(""),
          "width" -> Number(16),
          "height" -> Number(16),
          "steps" -> Number(1),
          "seed" -> Number(100),
          "cfg_scale" -> Number(4.0),
          "sampler_name" -> Str("Euler"),
          "scheduler" -> Str("Uniform"),
          "hires" -> Obj(Map(
            "enabled" -> Bool(true),
            "upscaler" -> Str("Latent"),
            "scale" -> Number(BigDecimal("1.5")),
            "second_pass_steps" -> Number(1),
            "denoising_strength" -> Number(BigDecimal("0.5"))
          )),
          "hooks" -> Obj(Map(
            "hires_fix_tweaks" -> Obj(Map(
              "enabled" -> Bool(true),
              "seed_offset" -> Number(7),
              "subseed" -> Number(200),
              "subseed_strength" -> Number(BigDecimal("0.2")),
              "repeat" -> Number(2),
              "seed_resize_from_w" -> Number(16),
              "seed_resize_from_h" -> Number(16)
            ))
          ))
        ))
      ), None), noopContext())

      val img2 = res2("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
      assertEquals(img2("width"), Number(64))
      assertEquals(img2("height"), Number(64))
      assertEquals(img2("hires_seed"), Number(107))
      assertEquals(img2("hires_subseed"), Number(200))
      assertEquals(img2("hires_repeat"), Number(2))
    finally
      deleteRecursively(outDir)
  }
