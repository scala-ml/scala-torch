package scalatorch.models.worker

import java.nio.file.{Files, Path}
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class WeightGradientNegPipKohyaSuite extends munit.FunSuite:
  test("WeightGradient parses {tokens:interval:weights:mode} and builds per-step schedules") {
    assert(WeightGradient.hasGradient("a {cat::0.5-1.5:a} in a park"))
    assert(!WeightGradient.hasGradient("a (cat:1.2) in a park"))
    assert(math.abs(WeightGradient.curve("a", 0.5) - 0.5) < 1e-12)
    assert(math.abs(WeightGradient.curve("ei", 0.0) - 0.0) < 1e-12)
    assert(math.abs(WeightGradient.curve("eo", 1.0) - 1.0) < 1e-12)
    assertEquals(WeightGradient.render("a {cat::0.5-1.5:a} here", 1, 3), "a (cat:0.5) here")
    assertEquals(WeightGradient.render("a {cat::0.5-1.5:a} here", 2, 3), "a (cat:1) here")
    assertEquals(WeightGradient.render("a {cat::0.5-1.5:a} here", 3, 3), "a (cat:1.5) here")
    val sched = WeightGradient.schedule("a {cat::0.5-1.5:a} here", 3)
    assertEquals(sched, Vector((1, "a (cat:0.5) here"), (2, "a (cat:1) here"), (3, "a (cat:1.5) here")))
  }

  test("NegPip extracts negative weights and negates value projections for negative-sign tokens") {
    val (cleaned, terms) = NegPip.extract("a red (cat:-1.25) and (dog:1.1)")
    assertEquals(cleaned, "a red  and (dog:1.1)")
    assertEquals(terms, Vector(("cat", -1.25)))

    val v = Tensor.fromFloats(Array(1f, 2f, 3f, 4f, 5f, 6f), Shape.from(Seq(1L, 1L, 3L, 2L)))
    val signed = NegPip.applySigns(v, Array(1f, -1f, 1f))
    assertEquals(signed.toFloatArray.toSeq, Seq(1f, 2f, -3f, -4f, 5f, 6f))
  }

  test("KohyaHrFix and TokenDownsampling spatial/token transforms match Python semantics") {
    val cfg = KohyaHrFixConfig(
      blockNumber = 1,
      downscaleFactor = 2.0,
      startPercent = 0.0,
      endPercent = 0.5,
      downscaleAfterSkip = true,
      downscaleMethod = "bicubic",
      upscaleMethod = "bicubic"
    )
    assert(KohyaHrFix.isActive(cfg, 800f))
    assert(!KohyaHrFix.isActive(cfg, 200f))
    val layers = Seq(2, 2, 2, 2)
    assertEquals(KohyaHrFix.blockLocation(1, layers), Some((0, 0)))
    assertEquals(KohyaHrFix.blockLocation(3, layers), Some((0, 2)))
    assertEquals(KohyaHrFix.blockLocation(4, layers), Some((1, 0)))

    val x = Tensor.fromFloats(Array.tabulate(1 * 2 * 4 * 4)(_.toFloat), Shape.from(Seq(1L, 2L, 4L, 4L)))
    val shrunk = KohyaHrFix.shrinkIfActive(x, cfg, 800f)
    assertEquals(shrunk.shape.dims, Seq(1L, 2L, 2L, 2L))
    val restored = KohyaHrFix.interpolate2d(shrunk, 4, 4, cfg.upscaleMethod)
    assertEquals(restored.shape.dims, Seq(1L, 2L, 4L, 4L))

    val tds = TokenDownsamplingConfig(factor = 2.0, factor2 = 1.0, disableAfter = 0.6)
    assert(tds.enabledFor(4))
    val tokens = Tensor.fromFloats(Array.tabulate(1 * 16 * 4)(_.toFloat), Shape.from(Seq(1L, 16L, 4L)))
    val downTokens = TokenDownsampling.resampleTokens(tokens, 4, 4, 2.0)
    assertEquals(downTokens.shape.dims, Seq(1L, 4L, 4L))
  }

  test("ConditioningHighresFix resolves weights and blends pixels into masked regions") {
    assertEquals(ConditioningHighresFix.resolveWeight("disabled", Some(0.6)), None)
    assertEquals(ConditioningHighresFix.resolveWeight("empty", Some(0.6)), Some(1.0f))
    assertEquals(ConditioningHighresFix.resolveWeight("full", Some(0.6)), Some(0.0f))
    assertEquals(ConditioningHighresFix.resolveWeight("medium", Some(0.6)), Some(0.5f))
    assertEquals(ConditioningHighresFix.resolveWeight("low", Some(0.6)), Some(0.7f))

    val img = Tensor.fromFloats(Array(0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 1f, 1f, 1f, 1f), Shape.from(Seq(1L, 3L, 2L, 2L)))
    val mask = Tensor.fromFloats(Array(0f, 1f, 0.2f, 1f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val cond = ConditioningHighresFix.conditionPixels(img, mask, 0.5f)
    val out = cond.toFloatArray
    assert(math.abs(out(0) - 0.5f) < 1e-6f)
    assert(math.abs(out(1) - 0.25f) < 1e-6f)
  }

  test("WeightGradient, NegPip, KohyaHrFix, TokenDownsampling, and ConditioningHighresFix execute in EulerImagePipeline and NativeDiffusionBackend") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val pipeline = EulerImagePipeline(
      ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"), root.resolve("tokenizer/merges.txt")),
      PretrainedClip.load(root.resolve("text_encoder")),
      PretrainedUNet.load(root.resolve("unet")),
      PretrainedVae.load(root.resolve("vae"))
    )
    val eulerCfg = EulerConfig.read(root.resolve("euler-demo.json"))
    val layout = eulerCfg.noise.nuiUniformEuler(4)

    val baseReq = DiffusionImageRequest(
      prompt = "a cat in a park",
      negativePrompt = "blurry",
      width = 16,
      height = 16,
      seed = 42L,
      guidanceScale = 7.5f,
      trainingSigmas = eulerCfg.noise.floatTrainingSigmas
    )
    val baseLatents = pipeline.generateLatents(baseReq, layout).toFloatArray

    val wgPrompt = "a {cat::0.2-9.8:e} in a park"
    val wgLatents = pipeline
      .generateLatents(
        baseReq.copy(
          prompt = wgPrompt,
          weightGradient = true
        ),
        layout
      )
      .toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, wgLatents), "expected WeightGradient schedule to alter latents")

    val negPipLatents = pipeline
      .generateLatents(
        baseReq.copy(prompt = "a cat (dog:-4.5) in a park", negPip = true),
        layout
      )
      .toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, negPipLatents), "expected NegPiP to alter latents")

    val kohyaLatents = pipeline
      .generateLatents(
        baseReq.copy(
          kohyaHrFix = Some(
            KohyaHrFixConfig(
              blockNumber = 1,
              downscaleFactor = 2.0,
              startPercent = 0.0,
              endPercent = 1.0,
              downscaleAfterSkip = true,
              downscaleMethod = "bicubic",
              upscaleMethod = "bicubic"
            )
          )
        ),
        layout
      )
      .toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, kohyaLatents), "expected Kohya HRFix to alter latents")

    val tdsLatents = pipeline
      .generateLatents(
        baseReq.copy(
          tokenDownsampling = Some(TokenDownsamplingConfig(factor = 2.0, factor2 = 2.0, disableAfter = 1.0))
        ),
        layout
      )
      .toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, tdsLatents), "expected Token Downsampling to alter latents")

    val outDir = Files.createTempDirectory("native-wg-negpip-kohya-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("weight_gradient"))
      assert(backend.features.contains("negpip"))
      assert(backend.features.contains("kohya_hrfix"))
      assert(backend.features.contains("token_downsampling"))
      assert(backend.features.contains("conditioning_highres_fix"))

      backend.execute(NativeRequest("1", "load_model", Map("path" -> Str(root.toString)), None), context)

      val hooks = Map[String, NativeJson](
        "weight_gradient" -> Obj(Map("enabled" -> Bool(true))),
        "negpip" -> Obj(Map("enabled" -> Bool(true))),
        "kohya_hrfix" -> Obj(
          Map(
            "enabled" -> Bool(true),
            "block_number" -> Number(1),
            "downscale_factor" -> Number(2.0),
            "start_percent" -> Number(0.0),
            "end_percent" -> Number(0.75),
            "downscale_after_skip" -> Bool(true),
            "downscale_method" -> Str("bicubic"),
            "upscale_method" -> Str("bicubic")
          )
        ),
        "token_downsampling" -> Obj(
          Map(
            "enabled" -> Bool(true),
            "factor" -> Number(2.0),
            "factor2" -> Number(1.0),
            "disable_after" -> Number(0.75)
          )
        ),
        "conditioning_highres_fix" -> Obj(
          Map(
            "enabled" -> Bool(true),
            "mode" -> Str("medium")
          )
        )
      )
      val params = Map[String, NativeJson](
        "prompt" -> Str("a {cat::0.4-1.6:ei} (dog:-1.2) in a park"),
        "negative_prompt" -> Str("blurry"),
        "width" -> Number(16),
        "height" -> Number(16),
        "steps" -> Number(4),
        "seed" -> Number(42),
        "cfg_scale" -> Number(7.5),
        "sampler_name" -> Str("Euler"),
        "scheduler" -> Str("Uniform"),
        "hooks" -> Obj(hooks)
      )
      val req = Map[String, NativeJson](
        "params" -> Obj(params),
        "output_dir" -> Str(outDir.toString),
        "preview_every" -> Number(0),
        "preview_method" -> Str("none")
      )
      val res = backend.execute(NativeRequest("2", "txt2img", req, None), context)
      val p = Path.of(res("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assert(Files.readAllBytes(p).nonEmpty)
      val warnings = res("warnings").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)
      assert(warnings.exists(_.contains("Conditioning Highres Fix")))
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
