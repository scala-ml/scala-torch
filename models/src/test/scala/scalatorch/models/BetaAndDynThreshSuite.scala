package scalatorch.models.worker

import java.nio.file.Files
import java.util.concurrent.atomic.AtomicBoolean
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class BetaAndDynThreshSuite extends munit.FunSuite:
  test("Beta Euler and DPM schedules match NUI sigmas.py beta schedule and deduplicate repeated steps") {
    val noise = DiffusionNoiseSchedule.linear(1000, 0.00085, 0.012, scaled = true)
    val schedule = noise.betaEuler(10, alpha = 0.6, beta = 0.6)
    // Verify against exact Beta(0.6, 0.6) quantiles on [1.0, 0.9, ..., 0.1] * 999:
    val expectedIndices = Vector(999, 960, 896, 811, 709, 500, 290, 188, 103, 39)
    // Check that each index corresponds to the rounded inverse of RegularizedBeta(x, 0.6, 0.6) = 1 - i/10
    for i <- 1 until 10 do
      val target = 1.0 - i.toDouble / 10.0
      val idx = math.rint(schedule.timesteps(i).toDouble).toInt
      val cdfLo = org.apache.commons.numbers.gamma.RegularizedBeta.value(math.max(0.0, (idx - 0.5) / 999.0), 0.6, 0.6)
      val cdfHi = org.apache.commons.numbers.gamma.RegularizedBeta.value(math.min(1.0, (idx + 0.5) / 999.0), 0.6, 0.6)
      assert(target >= cdfLo - 1e-9 && target <= cdfHi + 1e-9, s"step $i: target=$target not in [$cdfLo, $cdfHi] for idx=$idx")
    val actualIndices = schedule.timesteps.map(t => math.rint(t.toDouble).toInt)
    val expectedSigmas = actualIndices.map(i => noise.floatTrainingSigmas(i).toDouble) :+ 0.0
    assertEquals(schedule.sigmas.size, 11)
    schedule.sigmas.zip(expectedSigmas).foreach((a, b) => assertEqualsDouble(a, b, 1e-5))
    assertEquals(schedule.nominalInferenceSteps, Some(10))

    val dpm = DpmSchedule.beta(noise, 10, alpha = 0.6, beta = 0.6)
    assertEquals(dpm.timesteps, actualIndices.map(_.toFloat))
    dpm.sigmas.zip(expectedSigmas).foreach((a, b) => assertEqualsDouble(a, b, 1e-5))

    val shortNoise = DiffusionNoiseSchedule.linear(4, 0.001, 0.02)
    val deduplicated = shortNoise.betaEuler(10)
    assert(deduplicated.timesteps.size <= 4)
    assertEquals(deduplicated.nominalInferenceSteps, Some(10))

    intercept[IllegalArgumentException](noise.betaEuler(0))
    intercept[IllegalArgumentException](noise.betaEuler(10, alpha = 0.0))
    intercept[IllegalArgumentException](noise.betaEuler(10, beta = -1.0))
  }

  test("DynamicThresholding.interpretScale matches all 12 NUI schedule modes") {
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Constant", 1.0, 0.25, 2.0), 7.0, 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Linear Down", 1.0, 0.25, 2.0), 1.0 + 6.0 * 0.75, 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Half Cosine Down", 1.0, 0.25, 2.0), 1.0 + 6.0 * math.cos(0.25), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Cosine Down", 1.0, 0.25, 2.0), 1.0 + 6.0 * math.cos(0.25 * 1.5707), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Linear Up", 1.0, 0.25, 2.0), 1.0 + 6.0 * 0.25, 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Half Cosine Up", 1.0, 0.25, 2.0), 1.0 + 6.0 * (1.0 - math.cos(0.25)), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Cosine Up", 1.0, 0.25, 2.0), 1.0 + 6.0 * (1.0 - math.cos(0.25 * 1.5707)), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Power Up", 1.0, 0.25, 2.0), 1.0 + 6.0 * math.pow(0.25, 2.0), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Power Down", 1.0, 0.25, 2.0), 1.0 + 6.0 * (1.0 - math.pow(0.25, 2.0)), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Linear Repeating", 1.0, 0.1, 2.0), 1.0 + 6.0 * ((0.5 - 0.2) * 2.0), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Cosine Repeating", 1.0, 0.25, 2.0), 1.0 + 6.0 * (math.cos(0.25 * 6.28318 * 2.0) * 0.5 + 0.5), 1e-9)
    assertEqualsDouble(DynamicThresholding.interpretScale(7.0, "Sawtooth", 1.0, 0.3, 2.0), 1.0 + 6.0 * ((0.3 * 2.0) % 1.0), 1e-9)
  }

  test("DynamicThresholding.dynthresh clamps extreme CFG excursions and preserves x0 roundtrip in CFG") {
    val shape = Shape.from(Seq(1L, 2L, 2L, 2L))
    val uncond = Tensor.fromFloats(Array(0.0f, 0.1f, -0.1f, 0.05f, -0.05f, 0.2f, -0.2f, 0.0f), shape)
    val cond = Tensor.fromFloats(Array(0.5f, 1.2f, -1.5f, 0.8f, -0.4f, 2.0f, -2.1f, 0.3f), shape)
    val defaultCfg = DynamicThresholdingConfig(mimicScale = 3.0, thresholdPercentile = 0.95)
    val dtMeanAd = DynamicThresholding.dynthresh(cond, uncond, cfgScale = 12.0, frac = 0.5, defaultCfg)
    val rawCfg = uncond.toFloatArray.zip(cond.toFloatArray).map((u, c) => u + 12.0f * (c - u))
    val dtMax = dtMeanAd.toFloatArray.map(math.abs).max
    val rawMax = rawCfg.map(math.abs).max
    assert(dtMax < rawMax, s"expected dynamic thresholding to reduce peak magnitude ($dtMax < $rawMax)")

    for
      sep <- Seq(true, false)
      vm <- Seq("AD", "STD")
      sp <- Seq("MEAN", "ZERO")
    do
      val cfg = DynamicThresholdingConfig(
        mimicScale = 4.0,
        thresholdPercentile = 0.9,
        separateFeatureChannels = sep,
        variabilityMeasure = vm,
        scalingStartpoint = sp,
        interpolatePhi = 0.8
      )
      val out = DynamicThresholding.dynthresh(cond, uncond, cfgScale = 10.0, frac = 0.25, cfg)
      assertEquals(out.shape, shape)
      assert(out.toFloatArray.forall(_.isFinite))

    val noise = DiffusionNoiseSchedule.linear(1000, 0.00085, 0.012, scaled = true)
    val sample = Tensor.fromFloats(Array.fill(8)(0.25f), shape)
    for pred <- Seq(DiffusionPrediction.Epsilon, DiffusionPrediction.VPrediction) do
      val cNew = defaultCfg.rewriteConditional(
        sample = sample,
        uncond = uncond,
        cond = cond,
        timestep = 500.0f,
        guidanceScale = 12.0f,
        prediction = pred,
        sigmas = noise.floatTrainingSigmas
      )
      assertEquals(cNew.shape, shape)
      assert(cNew.toFloatArray.forall(_.isFinite))
  }

  test("NativeDiffusionBackend advertises Beta schedule and dynamic_thresholding hook and applies them in txt2img") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = java.nio.file.Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("native-beta-dynthresh-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("dynamic_thresholding"))
      val samplers = backend.execute(NativeRequest("0", "list_samplers", Map.empty, None), context)
      val schedulers = samplers("schedulers").asInstanceOf[Arr].values.collect { case Str(s) => s }
      assert(schedulers.contains("Beta"))

      backend.execute(NativeRequest("1", "load_model", Map("path" -> Str(root.toString)), None), context)

      def runGen(hooks: Map[String, NativeJson]): Array[Byte] =
        val params = Map[String, NativeJson](
          "prompt" -> Str("a test cat"),
          "negative_prompt" -> Str("blurry"),
          "width" -> Number(16),
          "height" -> Number(16),
          "steps" -> Number(3),
          "seed" -> Number(42),
          "cfg_scale" -> Number(9.0),
          "sampler_name" -> Str("Euler"),
          "scheduler" -> Str("Beta"),
          "hooks" -> Obj(hooks)
        )
        val req = Map[String, NativeJson](
          "params" -> Obj(params),
          "output_dir" -> Str(outDir.toString),
          "preview_every" -> Number(0),
          "preview_method" -> Str("none")
        )
        val res = backend.execute(NativeRequest("2", "txt2img", req, None), context)
        assertEquals(res("scheduler"), Str("Beta"))
        val imgObj = res("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields
        val imgPath = java.nio.file.Path.of(imgObj("path").asInstanceOf[Str].value)
        Files.readAllBytes(imgPath)

      val baselineBytes = runGen(Map.empty)
      val dtBytes = runGen(Map("dynamic_thresholding" -> Obj(Map(
        "enabled" -> Bool(true),
        "mimic_scale" -> Number(3.0),
        "threshold_percentile" -> Number(BigDecimal("0.90"))
      ))))
      assert(!java.util.Arrays.equals(baselineBytes, dtBytes), "expected Dynamic Thresholding to alter generated image")
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[java.nio.file.Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
