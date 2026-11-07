package scalatorch.models.worker

import munit.FunSuite
import scalatorch.{Shape, Tensor}
import scalatorch.models.*
import java.nio.file.{Files, Path}

class LlulAgentIncantationsChgSuite extends FunSuite:
  import NativeJson.*

  test("LlulConfig regionBox, upscale, downscale, slerp, and mixRegion"):
    val cfg = LlulConfig(weight = 0.5, multiply = 1, startSteps = 1, maxSteps = 5, intp = "slerp")
    assertEquals(cfg.multiplier, 2)
    val box = Llul.regionBox(8, 8, 0.25, 0.25, 2)
    assertEquals(box, Some((2, 6, 2, 6)))
    val grid = Tensor.fromFloats(Array.tabulate(1 * 4 * 8 * 8)(i => math.sin(i.toDouble * 0.2).toFloat), Shape.from(Seq(1L, 4L, 8L, 8L)))
    val cut = Llul.cutRegion(grid, cfg, 0.25, 0.25)
    assert(cut.isDefined)
    assertEquals(cut.get.shape, grid.shape)
    val extra = Tensor.fromFloats(grid.toFloatArray.map(_ * 2.0f), grid.shape)
    val mixed = Llul.mixRegion(grid, extra, cfg, 0.25, 0.25)
    assertEquals(mixed.shape, grid.shape)
    assert(mixed.toFloatArray.exists(v => v.isFinite))
    assert(mixed.toFloatArray.toSeq != grid.toFloatArray.toSeq)

  test("AgentAttention planMerge, merge, and unmerge preserve shape and merge similar tokens"):
    val cfg = AgentAttentionConfig(sx = 2, sy = 2, ratio = 0.25, agentRatio = 0.75)
    val x = Tensor.fromFloats(Array.tabulate(2 * 16 * 8)(i => math.cos(i.toDouble * 0.13).toFloat), Shape.from(Seq(2L, 16L, 8L)))
    val plan = AgentAttention.planMerge(x, 4, 4, cfg.sx, cfg.sy, 4, 12, Some(42L), useFp32 = false)
    assertEquals(plan.tokens, 16)
    val (features, agentsOpt) = AgentAttention.merge(plan, x)
    assertEquals(features.shape, Shape.from(Seq(2L, 12L, 8L)))
    assert(agentsOpt.isDefined)
    assertEquals(agentsOpt.get.shape, Shape.from(Seq(2L, 4L, 8L)))
    val restored = AgentAttention.unmerge(plan, features)
    assertEquals(restored.shape, x.shape)
    assert(restored.toFloatArray.forall(_.isFinite))

  test("Incantations CfgInterval 17 schedules, segBlurQuery, and T2I-0 CBS"):
    val cfg = IncantationsConfig(
      segActive = true,
      segBlurSigma = 2.0,
      cfgIntervalEnable = true,
      cfgIntervalSchedule = "Cosine",
      cfgIntervalLow = 0.1,
      cfgIntervalHigh = 50.0,
      t2i0Active = true,
      t2i0CorrectionStrength = 0.3
    )
    for schedName <- Incantations.Schedules do
      val v = Incantations.scheduleValue(schedName, 2, 10, 7.0)
      assert(v.isFinite && v > 0.0)
    val scheduledCfg = Incantations.cfgIntervalValue(cfg, 2, 10, 7.0)
    assert(scheduledCfg.isFinite && scheduledCfg > 1.0f)

    val q = Tensor.fromFloats(Array.tabulate(2 * 2 * 16 * 4)(i => (i % 7).toFloat), Shape.from(Seq(2L, 2L, 16L, 4L)))
    val blurred = Incantations.segBlurQuery(q, 2.0, 64, 64)
    assertEquals(blurred.shape, q.shape)
    assert(blurred.toFloatArray.toSeq != q.toFloatArray.toSeq)

    val cond = Tensor.fromFloats(Array.tabulate(1 * 10 * 8)(i => math.sin(i.toDouble * 0.3).toFloat), Shape.from(Seq(1L, 10L, 8L)))
    val corrected = Incantations.cbs(cond, 0.5, 2, 0.4, Vector(1, 2), tokenCount = 8)
    assertEquals(corrected.shape, cond.shape)
    assert(corrected.toFloatArray.toSeq != cond.toFloatArray.toSeq)

  test("CharacteristicGuidance stretch, splitBasis, projectBasis, and State"):
    val cfg = CharacteristicGuidanceConfig(regIni = 6.0, regRange = 2.0, ite = 3, noiseBase = 2, res = -2.0)
    assert(cfg.effectiveRegIni > 6.0)
    val g = Tensor.fromFloats(Array.tabulate(1 * 4 * 4 * 4)(i => (i - 32).toFloat * 0.05f), Shape.from(Seq(1L, 4L, 4L, 4L)))
    val basis = CharacteristicGuidance.splitBasis(g, 2)
    assertEquals(basis.length, 1 * 4 * 16 * 2)
    val ac = DiffusionNoiseSchedule.fromBetas(Vector.tabulate(1000)(i => 0.00085 + (0.012 - 0.00085) * i / 999.0)).alphasCumprod
    val st = new CharacteristicGuidance.State(cfg, 5, ac)
    val shifted = st.stepCorrection(
      0,
      500f,
      7.0f,
      g.shape,
      (uIn, cIn) => (uIn, Tensor.fromFloats(cIn.toFloatArray.map(_ * 0.9f), cIn.shape)),
      g,
      DiffusionPrediction.Epsilon
    )
    assert(shifted.isDefined)
    assertEquals(shifted.get._1.shape, g.shape)
    assertEquals(shifted.get._2.shape, g.shape)

  test("NativeDiffusionBackend advertises and executes llul, agent_attention, incantations, and characteristic_guidance"):
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("llul-agent-inc-chg")
    val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
    val feats = backend.features.toSet
    assert(feats.contains("llul"))
    assert(feats.contains("agent_attention"))
    assert(feats.contains("incantations"))
    assert(feats.contains("characteristic_guidance"))

    val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())

    backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)

    val reqParams = Map[String, NativeJson](
      "output_dir" -> Str(outDir.toString),
      "preview_every" -> Number(0),
      "params" -> Obj(
        Map(
          "prompt" -> Str("a scenic mountain"),
          "negative_prompt" -> Str("blurry"),
          "width" -> Number(64),
          "height" -> Number(64),
          "steps" -> Number(2),
          "seed" -> Number(42),
          "cfg_scale" -> Number(5),
          "sampler_name" -> Str("Euler"),
          "hooks" -> Obj(
            Map(
              "llul" -> Obj(
                Map(
                  "enabled" -> Bool(true),
                  "weight" -> Number(BigDecimal("0.2")),
                  "multiply" -> Number(1),
                  "start_steps" -> Number(1),
                  "max_steps" -> Number(2)
                )
              ),
              "agent_attention" -> Obj(
                Map(
                  "enabled" -> Bool(true),
                  "sx" -> Number(2),
                  "sy" -> Number(2),
                  "ratio" -> Number(BigDecimal("0.25")),
                  "agent_ratio" -> Number(BigDecimal("0.75"))
                )
              ),
              "incantations" -> Obj(
                Map(
                  "enabled" -> Bool(true),
                  "seg_active" -> Bool(true),
                  "seg_blur_sigma" -> Number(2),
                  "cfg_interval_enable" -> Bool(true),
                  "cfg_interval_schedule" -> Str("Cosine"),
                  "t2i0_active" -> Bool(true),
                  "t2i0_correction_strength" -> Number(BigDecimal("0.2"))
                )
              ),
              "characteristic_guidance" -> Obj(
                Map(
                  "enabled" -> Bool(true),
                  "ite" -> Number(2),
                  "res" -> Number(-1)
                )
              )
            )
          )
        )
      )
    )
    val res = backend.execute(NativeRequest("gen", "txt2img", reqParams, None), ctx)
    val images = res("images").asInstanceOf[Arr].values
    assertEquals(images.size, 1)
