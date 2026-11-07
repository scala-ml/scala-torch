package scalatorch.models.worker

import java.nio.file.{Files, Path}
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class AdditionalSamplersSuite extends munit.FunSuite:
  test("LMS, DPM2, DPM2 a, DPM++ 2S a, DPM++ 2M SDE, DPM++ 3M SDE, DPM++ SDE, UniPC, and LCM step solvers produce finite updates") {
    val noise = DiffusionNoiseSchedule.linear(1000, 0.00085, 0.012, scaled = true)
    val eulerSched = noise.karrasEuler(4)
    val dpmSched = DpmSchedule.karras(noise, 4)
    val shape = Shape.from(Seq(1L, 4L, 2L, 2L))
    val x0 = Variable(Tensor.fromFloats(Array.tabulate(16)(i => (i - 8) * 0.1f), shape), requiresGrad = false)
    val eps = Variable(Tensor.fromFloats(Array.tabulate(16)(i => (i + 1) * 0.05f), shape), requiresGrad = false)
    val z = Variable(Tensor.fromFloats(Array.tabulate(16)(i => (i % 5 - 2) * 0.2f), shape), requiresGrad = false)

    // LMS
    val lms = LmsScheduler(eulerSched)
    var lmsCur = x0
    for i <- 0 until 4 do
      lmsCur = lms.step(i, eps, lmsCur).previous
    assert(lmsCur.data.toFloatArray.forall(_.isFinite))
    assert(!java.util.Arrays.equals(lmsCur.data.toFloatArray, x0.data.toFloatArray))

    // DPM2 & DPM2 a
    val dpm2 = Kdpm2Scheduler(eulerSched, ancestral = false)
    val p1 = dpm2.predict(0, eps, x0)
    assert(p1.requiresSecondEvaluation)
    val c1 = dpm2.correct(0, p1, eps, None).previous
    assert(c1.data.toFloatArray.forall(_.isFinite))

    val dpm2a = Kdpm2Scheduler(eulerSched, ancestral = true)
    val p1a = dpm2a.predict(0, eps, x0, Some(z))
    val c1a = dpm2a.correct(0, p1a, eps, Some(z)).previous
    assert(c1a.data.toFloatArray.forall(_.isFinite))
    assert(!java.util.Arrays.equals(c1.data.toFloatArray, c1a.data.toFloatArray))

    // DPM++ 2S a
    val dpm2sa = DpmPlusPlus2Sa(eulerSched)
    val p2s = dpm2sa.predict(0, eps, x0)
    val c2s = dpm2sa.correct(0, p2s, eps, z).previous
    assert(c2s.data.toFloatArray.forall(_.isFinite))

    // DPM++ SDE
    val dpmsde = DpmPlusPlusSde(eulerSched)
    val psde = dpmsde.predict(0, eps, x0, z)
    val csde = dpmsde.correct(0, psde, eps, z).previous
    assert(csde.data.toFloatArray.forall(_.isFinite))

    // DPM++ 2M SDE & Heun & 3M SDE
    val sde2m = DpmPlusPlusMultistepSde(dpmSched.sigmas, order = 2, heun = false)
    val sde2mHeun = DpmPlusPlusMultistepSde(dpmSched.sigmas, order = 2, heun = true)
    val sde3m = DpmPlusPlusMultistepSde(dpmSched.sigmas, order = 3, heun = false)
    var cur2m = x0; var cur2mH = x0; var cur3m = x0
    for i <- 0 until 4 do
      cur2m = sde2m.step(i, eps, cur2m, z).previous
      cur2mH = sde2mHeun.step(i, eps, cur2mH, z).previous
      cur3m = sde3m.step(i, eps, cur3m, z).previous
    assert(cur2m.data.toFloatArray.forall(_.isFinite))
    assert(cur2mH.data.toFloatArray.forall(_.isFinite))
    assert(cur3m.data.toFloatArray.forall(_.isFinite))
    assert(!java.util.Arrays.equals(cur2m.data.toFloatArray, cur2mH.data.toFloatArray))

    // UniPC
    val unipc = UniPcScheduler(dpmSched.sigmas)
    var curUnipc = x0
    for i <- 0 until 4 do
      curUnipc = unipc.step(i, eps, curUnipc).previous
    assert(curUnipc.data.toFloatArray.forall(_.isFinite))

    // LCM
    val lcmSched = LcmSchedule(noise, 4)
    val lcm = LcmScheduler(lcmSched, DiffusionPrediction.Epsilon)
    var curLcm = x0
    for i <- 0 until 4 do
      curLcm = lcm.step(i, eps, curLcm, if i < 3 then Some(z) else None).previous
    assert(curLcm.data.toFloatArray.forall(_.isFinite))
  }

  test("NativeDiffusionBackend advertises all 15 NUI samplers in list_samplers and executes them in txt2img") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("native-samplers-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      val listed = backend.execute(NativeRequest("1", "list_samplers", Map.empty, None), context)
      val samplerRows = listed("samplers").asInstanceOf[Arr].values.map(_.asInstanceOf[Obj].fields)
      val names = samplerRows.map(_("name").asInstanceOf[Str].value).toSet
      val expected = Set(
        "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 2S a", "DPM++ 3M SDE",
        "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DDIM", "UniPC", "LCM"
      )
      assertEquals(names, expected)
      assert(samplerRows.forall(_("supported").asInstanceOf[Bool].value))

      backend.execute(NativeRequest("2", "load_model", Map("path" -> Str(root.toString)), None), context)

      for samplerName <- Vector("LMS", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 3M SDE", "UniPC", "LCM") do
        val params = Map[String, NativeJson](
          "prompt" -> Str("a red cat"),
          "negative_prompt" -> Str("blurry"),
          "width" -> Number(16),
          "height" -> Number(16),
          "steps" -> Number(2),
          "seed" -> Number(42),
          "cfg_scale" -> Number(7.5),
          "sampler_name" -> Str(samplerName),
          "scheduler" -> Str("Automatic")
        )
        val req = Map[String, NativeJson](
          "params" -> Obj(params),
          "output_dir" -> Str(outDir.toString),
          "preview_every" -> Number(0),
          "preview_method" -> Str("none")
        )
        val res = backend.execute(NativeRequest("3", "txt2img", req, None), context)
        val p = Path.of(res("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        assert(Files.readAllBytes(p).nonEmpty, s"empty output for sampler $samplerName")
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
