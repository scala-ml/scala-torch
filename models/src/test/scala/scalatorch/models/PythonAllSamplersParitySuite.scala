package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Path
import scalatorch.{Shape, Tensor}
import scalatorch.models.worker.NativeJson
import scalatorch.models.worker.NativeJson.*

class PythonAllSamplersParitySuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(600, "s")
  private val autoSchedules = Map(
    "DPM++ 2M" -> "Karras",
    "DPM++ SDE" -> "Karras",
    "DPM++ 2M SDE" -> "Exponential",
    "DPM++ 2M SDE Heun" -> "Exponential",
    "DPM++ 2S a" -> "Karras",
    "DPM++ 3M SDE" -> "Exponential",
    "Euler a" -> "Uniform",
    "Euler" -> "Uniform",
    "LMS" -> "Uniform",
    "Heun" -> "Uniform",
    "DPM2" -> "Karras",
    "DPM2 a" -> "Karras",
    "DDIM" -> "Uniform",
    "UniPC" -> "Uniform",
    "LCM" -> "Uniform"
  )

  private def resolveSchedule(sampler: String, schedule: String): String =
    if schedule == "Automatic" then autoSchedules.getOrElse(sampler, "Uniform") else schedule

  private def runSamplerLatents(
      pipe: EulerImagePipeline,
      noise: DiffusionNoiseSchedule,
      request: DiffusionImageRequest,
      sampler: String,
      rawSchedule: String,
      steps: Int,
      source: Option[DiffusionImageSource] = None
  ): (Tensor[Float], Vector[String]) =
    val warnings = scala.collection.mutable.LinkedHashSet.empty[String]
    val requested = resolveSchedule(sampler, rawSchedule)
    val stepsOffset = 1 // tiny-sd-pipe scheduler_config.json has steps_offset=1
    val defaultSpacing = DiffusionTimestepSpacing.Leading

    def ancestralSchedule(count: Int, req: String): EulerSchedule =
      val spacing = req match
        case "Uniform" | "Normal" => DiffusionTimestepSpacing.Linspace
        case "Simple" | "SGM Uniform" => DiffusionTimestepSpacing.Trailing
        case "DDIM" => DiffusionTimestepSpacing.Leading
        case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
          warnings += s"schedule '$req' is not available for sampler 'Euler a'; using Karras"
          warnings += s"schedule '$req' is not supported by sampler 'Euler a'; using the sampler's default spacing"
          defaultSpacing
        case _ =>
          warnings += s"schedule '$req' is not supported by sampler 'Euler a'; using the sampler's default spacing"
          defaultSpacing
      noise.euler(count, spacing, if spacing == DiffusionTimestepSpacing.Leading then stepsOffset else 0)

    def ddimSchedule(count: Int, req: String): DdimSchedule =
      val spacing = req match
        case "Uniform" | "Normal" => DiffusionTimestepSpacing.Linspace
        case "Simple" | "SGM Uniform" => DiffusionTimestepSpacing.Trailing
        case "DDIM" => DiffusionTimestepSpacing.Leading
        case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
          warnings += s"schedule '$req' is not available for sampler 'DDIM'; using Karras"
          warnings += s"schedule '$req' is not supported by sampler 'DDIM'; using the sampler's default spacing"
          defaultSpacing
        case _ =>
          warnings += s"schedule '$req' is not supported by sampler 'DDIM'; using the sampler's default spacing"
          defaultSpacing
      DdimSchedule(noise, count, spacing, if spacing == DiffusionTimestepSpacing.Leading then stepsOffset else 0)

    def eulerSchedule(count: Int, req: String): EulerSchedule = req match
      case "Karras" => noise.karrasEuler(count)
      case "Exponential" | "Polyexponential" => noise.exponentialEuler(count)
      case "Normal" => noise.normalEuler(count)
      case "Simple" => noise.simpleEuler(count)
      case "SGM Uniform" => noise.sgmUniformEuler(count)
      case "Align Your Steps" => noise.alignYourStepsEuler(count, pipe.sdxlConditioning.nonEmpty)
      case "KL Optimal" => noise.klOptimalEuler(count)
      case "DDIM" => noise.ddimEuler(count)
      case "Beta" => noise.betaEuler(count)
      case _ => noise.nuiUniformEuler(count)

    def dpmSchedule(count: Int, req: String, samplerName: String): DpmSchedule = req match
      case "Karras" => DpmSchedule.karras(noise, count)
      case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
        warnings += s"schedule '$req' is not available for sampler '$samplerName'; using Karras"
        DpmSchedule.karras(noise, count)
      case "Exponential" => DpmSchedule.exponential(noise, count)
      case "Simple" | "SGM Uniform" => DpmSchedule.trailing(noise, count)
      case "DDIM" => DpmSchedule.leading(noise, count, stepsOffset)
      case "Beta" => DpmSchedule.beta(noise, count)
      case _ => DpmSchedule.uniform(noise, count)

    def dpmSinglestepSchedule(count: Int, req: String): DpmSchedule = req match
      case "Karras" => DpmSchedule.karras(noise, count)
      case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
        warnings += s"schedule '$req' is not available for sampler 'DPM++ 2S a'; using Karras"
        DpmSchedule.karras(noise, count)
      case "Exponential" => DpmSchedule.exponential(noise, count)
      case "Beta" => DpmSchedule.beta(noise, count)
      case _ =>
        warnings += s"schedule '$req' is not supported by sampler 'DPM++ 2S a'; using the sampler's default spacing"
        DpmSchedule.uniform(noise, count)

    def discreteEulerSchedule(count: Int, req: String, samplerName: String, roundKarrasTimesteps: Boolean = false): EulerSchedule =
      val (spacing, conversion) = req match
        case "Karras" => (DiffusionTimestepSpacing.Linspace, "karras")
        case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
          warnings += s"schedule '$req' is not available for sampler '$samplerName'; using Karras"
          (DiffusionTimestepSpacing.Linspace, "karras")
        case "Exponential" => (DiffusionTimestepSpacing.Linspace, "exponential")
        case "Beta" => (DiffusionTimestepSpacing.Linspace, "beta")
        case "Simple" | "SGM Uniform" => (DiffusionTimestepSpacing.Trailing, "none")
        case "DDIM" => (DiffusionTimestepSpacing.Leading, "none")
        case _ => (DiffusionTimestepSpacing.Linspace, "none")
      noise.discreteSchedule(count, spacing, if spacing == DiffusionTimestepSpacing.Leading then stepsOffset else 0, conversion, roundKarrasTimesteps)

    val out = sampler match
      case "DPM++ 2M SDE" | "DPM++ 2M SDE Heun" | "DPM++ 3M SDE" =>
        pipe.generateDpmMultistepSde(
          request,
          dpmSchedule(steps, requested, sampler),
          order = if sampler == "DPM++ 3M SDE" then 3 else 2,
          heunVariant = (sampler == "DPM++ 2M SDE Heun"),
          source = source,
          returnLatents = true
        )
      case "UniPC" =>
        pipe.generateUniPc(request, dpmSchedule(steps, requested, "UniPC"), source = source, returnLatents = true)
      case "LCM" =>
        requested match
          case "Polyexponential" | "Align Your Steps" | "KL Optimal" =>
            warnings += s"schedule '$requested' is not available for sampler 'LCM'; using Karras"
            warnings += s"schedule '$requested' is not supported by sampler 'LCM'; using the sampler's default spacing"
          case "Karras" | "Exponential" | "Beta" =>
            warnings += s"schedule '$requested' is not supported by sampler 'LCM'; using the sampler's default spacing"
          case _ => ()
        if requested != "Uniform" then warnings += "LCM ignores the schedule type"
        pipe.generateLcm(request, LcmSchedule(noise, steps), source = source, returnLatents = true)
      case "LMS" =>
        pipe.generateLms(request, discreteEulerSchedule(steps, requested, "LMS"), source = source, returnLatents = true)
      case "DPM2" | "DPM2 a" =>
        pipe.generateKdpm2(
          request,
          discreteEulerSchedule(steps, requested, sampler, roundKarrasTimesteps = true),
          ancestral = (sampler == "DPM2 a"),
          source = source,
          returnLatents = true
        )
      case "DPM++ 2S a" =>
        pipe.generateDpm2Sa(request, dpmSinglestepSchedule(steps, requested), source = source, returnLatents = true)
      case "DPM++ SDE" =>
        pipe.generateDpmSde(request, discreteEulerSchedule(steps, requested, "DPM++ SDE"), source = source, returnLatents = true)
      case "DDIM" =>
        pipe.generateDdim(request, ddimSchedule(steps, requested), clipSample = false, clipRange = 1f, eta = 0.0, source = source, returnLatents = true)
      case "DPM++ 2M" =>
        pipe.generateDpmLatents(request, dpmSchedule(steps, requested, "DPM++ 2M"), source = source)
      case "Euler a" =>
        pipe.generateAncestralLatents(request, ancestralSchedule(steps, requested), source = source)
      case "Heun" =>
        pipe.generateHeun(request, discreteEulerSchedule(steps, requested, "Heun"), source = source, returnLatents = true)
      case _ =>
        pipe.generateLatents(request, eulerSchedule(steps, requested), source = source)
    (out, warnings.toVector)

  test("All 14 diffusers samplers x 12 schedules in txt2img and img2img match Python sdfx to floating-point precision") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "public tiny model required")
    val root = Path.of(configured.get).getParent
    val stream = getClass.getResourceAsStream("/python-all-samplers-schedules-reference.json")
    assert(stream != null, "python-all-samplers-schedules-reference.json resource missing")
    val fixture = try NativeJson.parse(new String(stream.readAllBytes(), UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val txt2imgCases = fixture("txt2img").asInstanceOf[Obj].fields
    val img2imgCases = fixture("img2img").asInstanceOf[Obj].fields

    val pipe = new EulerImagePipeline(
      ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"), root.resolve("tokenizer/merges.txt")).withNuiFallbackNormalization,
      PretrainedClip.load(root.resolve("text_encoder")),
      PretrainedUNet.load(root.resolve("unet")),
      PretrainedVae.load(root.resolve("vae"))
    )
    val noise = DiffusionNoiseSchedule.linear(1000, 0.00085, 0.012, scaled = true)
    val req = DiffusionImageRequest(
      prompt = "a red cat",
      negativePrompt = "blurry",
      width = 16,
      height = 16,
      seed = 42L,
      guidanceScale = 7.5f,
      torchCpuNoise = true,
      trainingSigmas = noise.floatTrainingSigmas
    )

    val sampleOutDir = sys.props.get("scalatorch.parity.samples.dir").map(Path.of(_))
    sampleOutDir.foreach(d => java.nio.file.Files.createDirectories(d))
    val showcaseSamplers = Vector("Euler", "Euler a", "Heun", "DPM++ 2M", "DPM++ 2S a", "DPM++ 2M SDE", "DPM++ 3M SDE", "DDIM", "LMS", "DPM2", "DPM2 a", "UniPC", "LCM")

    var worstLatentMax = 0.0
    var worstLatentKey = ""
    var worstImgMax = 0.0
    var worstImgKey = ""
    val failures = scala.collection.mutable.ArrayBuffer.empty[String]

    for (key, rawVal) <- txt2imgCases.toSeq.sortBy(_._1) do
      val parts = key.split("\\|", 2)
      val sampler = parts(0)
      val schedule = parts(1)
      val row = rawVal.asInstanceOf[Obj].fields
      val expectedLatent = row("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
      val expectedImage = row("image").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
      val expectedWarns = row("warnings").asInstanceOf[Arr].values.map(_.asInstanceOf[Str].value)

      val (actualLatentTensor, actualWarns) = runSamplerLatents(pipe, noise, req, sampler, schedule, steps = 4)
      val actualLatent = actualLatentTensor.toFloatArray
      val latDiffs = actualLatent.zip(expectedLatent).map((a, b) => math.abs(a.toDouble - b.toDouble))
      val latMax = latDiffs.max
      val latMean = latDiffs.sum / latDiffs.length
      if latMax > worstLatentMax then
        worstLatentMax = latMax
        worstLatentKey = s"txt2img:$key"

      val decodedTensor = pipe.decodeLatents(actualLatentTensor)
      val decoded = decodedTensor.toFloatArray.map(v => math.max(0f, math.min(1f, v / 2f + 0.5f)))
      if schedule == "Automatic" && showcaseSamplers.contains(sampler) then
        sampleOutDir.foreach { d =>
          val safeName = sampler.replace(" ", "_").replace("+", "p")
          val scalaImg = DiffusionImageIO.toBufferedImage(decodedTensor)
          javax.imageio.ImageIO.write(scalaImg, "png", d.resolve(s"scala_${safeName}.png").toFile)
          val pyTensor = Tensor.fromFloats(expectedImage.map(v => v * 2f - 1f), Shape.from(Seq(1L, 3L, 16L, 16L)))
          val pyImg = DiffusionImageIO.toBufferedImage(pyTensor)
          javax.imageio.ImageIO.write(pyImg, "png", d.resolve(s"python_${safeName}.png").toFile)
        }
      val imgDiffs = decoded.zip(expectedImage).map((a, b) => math.abs(a.toDouble - b.toDouble))
      val imgMax = imgDiffs.max
      if imgMax > worstImgMax then
        worstImgMax = imgMax
        worstImgKey = s"txt2img:$key"

      if actualWarns != expectedWarns then
        val msg = s"txt2img $key warnings mismatch: actual=$actualWarns vs expected=$expectedWarns"
        println("FAIL: " + msg)
        failures += msg
      if latMax >= 1e-3 || !latMax.isFinite then
        val msg = f"txt2img $key latent max=$latMax%.6f mean=$latMean%.6f"
        println("FAIL: " + msg)
        failures += msg
      if imgMax >= 1e-3 || !imgMax.isFinite then
        val msg = f"txt2img $key image max=$imgMax%.6f"
        println("FAIL: " + msg)
        failures += msg

    // Img2Img cases
    val srcData = Array.tabulate(3 * 16 * 16)(i => (i % 256).toFloat / 255.0f * 2.0f - 1.0f)
    val srcTensor = Tensor.fromFloats(srcData, Shape.from(Seq(1L, 3L, 16L, 16L)))
    val imgSource = Some(DiffusionImageSource(srcTensor, 0.75))

    for (key, rawVal) <- img2imgCases.toSeq.sortBy(_._1) do
      val parts = key.split("\\|", 2)
      val sampler = parts(0)
      val schedule = parts(1)
      val row = rawVal.asInstanceOf[Obj].fields
      val expectedLatent = row("latent").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat).toArray
      val (actualLatentTensor, _) = runSamplerLatents(pipe, noise, req, sampler, schedule, steps = 4, source = imgSource)
      val actualLatent = actualLatentTensor.toFloatArray
      val latDiffs = actualLatent.zip(expectedLatent).map((a, b) => math.abs(a.toDouble - b.toDouble))
      val latMax = latDiffs.max
      val latMean = latDiffs.sum / latDiffs.length
      if latMax > worstLatentMax then
        worstLatentMax = latMax
        worstLatentKey = s"img2img:$key"
      if latMax >= 1e-3 || !latMax.isFinite then
        val msg = f"img2img $key latent max=$latMax%.6f mean=$latMean%.6f"
        println("FAIL: " + msg)
        failures += msg

    println(f"All-sampler Python parity summary: worst latent diff = $worstLatentMax%.6e ($worstLatentKey), worst image diff = $worstImgMax%.6e ($worstImgKey)")
    assert(failures.isEmpty, s"${failures.size} parity failures:\n" + failures.mkString("\n"))
  }
