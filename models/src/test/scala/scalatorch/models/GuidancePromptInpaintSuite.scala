package scalatorch.models.worker

import java.nio.file.{Files, Path}
import scalatorch.*
import scalatorch.io.{SafeDType, TensorInfo}
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class GuidancePromptInpaintSuite extends munit.FunSuite:
  test("PromptSchedule parses [from:to:when], [to:when], [from::when], and [a|b|c] matching A1111 semantics") {
    val s1 = PromptSchedule.schedule("a [cat:dog:0.5] in a park", 10)
    assertEquals(s1, Vector((5, "a cat in a park"), (10, "a dog in a park")))

    val s2 = PromptSchedule.schedule("a [dog:3] in a park", 10)
    assertEquals(s2, Vector((3, "a  in a park"), (10, "a dog in a park")))

    val s3 = PromptSchedule.schedule("a [cat::4] in a park", 10)
    assertEquals(s3, Vector((4, "a cat in a park"), (10, "a  in a park")))

    val s4 = PromptSchedule.schedule("[red|blue] flower", 4)
    assertEquals(
      s4,
      Vector(
        (1, "red flower"),
        (2, "blue flower"),
        (3, "red flower"),
        (4, "blue flower")
      )
    )
    assertEquals(PromptSchedule.textAt(s1, 1), "a cat in a park")
    assertEquals(PromptSchedule.textAt(s1, 5), "a cat in a park")
    assertEquals(PromptSchedule.textAt(s1, 6), "a dog in a park")
    assert(PromptSchedule.hasSchedule("a [cat:dog:0.5]"))
    assert(!PromptSchedule.hasSchedule("plain prompt [emphasis]"))
  }

  test("SingleFileSd2Config and SingleFileSdxlConfig accept 9-channel inpainting U-Nets") {
    def info(shape: Long*): TensorInfo =
      TensorInfo(SafeDType.F32, Shape.from(shape), 0L, shape.product * 4L)

    val sd2 = scala.collection.mutable.Map[String, TensorInfo](
      "model.diffusion_model.input_blocks.0.0.weight" -> info(320, 9, 3, 3),
      "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight" -> info(320, 1024),
      "model.diffusion_model.out.2.weight" -> info(4, 320, 3, 3),
      "cond_stage_model.model.token_embedding.weight" -> info(49408, 1024),
      "cond_stage_model.model.positional_embedding" -> info(77, 1024),
      "cond_stage_model.model.ln_final.weight" -> info(1024),
      "cond_stage_model.model.ln_final.bias" -> info(1024),
      "first_stage_model.encoder.conv_in.weight" -> info(128, 3, 3, 3),
      "first_stage_model.decoder.conv_out.weight" -> info(3, 128, 3, 3)
    )
    (0 to 22).foreach(i => sd2(s"cond_stage_model.model.transformer.resblocks.$i.attn.in_proj_weight") = info(3072, 1024))
    assertEquals(SingleFileSd2Config.unetConfigForTensors(sd2.toMap).inChannels, 9)

    val sdxl = scala.collection.mutable.Map[String, TensorInfo](
      "model.diffusion_model.input_blocks.0.0.weight" -> info(320, 9, 3, 3),
      "model.diffusion_model.label_emb.0.0.weight" -> info(1280, 2816),
      "model.diffusion_model.label_emb.0.2.weight" -> info(1280, 1280),
      "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight" -> info(640, 2048),
      "model.diffusion_model.out.2.weight" -> info(4, 320, 3, 3),
      "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight" -> info(49408, 768),
      "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight" -> info(77, 768),
      "conditioner.embedders.1.model.token_embedding.weight" -> info(49408, 1280),
      "conditioner.embedders.1.model.positional_embedding" -> info(77, 1280),
      "conditioner.embedders.1.model.ln_final.weight" -> info(1280),
      "conditioner.embedders.1.model.text_projection" -> info(1280, 1280),
      "first_stage_model.encoder.conv_in.weight" -> info(128, 3, 3, 3),
      "first_stage_model.decoder.conv_out.weight" -> info(3, 128, 3, 3)
    )
    (0 to 11).foreach(i => sdxl(s"conditioner.embedders.0.transformer.text_model.encoder.layers.$i.self_attn.q_proj.weight") = info(768, 768))
    (0 to 31).foreach(i => sdxl(s"conditioner.embedders.1.model.transformer.resblocks.$i.attn.in_proj_weight") = info(3840, 1280))
    assertEquals(SingleFileSdxlConfig.unetConfigForTensors(sdxl.toMap).inChannels, 9)
  }

  test("APG, Mahiro, PAG, and prompt schedule execute in NativeDiffusionBackend") {
    assert(math.abs(GuidanceHooks.apgMomentum(0.5, 0.18, 500.0) - (0.5 - 0.5 * math.pow(0.18, 4) * 500.0)) < 1e-12)
    val shape = Shape.from(Seq(1L, 4L, 2L, 2L))
    val cond = Tensor.fromFloats(Array.tabulate(16)(i => (i + 1) * 0.2f), shape)
    val uncond = Tensor.fromFloats(Array.tabulate(16)(i => (i - 4) * 0.1f), shape)
    val cfg = Tensor.fromFloats(Array.tabulate(16)(i => uncond.toFloatArray(i) + 7.0f * (cond.toFloatArray(i) - uncond.toFloatArray(i))), shape)
    val mah = GuidanceHooks.mahiro(cond, uncond, cfg, 7.0)
    assert(mah.toFloatArray.forall(_.isFinite))
    assert(!java.util.Arrays.equals(mah.toFloatArray, cfg.toFloatArray))

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
    val sched = eulerCfg.noise.nuiUniformEuler(2)
    val baseReq = DiffusionImageRequest(prompt = "a red cat", negativePrompt = "blurry", width = 16, height = 16, seed = 42, guidanceScale = 7.5f, trainingSigmas = eulerCfg.noise.floatTrainingSigmas)
    val baseLatents = pipeline.generateLatents(baseReq, sched).toFloatArray
    val schedLatents = pipeline.generateLatents(baseReq.copy(prompt = "a [red:(blue:5.0):1] cat"), sched).toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, schedLatents), "expected prompt schedule to alter final latents")

    val apgMahiroLatents = pipeline.generateLatents(baseReq.copy(apg = Some(ApgConfig(momentum = 0.5)), mahiro = true), sched).toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, apgMahiroLatents), "expected APG + Mahiro to alter final latents")

    val pagLatents = pipeline.generateLatents(baseReq.copy(pag = Some(PagConfig(scale = 3.0, layers = Vector("mid", "down", "up")))), sched).toFloatArray
    assert(!java.util.Arrays.equals(baseLatents, pagLatents), "expected PAG to alter final latents")

    val outDir = Files.createTempDirectory("native-guidance-hooks-")
    try
      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("apg"))
      assert(backend.features.contains("mahiro"))
      assert(backend.features.contains("pag"))
      backend.execute(NativeRequest("1", "load_model", Map("path" -> Str(root.toString)), None), context)

      val hooks = Map[String, NativeJson](
        "apg" -> Obj(Map("enabled" -> Bool(true), "momentum" -> Number(0.5))),
        "mahiro" -> Obj(Map("enabled" -> Bool(true))),
        "pag" -> Obj(Map("enabled" -> Bool(true), "scale" -> Number(3.0), "layers" -> Str("mid,down,up")))
      )
      val params = Map[String, NativeJson](
        "prompt" -> Str("a [red:blue:1] cat"),
        "negative_prompt" -> Str("blurry"),
        "width" -> Number(16),
        "height" -> Number(16),
        "steps" -> Number(2),
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
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
