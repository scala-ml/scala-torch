package scalatorch.models.worker

import munit.FunSuite
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*
import java.awt.image.BufferedImage
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO

class SagReferenceControlLLLiteFooocusSuite extends FunSuite:

  private def writeSafetensors(path: Path, tensors: Seq[(String, Seq[Long], Array[Float])]): Unit =
    var offset = 0L
    val entries = tensors.map { case (name, shape, values) =>
      val byteLen = values.length.toLong * 4L
      val start = offset
      val end = offset + byteLen
      offset = end
      s""""$name":{"dtype":"F32","shape":[${shape.mkString(",")}],"data_offsets":[$start,$end]}"""
    }
    val header = entries.mkString("{", ",", "}")
    val jsonBytes = header.getBytes(StandardCharsets.UTF_8)
    val pad = (8 - (jsonBytes.length % 8)) % 8
    val paddedLen = jsonBytes.length + pad
    val out = Files.newOutputStream(path)
    try
      val lenBuf = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN).putLong(paddedLen.toLong)
      out.write(lenBuf.array())
      out.write(jsonBytes)
      if pad > 0 then out.write(Array.fill[Byte](pad)(' '.toByte))
      tensors.foreach { case (_, _, values) =>
        val bb = java.nio.ByteBuffer.allocate(values.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(values)
        out.write(bb.array())
      }
    finally out.close()

  private def writeTestPng(path: Path, width: Int = 16, height: Int = 16): Unit =
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until height; x <- 0 until width do
      val r = (x * 16) & 0xff
      val g = (y * 16) & 0xff
      val b = ((x + y) * 8) & 0xff
      img.setRGB(x, y, (r << 16) | (g << 8) | b)
    ImageIO.write(img, "png", path.toFile)

  test("SAG: mid-block QK attention map capture, Gaussian blur degradation, and SAG guidance delta"):
    val store = new Sag.MapStore()
    val q = Tensor.fromFloats(
      Array(
        1f, 0f, 0f, 1f, 2f, 1f, -1f, 0f,
        0f, 1f, 1f, 0f, 1f, 2f, 0f, -1f
      ),
      Shape.from(Seq(1L, 2L, 4L, 2L))
    )
    val k = Tensor.fromFloats(
      Array(
        2f, 0f, 0f, 1f, 3f, 2f, -1f, 0f,
        0f, 2f, 1f, 0f, 2f, 3f, 0f, -1f
      ),
      Shape.from(Seq(1L, 2L, 4L, 2L))
    )
    Sag.withStore(Some(store)) {
      Sag.recordFromQK(q, k)
    }
    assert(store.midAttentionProbs.isDefined)
    val probs = store.midAttentionProbs.get
    assertEquals(probs.shape.dims, Vector(1L, 4L))
    val probSum = probs.toFloatArray.sum
    assert(math.abs(probSum - 1.0f) < 1e-4f, s"attention probabilities should sum to 1.0 (got $probSum)")

    val xt = Tensor.fromFloats(Array.tabulate(16)(i => (i % 5).toFloat), Shape.from(Seq(1L, 1L, 4L, 4L)))
    val eps = Tensor.fromFloats(Array.tabulate(16)(i => ((i % 3) * 0.2f)), Shape.from(Seq(1L, 1L, 4L, 4L)))
    val degraded = Sag.degradeLatent(xt, eps, 1.0, store.midAttentionProbs, 1.5)
    assertEquals(degraded.shape.dims, Vector(1L, 1L, 4L, 4L))
    assert(degraded.toFloatArray.forall(_.isFinite))
    assert(degraded.toFloatArray.zip(xt.toFloatArray).exists { case (a, b) => math.abs(a - b) > 1e-5f })

    val guided = Sag.applyGuidance(xt, eps, degraded, 0.75)
    assertEquals(guided.shape.dims, Vector(1L, 1L, 4L, 4L))
    assert(guided.toFloatArray.forall(_.isFinite))

  test("ReferenceOnly: token concatenation, unconditional style fidelity blending, and image normalization"):
    val dir = Files.createTempDirectory("ref-only-test")
    val imgPath = dir.resolve("ref.png")
    writeTestPng(imgPath, 16, 16)
    val normImg = ReferenceOnly.loadNormalizedImageTensor(imgPath.toString, 8, 8, "Reference-only")
    assertEquals(normImg.shape.dims, Vector(1L, 3L, 8L, 8L))
    assert(normImg.toFloatArray.forall(v => v.isFinite && v >= -1.0f && v <= 1.0f))

    val a = Tensor.fromFloats(Array(1f, 2f, 3f, 4f), Shape.from(Seq(1L, 2L, 2L)))
    val b = Tensor.fromFloats(Array(5f, 6f, 7f, 8f), Shape.from(Seq(1L, 2L, 2L)))
    val cat = ReferenceOnly.concatTokens(a, b)
    assertEquals(cat.shape.dims, Vector(1L, 4L, 2L))
    assertEquals(cat.toFloatArray.toVector, Vector(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))

    val blended = ReferenceOnly.blendUncond(a, b, 0.25)
    assertEquals(blended.toFloatArray.toVector, Vector(2f, 3f, 4f, 5f))

  test("Revision & IpAdapter: SDXL pooled conditioning injection and cross-attention context token augmentation"):
    val dir = Files.createTempDirectory("rev-ipa-test")
    val imgPath = dir.resolve("prompt.png")
    writeTestPng(imgPath, 16, 16)

    val cond = Tensor.fromFloats(Array.fill(77 * 8)(1.0f), Shape.from(Seq(1L, 77L, 8L)))
    val pooled = Tensor.fromFloats(Array.fill(8)(2.0f), Shape.from(Seq(1L, 8L)))
    val revCfg = RevisionConfig(enabled = true, image = imgPath.toString, strength = 1.0, ignorePrompt = false)
    val (revCond, revPool) = Revision.applyToSdxlConditioning(cond, pooled, revCfg)
    assertEquals(revCond.shape.dims, Vector(1L, 77L, 8L))
    assertEquals(revPool.shape.dims, Vector(1L, 8L))
    assert(revPool.toFloatArray.zip(pooled.toFloatArray).exists { case (x, y) => math.abs(x - y) > 1e-4f })

    val (_, revPoolIgnored) = Revision.applyToSdxlConditioning(cond, pooled, revCfg.copy(ignorePrompt = true))
    assert(revPoolIgnored.toFloatArray.zip(revPool.toFloatArray).forall { case (ig, full) => math.abs((full - ig) - 2.0f) < 1e-4f })

    val ipaModelPath = dir.resolve("ip-adapter.safetensors")
    writeSafetensors(ipaModelPath, Seq(("ip_adapter.proj.weight", Seq(4L, 4L), Array.fill(16)(0.1f))))
    val ipaCfg = IpAdapterConfig(enabled = true, model = ipaModelPath.toString, image = imgPath.toString, scale = 0.8)
    val loadedIpa = IpAdapter.load(ipaCfg, contextDim = 8)
    assertEquals(loadedIpa.imageTokens.shape.dims, Vector(1L, 4L, 8L))
    val augCtx = IpAdapter.augmentCrossAttentionContext(cond, loadedIpa)
    assertEquals(augCtx.shape.dims, Vector(1L, 81L, 8L))

  test("ControlLLLite & FooocusInpaint: attention input modulation and 9-channel convIn head residual"):
    val dir = Files.createTempDirectory("lllite-fooocus-test")
    val imgPath = dir.resolve("cond.png")
    writeTestPng(imgPath, 8, 8)
    val llliteModel = dir.resolve("lllite.safetensors")
    writeSafetensors(llliteModel, Seq(("lllite_unet_down_blocks_0.weight", Seq(4L, 4L), Array.fill(16)(0.1f))))

    val llCfg = ControlLLLiteConfig(enabled = true, model = llliteModel.toString, image = imgPath.toString, strength = 1.0, startPercent = 0.0, endPercent = 1.0)
    val (cImg, gain) = ControlLLLite.load(llCfg, 8, 8, None)
    val state = ControlLLLite.ActiveLLLite(llCfg, cImg, 0.5, gain)
    val tokensIn = Tensor.fromFloats(Array.fill(4 * 8)(1.0f), Shape.from(Seq(1L, 4L, 8L)))
    val modulated = ControlLLLite.modulateAttentionInput(tokensIn, state)
    assertEquals(modulated.shape.dims, Vector(1L, 4L, 8L))
    assert(modulated.toFloatArray.zip(tokensIn.toFloatArray).exists { case (a, b) => math.abs(a - b) > 1e-6f })

    val patchPath = dir.resolve("inpaint_v26.patch.safetensors")
    val headPath = dir.resolve("fooocus_inpaint_head.safetensors")
    writeSafetensors(patchPath, Seq(("patch", Seq(1L), Array(1.0f))))
    writeSafetensors(
      headPath,
      Seq(
        ("head.weight", Seq(320L, 9L, 3L, 3L), Array.fill(320 * 9 * 3 * 3)(0.01f)),
        ("head.bias", Seq(320L), Array.fill(320)(0.05f))
      )
    )
    val fiCfg = FooocusInpaintConfig(enabled = true, model = patchPath.toString, head = headPath.toString, startPercent = 0.0, endPercent = 1.0)
    val headWeights = FooocusInpaint.loadHead(fiCfg)
    val mask1 = Tensor.fromFloats(Array(1f, 0f, 1f, 0f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val lat4 = Tensor.fromFloats(Array.fill(16)(0.5f), Shape.from(Seq(1L, 4L, 2L, 2L)))
    val feed5 = FooocusInpaint.buildFeed5Channel(mask1, lat4)
    assertEquals(feed5.shape.dims, Vector(1L, 5L, 2L, 2L))
    val convInOut = Tensor.fromFloats(Array.fill(320 * 4)(1.0f), Shape.from(Seq(1L, 320L, 2L, 2L)))
    val fiOut = FooocusInpaint.applyConvInResidual(lat4, convInOut, FooocusInpaint.ActiveFooocusInpaint(fiCfg, headWeights, feed5, 0.5))
    assertEquals(fiOut.shape.dims, Vector(1L, 320L, 2L, 2L))
    assert(fiOut.toFloatArray.zip(convInOut.toFloatArray).exists { case (a, b) => math.abs(a - b) > 1e-4f })

  test("CompileAccel & NativeDiffusionBackend: advertises all Batch 17 features, validates compile compatibility, and runs end-to-end txt2img with SAG, ReferenceOnly, and IpAdapter"):
    val cCfg = CompileConfig(enabled = true, backend = "inductor", mode = "default", minBlockSize = 5, optimizationLevel = 3)
    CompileAccel.validateJob(cCfg, Set("compile", "model_keyword"), hasControlNet = false, hasLoras = false, hasComposableLora = false)
    intercept[IllegalArgumentException] {
      CompileAccel.validateJob(cCfg, Set("compile", "pag"), hasControlNet = false, hasLoras = false, hasComposableLora = false)
    }

    val configured = sys.props.get("scalatorch.clip.dir").orElse {
      Seq(
        Path.of(".superpowers/models/tiny-sd-pipe/text_encoder"),
        Path.of("../.superpowers/models/tiny-sd-pipe/text_encoder")
      ).find(Files.isDirectory(_)).map(_.toAbsolutePath.normalize().toString)
    }
    assume(configured.nonEmpty, "tiny model required")
    val modelDir = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("batch17-out")
    val imgPath = outDir.resolve("ref_input.png")
    writeTestPng(imgPath, 8, 8)
    val ipaPath = outDir.resolve("ip_test.safetensors")
    writeSafetensors(ipaPath, Seq(("ip_adapter.proj.weight", Seq(4L, 4L), Array.fill(16)(0.1f))))

    val backend = new NativeDiffusionBackend(modelDir, modelDir.resolve("euler-demo.json"))
    val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())
    val features = backend.features.toSet
    for expected <- Seq("sag", "reference", "revision", "ip_adapter", "controllllite", "fooocus_inpaint", "compile", "taesd", "spandrel", "vram_estimator") do
      assert(features.contains(expected), s"missing feature advertisement: $expected")

    backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)
    val genParams = Map(
      "output_dir" -> Str(outDir.toString),
      "params" -> Obj(Map(
        "prompt" -> Str("a red apple"),
        "negative_prompt" -> Str("blurry"),
        "width" -> Number(8),
        "height" -> Number(8),
        "steps" -> Number(2),
        "seed" -> Number(42),
        "cfg_scale" -> Number(5.0),
        "sampler_name" -> Str("Euler"),
        "hooks" -> Obj(Map(
          "sag" -> Obj(Map("enabled" -> Bool(true), "scale" -> Number(0.5), "blur_sigma" -> Number(1.5))),
          "reference" -> Obj(Map("enabled" -> Bool(true), "image" -> Str(imgPath.toString), "style_fidelity" -> Number(0.5), "attention_auto_machine_weight" -> Number(1.0))),
          "ip_adapter" -> Obj(Map("enabled" -> Bool(true), "model" -> Str(ipaPath.toString), "image" -> Str(imgPath.toString), "scale" -> Number(0.6)))
        ))
      ))
    )
    val resp = backend.execute(NativeRequest("gen17", "txt2img", genParams, None), ctx)
    val imgs = resp("images").asInstanceOf[Arr].values
    assertEquals(imgs.size, 1)
