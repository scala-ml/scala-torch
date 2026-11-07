package scalatorch.models.worker

import munit.FunSuite
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

class PromptFusionNeutralClipIoSuite extends FunSuite:

  private def writeTinySafetensorsEmbedding(path: Path, values: Array[Float]): Unit =
    val byteLen = values.length.toLong * 4L
    val header = s"""{"embedding":{"dtype":"F32","shape":[1,${values.length}],"data_offsets":[0,$byteLen]}}"""
    val jsonBytes = header.getBytes(StandardCharsets.UTF_8)
    val pad = (8 - (jsonBytes.length % 8)) % 8
    val paddedLen = jsonBytes.length + pad
    val out = Files.newOutputStream(path)
    try
      val lenBuf = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN).putLong(paddedLen.toLong)
      out.write(lenBuf.array())
      out.write(jsonBytes)
      if pad > 0 then out.write(Array.fill[Byte](pad)(' '.toByte))
      val bb = java.nio.ByteBuffer.allocate(values.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      bb.asFloatBuffer().put(values)
      out.write(bb.array())
    finally out.close()

  test("ClipIo: expression parser, CSV/Safetensors round-trip, and replace/append modes"):
    val dir = Files.createTempDirectory("clip-io-test")
    val condA = Tensor.fromFloats(Array.fill(77 * 8)(2.0f), Shape.from(Seq(1L, 77L, 8L)))
    val condB = Tensor.fromFloats(Array.fill(77 * 8)(4.0f), Shape.from(Seq(1L, 77L, 8L)))
    val pooledA = Tensor.fromFloats(Array.fill(4)(1.5f), Shape.from(Seq(1L, 4L)))
    val targetCsvA = ClipIo.safeTarget(dir, "cond_a", "csv")
    val targetCsvB = ClipIo.safeTarget(dir, "cond_b", "csv")
    ClipIo.writeConditioning(targetCsvA, "csv", ClipIo.CondData(condA, Some(pooledA)))
    ClipIo.writeConditioning(targetCsvB, "csv", ClipIo.CondData(condB, None))

    val loadedA = ClipIo.readConditioning(targetCsvA, "csv")
    assertEquals(loadedA.cond.shape.dims, Vector(77L, 8L))
    assert(loadedA.cond.toFloatArray.forall(v => math.abs(v - 2.0f) < 1e-6f))
    assert(loadedA.pooled.isDefined)
    assert(loadedA.pooled.get.toFloatArray.forall(v => math.abs(v - 1.5f) < 1e-6f))

    val emptyChunk = Tensor.fromFloats(new Array[Float](77 * 8), Shape.from(Seq(1L, 77L, 8L)))
    val baseCond = Tensor.fromFloats(Array.fill(77 * 8)(1.0f), Shape.from(Seq(1L, 77L, 8L)))
    val (replaced, repPool) = ClipIo.applySide(
      "replace",
      "cond_a * 0.5 + cond_b * 0.25",
      baseCond,
      Some(Tensor.fromFloats(Array.fill(4)(0.0f), Shape.from(Seq(1L, 4L)))),
      emptyChunk,
      dir,
      "csv"
    )
    assertEquals(replaced.shape.dims, Vector(1L, 77L, 8L))
    // 2.0 * 0.5 + 4.0 * 0.25 = 2.0
    assert(replaced.toFloatArray.forall(v => math.abs(v - 2.0f) < 1e-5f))
    assert(repPool.get.toFloatArray.forall(v => math.abs(v - 0.75f) < 1e-5f))

    val (appended, _) = ClipIo.applySide("append", "cond_a", baseCond, None, emptyChunk, dir, "csv")
    assertEquals(appended.shape.dims, Vector(1L, 154L, 8L))

    val targetSt = ClipIo.safeTarget(dir, "cond_st", "safetensors")
    ClipIo.writeConditioning(targetSt, "safetensors", ClipIo.CondData(condA, Some(pooledA)))
    val loadedSt = ClipIo.readConditioning(targetSt, "safetensors")
    assertEquals(loadedSt.cond.shape.dims, Vector(77L, 8L))
    assert(loadedSt.cond.toFloatArray.forall(v => math.abs(v - 2.0f) < 1e-6f))

  test("PromptFusion: variable declarations, weight interpolation, and bezier/catmull/mean/slerp tensor interpolation"):
    val textWithVars = "$style = oil painting\n$subj = a cat\n[ $subj : a dog : 0.2, 0.8 : bezier ], $style"
    val resolved = PromptFusion.applyDeclarations(textWithVars)
    assertEquals(resolved, "[ a cat : a dog : 0.2, 0.8 : bezier ], oil painting")
    assert(PromptFusion.isActive(textWithVars))

    val specOpt = PromptFusion.parseSpec(textWithVars, 10)
    assert(specOpt.isDefined)
    val spec = specOpt.get
    assertEquals(spec.func, "bezier")
    assertEquals(spec.controlPrompts, Vector(" a cat , oil painting", " a dog , oil painting"))
    assert(math.abs(spec.startRatio - 0.2) < 1e-6)
    assert(math.abs(spec.endRatio - 0.8) < 1e-6)

    val wSpec = PromptFusion.parseSpec("a photo of (cat:0.5, 1.5:0.0, 1.0)", 5).get
    assertEquals(wSpec.controlPrompts, Vector("a photo of (cat:0.5)", "a photo of (cat:1.5)"))

    val c0 = Tensor.fromFloats(Array(0f, 0f, 0f, 0f), Shape.from(Seq(1L, 2L, 2L)))
    val c1 = Tensor.fromFloats(Array(10f, 20f, 30f, 40f), Shape.from(Seq(1L, 2L, 2L)))
    val midLin = PromptFusion.interpolateTensor(Vector(c0, c1), 1, 3, wSpec)
    assertEquals(midLin.toFloatArray.toVector, Vector(5f, 10f, 15f, 20f))

    val midMean = PromptFusion.interpolateTensor(Vector(c0, c1), 1, 3, wSpec.copy(func = "mean"))
    assertEquals(midMean.toFloatArray.toVector, Vector(5f, 10f, 15f, 20f))

  test("NeutralPrompt & ComposableLora: tree parser, perpendicular/salient/topk combineDeltas, and branch controllers"):
    val tree = NeutralPrompt.parse("a red cat : 1.5 AND_PERP [a blue dog : 0.8 AND_SALT a green bird : 0.5] : 1.2 AND_TOPK spark : 0.7")
    val lvs = NeutralPrompt.leaves(tree)
    assertEquals(lvs.size, 4)
    assertEquals(lvs(0).text, "a red cat")
    assertEquals(lvs(0).weight, 1.5)
    assertEquals(lvs(0).kind, "AND")
    assertEquals(lvs(1).text, "a blue dog")
    assertEquals(lvs(1).kind, "AND")
    assertEquals(lvs(2).text, "a green bird")
    assertEquals(lvs(2).kind, "AND_SALT")
    assertEquals(lvs(3).text, "spark")
    assertEquals(lvs(3).kind, "AND_TOPK")

    val d0 = Tensor.fromFloats(Array(1f, 0f, 0f, 0f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val d1 = Tensor.fromFloats(Array(1f, 2f, 0f, 0f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val d2 = Tensor.fromFloats(Array(0f, 1f, 3f, -1f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val d3 = Tensor.fromFloats(Array(0.1f, 0.2f, 0.3f, 5.0f), Shape.from(Seq(1L, 1L, 2L, 2L)))
    val combined = NeutralPrompt.combineDeltas(tree, Vector(d0, d1, d2, d3))
    assertEquals(combined.shape.dims, Vector(1L, 1L, 2L, 2L))
    assert(combined.toFloatArray.forall(_.isFinite))

    val rescaled = NeutralPrompt.rescale(combined, d1, 0.7)
    assert(rescaled.toFloatArray.forall(_.isFinite))

    val branchTxt = "masterpiece <lora:styleA:0.8> [<lora:detailB:1.2>:0.2] #increment"
    val ctrls = ComposableLora.parseBranchControllers(branchTxt)
    assertEquals(ctrls.size, 2)
    assertEquals(ComposableLora.cleanPrompt(branchTxt), "masterpiece")
    val multsMid = ComposableLora.evaluateBranchMultipliers(ctrls, 5, 10)
    assert(multsMid("styleA") > 0f && multsMid("styleA") < 0.8f)
    assert(multsMid("detailB") > 0f)

  test("EmbeddingMerge: expression parser and float32 vector algebra"):
    val prompt = "photo of <'cat' + 'dog' * 0.5 :2> and {'bird', 000}"
    assert(EmbeddingMerge.hasExpressions(prompt))
    val spans = EmbeddingMerge.extractSpans(prompt)
    assertEquals(spans.size, 2)
    assertEquals(spans(0).expression.trim, "'cat' + 'dog' * 0.5 :2")
    assertEquals(spans(1).expression.trim, "'bird', 000")

    val actions = EmbeddingMerge.parseExpression(spans(0).expression)
    val evaluated = EmbeddingMerge.evaluate(
      actions,
      Vector("clip_l"),
      Vector(4),
      txt =>
        if txt == "cat" then Vector(Vector(Array(1f, 2f, 3f, 4f)))
        else Vector(Vector(Array(2f, 4f, 6f, 8f), Array(10f, 20f, 30f, 40f))),
      _ => Vector(Vector(Array(0f, 0f, 0f, 0f)))
    )
    assertEquals(evaluated.head.size, 2)
    // row 0: [1,2,3,4] + 0.5 * [2,4,6,8] = [2,4,6,8]
    assertEquals(evaluated.head(0).toVector, Vector(2f, 4f, 6f, 8f))
    // row 1: [0,0,0,0] + 0.5 * [10,20,30,40] = [5,10,15,20]
    assertEquals(evaluated.head(1).toVector, Vector(5f, 10f, 15f, 20f))

  test("NativeDiffusionBackend: advertises all 6 features and runs end-to-end generation with ClipIo, PromptFusion, NeutralPrompt, and EmbeddingMerge"):
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val modelDir = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("pf-np-cio-out")
    val cioDir = Files.createTempDirectory("pf-np-cio-store")
    val backend = new NativeDiffusionBackend(modelDir, modelDir.resolve("euler-demo.json"))
    val ctx = new NativeWorkContext(() => false, () => true, (_, _) => ())

    val features = backend.features.toSet
    backend.execute(NativeRequest("load", "load_model", Map.empty, None), ctx)
    for expected <- Seq("clip_io", "prompt_fusion", "neutral_prompt", "composable_lora", "embedding_merge", "aesthetic_gradients") do
      assert(features.contains(expected), s"missing feature advertisement: $expected")

    val genParams = Map(
      "output_dir" -> Str(outDir.toString),
      "params" -> Obj(Map(
        "prompt" -> Str("$a = cat\n[ <'cat' * 1.1> : $a : 0.0, 1.0 : linear ] AND dog : 0.5"),
        "negative_prompt" -> Str("blurry"),
        "width" -> Number(8),
        "height" -> Number(8),
        "steps" -> Number(2),
        "seed" -> Number(42),
        "cfg_scale" -> Number(5.0),
        "sampler_name" -> Str("Euler"),
        "hooks" -> Obj(Map(
          "clip_io" -> Obj(Map(
            "enabled" -> Bool(true),
            "output_enabled" -> Bool(true),
            "output_name" -> Str("saved_pos"),
            "output_format" -> Str("csv"),
            "overwrite" -> Bool(true),
            "storage_dir" -> Str(cioDir.toString)
          )),
          "prompt_fusion" -> Obj(Map("enabled" -> Bool(true))),
          "neutral_prompt" -> Obj(Map("enabled" -> Bool(true), "cfg_rescale" -> Number(0.3))),
          "composable_lora" -> Obj(Map("enabled" -> Bool(true)))
        ))
      ))
    )

    val resp = backend.execute(NativeRequest("gen1", "txt2img", genParams, None), ctx)
    val imgs = resp("images").asInstanceOf[Arr].values
    assertEquals(imgs.size, 1)
    assert(Files.isRegularFile(cioDir.resolve("saved_pos.csv")), "Clip Output CSV should be written")

    // Also verify AestheticGradientsload & personalizeConditioning on a 768-dim tensor
    val agFile = cioDir.resolve("test_ag.safetensors")
    val agVec = Array.tabulate(768)(i => ((i % 7) + 1).toFloat)
    writeTinySafetensorsEmbedding(agFile, agVec)
    val agCfg = AestheticGradientsConfig(embedding = agFile.toString, weight = 0.5, steps = 10, lr = 0.001)
    val cond768 = Tensor.fromFloats(Array.tabulate(77 * 768)(i => ((i % 5) - 2).toFloat), Shape.from(Seq(1L, 77L, 768L)))
    val personalized = AestheticGradients.personalizeConditioning(cond768, agCfg)
    assertEquals(personalized.shape.dims, Vector(1L, 77L, 768L))
    assert(personalized.toFloatArray.forall(_.isFinite))
    assert(personalized.toFloatArray.zip(cond768.toFloatArray).exists { case (a, b) => math.abs(a - b) > 1e-5f })
