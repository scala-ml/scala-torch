package scalatorch.models.worker

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scalatorch.*
import scalatorch.models.*
import scalatorch.models.worker.NativeJson.*

class MetadataKeywordModelSamplingSuite extends munit.FunSuite:
  private def writeDummySafetensors(path: Path, metadata: Map[String, String], floats: Array[Float]): Unit =
    val tensorBytes = floats.length * 4
    val metaEntries = metadata.map { (k, v) => s""""$k":"${v.replace("\"", "\\\"")}"""" }.mkString(",")
    val headerJson = s"""{"__metadata__":{$metaEntries},"weight":{"dtype":"F32","shape":[${floats.length}],"data_offsets":[0,$tensorBytes]}}"""
    val headerBytes = headerJson.getBytes(UTF_8)
    val buf = ByteBuffer.allocate(8 + headerBytes.length + tensorBytes).order(ByteOrder.LITTLE_ENDIAN)
    buf.putLong(headerBytes.length.toLong)
    buf.put(headerBytes)
    floats.foreach(buf.putFloat)
    Files.write(path, buf.array())

  test("metadata_utils reads and writes safetensors metadata without corrupting tensor bytes") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val tempDir = Files.createTempDirectory("native-metadata-utils-")
    try
      val sfPath = tempDir.resolve("sample.safetensors")
      writeDummySafetensors(sfPath, Map("ss_output_name" -> "my_lora", "nested" -> """{"a":1}"""), Array(1.5f, -2.5f, 3.25f, 4.0f))

      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("metadata_utils"))

      val readRes = backend.execute(
        NativeRequest("1", "metadata_utils", Map("operation" -> Str("read"), "path" -> Str(sfPath.toString)), None),
        context
      )
      assertEquals(readRes("tensor_bytes"), Number(16))
      val metaObj = readRes("metadata").asInstanceOf[Obj].fields
      assertEquals(metaObj("ss_output_name"), Str("my_lora"))
      val token = readRes("source_token").asInstanceOf[Str].value
      assert(token.nonEmpty)

      val writeRes = backend.execute(
        NativeRequest(
          "2",
          "metadata_utils",
          Map(
            "operation" -> Str("write"),
            "path" -> Str(sfPath.toString),
            "new_name" -> Str("sample_edited"),
            "metadata_json" -> Str("""{"ss_output_name":"edited_lora","tag_frequency":{"cat":5}}"""),
            "source_token" -> Str(token)
          ),
          None
        ),
        context
      )
      val outPath = Path.of(writeRes("path").asInstanceOf[Str].value)
      assert(Files.isRegularFile(outPath))
      assertEquals(writeRes("tensor_bytes"), Number(16))

      val reread = backend.execute(
        NativeRequest("3", "metadata_utils", Map("operation" -> Str("read"), "path" -> Str(outPath.toString)), None),
        context
      )
      val updatedMeta = reread("metadata").asInstanceOf[Obj].fields
      assertEquals(updatedMeta("ss_output_name"), Str("edited_lora"))
      assertEquals(updatedMeta("tag_frequency"), Str("""{"cat":5}"""))
    finally
      val files = Files.walk(tempDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }

  test("model_keyword_plan plans prompts and extracts batch LoRA activations") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val tempDir = Files.createTempDirectory("native-mk-plan-")
    try
      val ckptFile = tempDir.resolve("custom_model.safetensors")
      // legacy_hash reads at 0x100000; create a 1.1 MB file
      val bytes = new Array[Byte](0x110000)
      java.util.Arrays.fill(bytes, 42.toByte)
      Files.write(ckptFile, bytes)
      val hash = KeywordMappings.legacyHash(ckptFile, () => ())
      Files.writeString(tempDir.resolve("custom-mappings.txt"), s"$hash, anime style | detailed, custom_model.safetensors\n", UTF_8)

      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      val res = backend.execute(
        NativeRequest(
          "1",
          "model_keyword_plan",
          Map(
            "checkpoint" -> Str(ckptFile.toString),
            "mapping_dir" -> Str(tempDir.toString),
            "preview" -> Str("1girl <lora:test_lora:0.8>"),
            "prompts" -> Arr(Vector(Str("1girl <lora:test_lora:0.8>"), Str("1girl <lora:test_lora:0.8>"))),
            "batch_size" -> Number(2),
            "choice_seed" -> Number(42),
            "options" -> Obj(Map("model_choice" -> Str("keyword1"), "placement" -> Str("keyword prompt")))
          ),
          None
        ),
        context
      )
      assert(res("preview").asInstanceOf[Str].value.contains("anime style"))
      val prompts = res("prompts").asInstanceOf[Arr].values.collect { case Str(s) => s }
      assertEquals(prompts.size, 2)
      assert(prompts.forall(_.contains("anime style")))
      val batches = res("batches").asInstanceOf[Arr].values
      assertEquals(batches.size, 1)
      val batch0 = batches.head.asInstanceOf[Obj].fields
      val adapters = batch0("adapters").asInstanceOf[Arr].values
      assertEquals(adapters.size, 1)
      assertEquals(adapters.head.asInstanceOf[Obj].fields("name"), Str("test_lora"))
    finally
      val files = Files.walk(tempDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }

  test("txt2img supports non-empty styles, system_info_fields, and model_sampling hook with zsnr") {
    val configured = sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty, "tiny model required")
    val root = Path.of(configured.get).getParent
    val outDir = Files.createTempDirectory("native-styles-sysinfo-ms-")
    try
      val zsnrCfg = EulerConfig.fromJson(
        """{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","prediction_type":"v_prediction","rescale_betas_zero_snr":true}"""
      )
      assertEquals(zsnrCfg.prediction, DiffusionPrediction.VPrediction)
      assert(zsnrCfg.noise.sigmas.last > 20.0)

      val backend = new NativeDiffusionBackend(root, root.resolve("euler-demo.json"))
      val context = new NativeWorkContext(() => false, () => true, (_, _) => ())
      assert(backend.features.contains("model_sampling"))
      backend.execute(NativeRequest("1", "load_model", Map("path" -> Str(root.toString)), None), context)

      def runGen(hooks: Map[String, NativeJson]): Map[String, NativeJson] =
        val params = Map[String, NativeJson](
          "prompt" -> Str("a test cat"),
          "negative_prompt" -> Str("blurry"),
          "width" -> Number(16),
          "height" -> Number(16),
          "steps" -> Number(2),
          "seed" -> Number(42),
          "cfg_scale" -> Number(7.5),
          "sampler_name" -> Str("Euler"),
          "scheduler" -> Str("Uniform"),
          "styles" -> Arr(Vector(Str("cinematic"), Str("photorealistic"))),
          "hooks" -> Obj(hooks)
        )
        val req = Map[String, NativeJson](
          "params" -> Obj(params),
          "output_dir" -> Str(outDir.toString),
          "preview_every" -> Number(0),
          "preview_method" -> Str("none"),
          "system_info_fields" -> Arr(Vector(Str("version.app"), Str("version.engine"), Str("runtime.backend")))
        )
        backend.execute(NativeRequest("2", "txt2img", req, None), context)

      val baseRes = runGen(Map.empty)
      val sysInfo = baseRes("system_info").asInstanceOf[Obj].fields
      assertEquals(sysInfo("version.app"), Str("stablediffusion-nui"))
      assertEquals(sysInfo("version.engine"), Str("scala-torch-0.1.0"))

      val msRes = runGen(Map("model_sampling" -> Obj(Map("enabled" -> Bool(true), "sampling" -> Str("v_prediction_zsnr"), "zsnr" -> Bool(true)))))
      val basePath = Path.of(baseRes("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val msPath = Path.of(msRes("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      assert(!java.util.Arrays.equals(Files.readAllBytes(basePath), Files.readAllBytes(msPath)))
    finally
      val files = Files.walk(outDir)
      try files.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => { Files.delete(p); () })
      finally files.close()
  }
