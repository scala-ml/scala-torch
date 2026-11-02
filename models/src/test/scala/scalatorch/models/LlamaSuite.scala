package scalatorch.models

import scalatorch.*
import scalatorch.nn.Module
import java.nio.file.Path
import java.nio.file.Files

class LlamaSuite extends munit.FunSuite:
  private val configJson = """{"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,"bos_token_id":1,"eos_token_id":2} """
  private def tokens = Tensor.fromLongs(Array(1L,3L,4L,2L), Shape.from(Seq(2L,2L)))
  test("strict supported config and architecture validation") {
    val c = LlamaConfig.fromJson(configJson)
    assertEquals(c.headDim, 2)
    assertEquals(c.numKeyValueHeads, 1)
    for extra <- Seq("\"rope_scaling\":{}", "\"attention_bias\":true", "\"hidden_act\":\"relu\"", "\"sliding_window\":4", "\"head_dim\":4", "\"partial_rotary_factor\":0.5") do
      val text = configJson.trim.dropRight(1) + "," + extra + "}"
      intercept[IllegalArgumentException](LlamaConfig.fromJson(text))
    intercept[IllegalArgumentException](LlamaConfig.fromJson(configJson.replace("\"hidden_size\":4", "\"hidden_size\":4.0")))
    intercept[IllegalArgumentException](LlamaConfig.fromJson(configJson.replace("\"hidden_size\":4", "\"hidden_size\":4,\"hidden_size\":4")))
    intercept[IllegalArgumentException](LlamaConfig.fromJson(configJson.replace("\"num_attention_heads\":2", "\"num_attention_heads\":3")))
  }
  test("composed Llama forward is differentiable and inference detached") {
    val model = LlamaModel.init(LlamaConfig.fromJson(configJson), seed = 7)
    val output = model.forward(tokens)
    assertEquals(output.data.shape.dims, Vector(2L,2L,8L))
    assert(output.data.toFloatArray.forall(_.isFinite))
    output.sum().backward()
    assert(Module.parameters(model).forall(_.grad.nonEmpty))
    assert(model.logits(tokens).toFloatArray.forall(_.isFinite))
    assertEquals(model.namedWeights.map(_._1).distinct.size, 12)
  }
  test("tied embeddings share parameter identity and limits reject before allocation") {
    val c = LlamaConfig.fromJson(configJson).copy(tieWordEmbeddings = true)
    val model = LlamaModel.init(c, seed = 1)
    assert(model.embedding.weight eq model.lmHead.weight)
    assertEquals(Module.parameters(model).size, 11)
    intercept[IllegalArgumentException](LlamaModel.init(c, 1, limits = LlamaLimits(maxParameters = 1)))
    val bounded = LlamaModel.init(c, 1, limits = LlamaLimits(maxLogitsElements = 8))
    intercept[IllegalArgumentException](bounded.logits(tokens))
    intercept[IllegalArgumentException](model.generateIds(Vector(1L), 20))
    assertEquals(model.generateIds(Vector(1L), 0), Vector(1L))
  }
  test("published tiny Llama logits match independent Candle reference") {
    val directory = sys.props.get("scalatorch.llama.dir")
    assume(directory.nonEmpty, "set scalatorch.llama.dir for public checkpoint gate")
    def sha(name: String): String = java.security.MessageDigest.getInstance("SHA-256")
      .digest(Files.readAllBytes(Path.of(directory.get).resolve(name))).map(b => f"${b & 255}%02x").mkString
    assertEquals(sha("config.json"), "5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382")
    assertEquals(sha("model.safetensors"), "49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992")
    val model = LlamaModel.load(Path.of(directory.get))
    val input = Tensor.fromLongs(Array(1L,42L,17L,321L), Shape.from(Seq(1L,4L)))
    val logits = model.logits(input).toFloatArray
    val stream = getClass.getResourceAsStream("/llama/tiny-candle-f32.bin")
    assert(stream != null)
    val bytes = try stream.readAllBytes() finally stream.close()
    val referenceSha = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
    assertEquals(referenceSha, "9c94f848e6203198bd0d469cb529d4130aca4f8c7d53a44d8ceca917212cf2c4")
    val buffer = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    assertEquals(logits.length * 4, bytes.length)
    var maxError = 0.0
    var i = 0
    while i < logits.length do
      val expected = buffer.getFloat()
      val error = math.abs(logits(i).toDouble - expected)
      maxError = math.max(maxError, error)
      assert(error <= 0.00001 + 0.00001 * math.abs(expected), s"logit $i difference=$error")
      i += 1
    println(s"public Llama full-vocab logits=${logits.length} maxAbsError=$maxError")
    val next = model.generateIds(Vector(1L,42L,17L,321L), 1).last
    val last = logits.takeRight(model.config.vocabSize)
    assertEquals(next, last.indices.maxBy(last).toLong)
    assertEquals(next, 16196L)
  }
  test("empty batches and invalid token/context shapes are explicit") {
    val model = LlamaModel.init(LlamaConfig.fromJson(configJson), 1)
    val empty = Tensor.fromLongs(Array.emptyLongArray, Shape.from(Seq(2L,0L)))
    assertEquals(model.logits(empty).shape.dims, Vector(2L,0L,8L))
    intercept[IllegalArgumentException](model.logits(Tensor.fromLongs(Array(-1L), Shape.from(Seq(1L,1L)))))
    intercept[IllegalArgumentException](model.logits(Tensor.fromLongs(Array(8L), Shape.from(Seq(1L,1L)))))
    intercept[IllegalArgumentException](model.logits(Tensor.fromLongs(Array(1L), Shape.from(Seq(1L)))))
    intercept[IllegalArgumentException](model.logits(Tensor.fromLongs(Array.fill(17)(1L), Shape.from(Seq(1L,17L)))))
  }
  test("single-file loader handles omitted tied head and rejects name shape or tie conflicts") {
    val directory = Files.createTempDirectory("scala-torch-llama-load-")
    val json = configJson.trim.dropRight(1) + ",\"tie_word_embeddings\":true}"
    val model = LlamaModel.init(LlamaConfig.fromJson(json), 5)
    val weights = model.namedWeights.filterNot(_._1 == "lm_head.weight").map((name, p) => name -> p.data)
    def write(values: Vector[(String,Tensor[Float])]): Unit =
      val bytes = java.nio.ByteBuffer.allocate(values.map(_._2.shape.numel.toInt * 4).sum).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      var offset = 0L
      val entries = values.map { (name, tensor) =>
        val begin = offset
        tensor.toFloatArray.foreach(bytes.putFloat)
        offset += tensor.shape.numel * 4
        s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${tensor.shape.dims.mkString(",")}],\"data_offsets\":[$begin,$offset]}"
      }
      val header = entries.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
      val prefix = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).array()
      Files.writeString(directory.resolve("config.json"), json)
      Files.write(directory.resolve("model.safetensors"), prefix ++ header ++ bytes.array())
    try
      write(weights)
      val loaded = LlamaModel.load(directory, trainable = true)
      assert(loaded.embedding.weight eq loaded.lmHead.weight)
      assertEquals(loaded.logits(tokens).toFloatArray.toSeq, model.logits(tokens).toFloatArray.toSeq)
      loaded.forward(tokens).sum().backward()
      assert(Module.parameters(loaded).forall(_.grad.nonEmpty))
      write(weights.drop(1))
      intercept[IllegalArgumentException](LlamaModel.load(directory))
      write(weights :+ ("extra" -> Tensor.fromFloats(Array(1f), Shape.from(Seq.empty))))
      intercept[IllegalArgumentException](LlamaModel.load(directory))
      val (name, value) = weights.head
      write(weights.updated(0, name -> value.reshape(Shape.from(Seq(value.shape.numel)))))
      intercept[IllegalArgumentException](LlamaModel.load(directory))
      val wrongHead = Tensor.fromFloats(Array.fill(32)(9f), Shape.from(Seq(8L,4L)))
      write(weights :+ ("lm_head.weight" -> wrongHead))
      intercept[IllegalArgumentException](LlamaModel.load(directory))
    finally
      Files.deleteIfExists(directory.resolve("model.safetensors"))
      Files.deleteIfExists(directory.resolve("config.json"))
      Files.deleteIfExists(directory)
  }
  test("initializer configuration and greedy EOS/tie behavior are deterministic") {
    val json = configJson.trim.dropRight(1) + ",\"initializer_range\":0}"
    val c = LlamaConfig.fromJson(json).copy(eosTokenIds = Vector(0L))
    val model = LlamaModel.init(c, 7)
    assert(model.logits(tokens).toFloatArray.forall(_ == 0f))
    assertEquals(model.generateIds(Vector(1L), 3), Vector(1L,0L))
    val noEos = LlamaModel.init(c.copy(eosTokenIds = Vector.empty), 7)
    assertEquals(noEos.generateIds(Vector(1L), 2), Vector(1L,0L,0L))
  }
