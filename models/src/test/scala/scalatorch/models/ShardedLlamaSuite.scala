package scalatorch.models

import scalatorch.*
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import scala.util.Using

class ShardedLlamaSuite extends munit.FunSuite:
  test("sharded tied model preserves logits aliases and gradients") {
    val dir = Files.createTempDirectory("scala-torch-sharded-llama-")
    val json = """{"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"max_position_embeddings":16,"tie_word_embeddings":true}"""
    val original = LlamaModel.init(LlamaConfig.fromJson(json),7)
    val weights = original.namedWeights.filterNot(_._1 == "lm_head.weight")
    var total = 0L
    val map = Vector.newBuilder[String]
    try
      Files.writeString(dir.resolve("config.json"),json)
      weights.zipWithIndex.groupBy(_._2 % 2).foreach { (group,values) =>
        val filename = s"part-$group.safetensors"
        val data = ByteBuffer.allocate(values.map(_._1._2.data.shape.numel.toInt*4).sum).order(ByteOrder.LITTLE_ENDIAN)
        var offset = 0L
        val entries = values.map { case ((name,parameter),_) =>
          val tensor = parameter.data
          val start = offset
          tensor.toFloatArray.foreach(data.putFloat)
          offset += tensor.shape.numel*4
          map += s"\"$name\":\"$filename\""
          s"\"$name\":{\"dtype\":\"F32\",\"shape\":[${tensor.shape.dims.mkString(",")}],\"data_offsets\":[$start,$offset]}"
        }
        total += offset
        val header = entries.mkString("{",",","}").getBytes(java.nio.charset.StandardCharsets.UTF_8)
        Files.write(dir.resolve(filename),ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).array() ++ header ++ data.array())
      }
      Files.writeString(dir.resolve("model.safetensors.index.json"),s"""{"metadata":{"total_size":$total},"weight_map":{${map.result().mkString(",")}}}""")
      val loaded = LlamaModel.load(dir,trainable=true)
      assert(loaded.embedding.weight eq loaded.lmHead.weight)
      val input = Tensor.fromLongs(Array(1L,3L),Shape.from(Seq(1L,2L)))
      assertEquals(loaded.logits(input).toFloatArray.toVector,original.logits(input).toFloatArray.toVector)
      loaded.forward(input).sum().backward()
      assert(scalatorch.nn.Module.parameters(loaded).forall(_.grad.nonEmpty))
    finally
      Using.resource(Files.list(dir))(_.forEach(p => Files.deleteIfExists(p)))
      Files.deleteIfExists(dir)
  }
  test("Rust-resharded public checkpoint matches independent Candle logits") {
    val dir = sys.props.get("scalatorch.llama.sharded.dir").getOrElse(assume(false,"set scalatorch.llama.sharded.dir for public shard gate").toString)
    val hashes = Map(
      "config.json" -> "5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382",
      "model-00001-of-00002.safetensors" -> "a5d05cc8c8a5107b4f14c77671fbf0924de9fe5eabfa475d91a3c32995bf1555",
      "model-00002-of-00002.safetensors" -> "71393387b01ff466e85a7bb3826e113fea16f51da432dde3e66be22094761730",
      "model.safetensors.index.json" -> "3393dbcd8d3936382c7ad054951eb3605b663dc3a968e47b1d6fa31cdf868247")
    def sha(bytes: Array[Byte]): String = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
    hashes.foreach((name,hash) => assertEquals(sha(Files.readAllBytes(Path.of(dir,name))),hash))
    val model = LlamaModel.load(Path.of(dir))
    val input = Tensor.fromLongs(Array(1L,42L,17L,321L),Shape.from(Seq(1L,4L)))
    val actual = model.logits(input).toFloatArray
    val stream = getClass.getResourceAsStream("/llama/tiny-candle-f32.bin")
    val bytes = try stream.readAllBytes() finally stream.close()
    assertEquals(sha(bytes),"9c94f848e6203198bd0d469cb529d4130aca4f8c7d53a44d8ceca917212cf2c4")
    val reference = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(reference.remaining(),actual.length*4)
    actual.foreach { value =>
      val expected = reference.getFloat()
      assert(math.abs(value.toDouble-expected) <= 1e-5 + 1e-5*math.abs(expected))
    }
    assertEquals(model.generateIds(Vector(1L,42L,17L,321L),1).last,16196L)
  }
