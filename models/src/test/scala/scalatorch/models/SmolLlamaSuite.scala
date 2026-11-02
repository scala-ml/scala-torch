package scalatorch.models

import munit.FunSuite
import scalatorch.{Tensor, Shape}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path}
import java.security.{DigestInputStream, MessageDigest}
import java.io.OutputStream

class SmolLlamaSuite extends FunSuite:
  private def sha(path: Path): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val input = new DigestInputStream(Files.newInputStream(path),digest)
    try input.transferTo(OutputStream.nullOutputStream()) finally input.close()
    digest.digest().map(b => f"${b & 255}%02x").mkString
  private def resource(name: String): Array[Byte] =
    val input = getClass.getResourceAsStream(s"/llama/smollm2/$name")
    require(input != null, s"missing reference: $name")
    val bytes = try input.readAllBytes() finally input.close()
    val hashes = Map(
      "ids.txt" -> "456ccb39f252eedc58f2286798457bba2c1bcd43fb736fbda554000c35c0f626",
      "logits-f32.bin" -> "3e72506c0e7d8bb67a2601a18931566a17c20ee1fa14979ccd16b9a447f4d5f4",
      "text.txt" -> "a1b7eb2ee7a6aded8dda4e6cf30826f5afffb28a5597ee9389e91eb326d4e319")
    assertEquals(MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString,hashes(name))
    bytes

  test("trained SmolLM2 BF16 weights match Candle F32 logits and greedy text offline") {
    val dir = sys.props.get("scalatorch.smollm.model.dir").getOrElse(assume(false,"set scalatorch.smollm.model.dir for trained-model gate").toString)
    val directory = Path.of(dir)
    assertEquals(sha(directory.resolve("config.json")),"8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a")
    assertEquals(sha(directory.resolve("model.safetensors")),"5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c")
    assertEquals(sha(directory.resolve("tokenizer.json")),"9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
    val tokenizer = ByteBpeTokenizer.read(directory.resolve("tokenizer.json"))
    val model = LlamaModel.load(directory)
    val prompt = SmolChat.format(Vector(ChatMessage("user","What is the capital of France?")))
    val lines = new String(resource("ids.txt"),java.nio.charset.StandardCharsets.UTF_8).linesIterator.toVector
    assertEquals(lines.size,2)
    val promptIds = lines(0).split(",").map(_.toLong).toVector
    val expectedIds = lines(1).split(",").map(_.toLong).toVector
    assertEquals(tokenizer.encode(prompt).toVector,promptIds)
    val reference = ByteBuffer.wrap(resource("logits-f32.bin")).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(reference.remaining(),4 * model.config.vocabSize * 4)
    val cache = model.newCache(capacity=promptIds.size+16)
    var maxError = 0.0
    try
      for step <- 0 until 4 do
        val ids = if step == 0 then promptIds else Vector(expectedIds(step-1))
        val actual = cache.decode(Tensor.fromLongs(ids.toArray,Shape.from(Seq(1L,ids.size.toLong)))).toFloatArray.takeRight(model.config.vocabSize)
        actual.foreach { value =>
          val expected = reference.getFloat()
          val error = math.abs(value.toDouble-expected.toDouble)
          maxError = math.max(maxError,error)
          assert(error <= 2e-4 + 2e-4*math.abs(expected.toDouble), clues(step,value,expected,error))
        }
    finally cache.close()
    println(s"SmolLM2 Candle max absolute logit error: $maxError")
    val generated = TextGeneration.generate(model,tokenizer,prompt,16)
    assertEquals(generated.ids,expectedIds)
    assertEquals(generated.text,new String(resource("text.txt"),java.nio.charset.StandardCharsets.UTF_8))
    assert(generated.text.contains("Paris"))
  }
