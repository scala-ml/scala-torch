package scalatorch.models

import scalatorch.*
import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets.UTF_8

class ClipFastTokenizerSuite extends munit.FunSuite:
  private val hashes=Map("vocab.json"->"f27915ca8a787e651ea25684f581739403b46354a9a5be29a5afad0112d60d46","merges.txt"->"38e3006b0c2382dc853454c0643ce3d6f62544c7d6ded433d8d15dab639f0b5b","reference.json"->"69a0ae2f8b0dfd2d3399e548ab671fabdd713d8cb46db5a60c6e9fb1d58a8533")
  private def resource(name: String): Array[Byte] =
    val stream=getClass.getResourceAsStream(s"/tiny-clip-tokenizer/$name")
    val bytes=try stream.readAllBytes() finally stream.close()
    val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes))
    require(hash==hashes(name),"CLIP tokenizer fixture SHA256 mismatch")
    bytes
  private def tokenizer(maxLength: Int=77): ClipFastTokenizer =
    ClipFastTokenizer.fromText(new String(resource("vocab.json"),UTF_8),new String(resource("merges.txt"),UTF_8),maxLength)
  test("CLIP fast-profile IDs match the independent multilingual reference") {
    val model=tokenizer()
    val cases=TokenizerJson.parse(new String(resource("reference.json"),UTF_8)).arr
    for entry <- cases do
      val fields=entry.obj
      val expected=fields("ids").arr.map(_.int.toLong)
      assertEquals(model.encode(fields("text").str).toVector,expected,fields("text").str)
  }
  test("CLIP batching pads with EOS and truncates only by explicit request") {
    val model=tokenizer(5)
    val batch=model.batch(Vector("", "a"))
    assertEquals(batch.ids.shape,Shape.from(Seq(2,5)))
    assertEquals(batch.ids.toLongArray.take(5).toVector,Vector(0L,1L,1L,1L,1L))
    assertEquals(batch.attentionMask.toLongArray.take(5).toVector,Vector(1L,1L,0L,0L,0L))
    intercept[IllegalArgumentException](model.batch(Vector("a red cat on a blue chair")))
    val truncated=model.batch(Vector("a red cat on a blue chair"),truncate=true)
    assertEquals(truncated.ids.toLongArray.last,1L)
    assertEquals(truncated.attentionMask.toLongArray.toVector,Vector.fill(5)(1L))
    assertEquals(model.batch(Vector.empty).ids.shape,Shape.from(Seq(0,5)))
    intercept[IllegalArgumentException](model.batch(Vector("a","b"),maxElements=9))
    assertEquals(model.encode("a",addSpecialTokens=false).length,1)
  }
  test("CLIP tokenizer validates vocabulary merges Unicode and context bounds") {
    val vocab=new String(resource("vocab.json"),UTF_8)
    val merges=new String(resource("merges.txt"),UTF_8)
    intercept[IllegalArgumentException](ClipFastTokenizer.fromText(vocab,merges,1))
    intercept[IllegalArgumentException](ClipFastTokenizer.fromText("{}",merges))
    intercept[IllegalArgumentException](ClipFastTokenizer.fromText(vocab,merges+"\nmissing missing\n"))
    intercept[IllegalArgumentException](ClipFastTokenizer.fromText(vocab,merges+"\nĠ t\n"))
    intercept[IllegalArgumentException](tokenizer().encode("\ud800"))
    intercept[IllegalArgumentException](tokenizer().encode("a"*16385))
    val dir=Files.createTempDirectory("clip-fast-tokenizer")
    try
      Files.writeString(dir.resolve("vocab.json"),vocab)
      Files.writeString(dir.resolve("merges.txt"),merges)
      assertEquals(ClipFastTokenizer.load(dir.resolve("vocab.json"),dir.resolve("merges.txt")).encode("a cat").toVector,
        tokenizer().encode("a cat").toVector)
    finally
      Files.delete(dir.resolve("vocab.json"));Files.delete(dir.resolve("merges.txt"));Files.delete(dir)
  }

  test("natural-language tokens feed the frozen public CLIP encoder") {
    val dir=sys.props.get("scalatorch.clip.dir").map(Path.of(_))
    assume(dir.nonEmpty,"set scalatorch.clip.dir to pinned public CLIP")
    val clip=PretrainedClip.load(dir.get)
    val batch=tokenizer().batch(Vector("a red cat", "a blue boat"))
    val output=clip(batch.ids)
    assertEquals(output.data.shape,Shape.from(Seq(2,77,32)))
    assert(!output.requiresGrad && output.data.toFloatArray.forall(_.isFinite))
    val values=output.data.toFloatArray
    assert(values.take(77*32).zip(values.drop(77*32)).exists((a,b) => math.abs(a-b)>1e-5))
  }
