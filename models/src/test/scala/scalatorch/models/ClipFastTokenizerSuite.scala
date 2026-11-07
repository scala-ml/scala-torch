package scalatorch.models

import scalatorch.*
import java.nio.file.{Files,Path}
import java.nio.charset.StandardCharsets.UTF_8

class ClipFastTokenizerSuite extends munit.FunSuite:
  test("removing an embedding remaps later token IDs without changing the original tokenizer") {
    val base=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1,"a</w>":2}""","#version: 0.2\n")
    val original=base.withAddedTokens(Vector("first"->2,"second"->1))
    val (removed,keep)=original.withoutAddedTokenGroup("first")
    assertEquals(keep,Vector(0L,1L,2L,5L))
    assertEquals(removed.encode("second",false).toVector,Vector(3L))
    assertEquals(original.encode("second",false).toVector,Vector(5L))
    val replaced=removed.withAddedTokens(Vector("first"->1))
    assertEquals(replaced.encode("first second",false).toVector,Vector(4L,3L))
    intercept[IllegalArgumentException](original.withoutAddedTokenGroup("missing"))
  }
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
  test("added textual-inversion tokens preserve vocabulary and expand vector rows in order") {
    val base=tokenizer()
    val extended=base.withAddedTokens(Vector("sks"->2))
    val first=base.vocabSize.toLong
    assertEquals(extended.vocabSize,base.vocabSize+2)
    assertEquals(extended.encode("sks").toVector,Vector(extended.bosId,first,first+1,extended.eosId))
    assertEquals(extended.encode("sks_1").toVector,Vector(extended.bosId,first+1,extended.eosId))
    assertEquals(base.encode("sks").toVector,tokenizer().encode("sks").toVector)
    intercept[IllegalArgumentException](extended.withAddedTokens(Vector("sks"->1)))
  }
  test("NUI fallback normalization is shared by prompt chunks and added-token copies") {
    val fast=tokenizer()
    val native=fast.withNuiFallbackNormalization
    val text="日本語 A\u0000B"
    val ids=native.encode(text,addSpecialTokens=false).toVector
    assertNotEquals(ids,fast.encode(text,addSpecialTokens=false).toVector)
    assertEquals(ClipPromptEncoding.chunks(native,text).head.tokens.slice(1,1+ids.size),ids)
    val extended=native.withAddedTokens(Vector("sks"->2))
    assertEquals(extended.encode(text,addSpecialTokens=false).toVector,ids)
    assertEquals(extended.encode("sks",addSpecialTokens=false).toVector,Vector(fast.vocabSize.toLong,fast.vocabSize.toLong+1))
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
