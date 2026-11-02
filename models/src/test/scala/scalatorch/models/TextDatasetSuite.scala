package scalatorch.models

import java.nio.file.Files

class TextDatasetSuite extends munit.FunSuite:
  test("packed batches cover every next-token target and ignore the final padding") {
    val source=Array(1L,2L,3L,4L,5L,6L)
    val data=TextDataset.fromTokens(source,8,0,sequenceLength=2,batchSize=2)
    source(0)=7L
    assertEquals(data.batchesPerEpoch,2L)
    val first=data.batch(0)
    assertEquals(first.input.toLongArray.toVector,Vector(1L,2L,3L,4L))
    assertEquals(first.targets.toLongArray.toVector,Vector(2L,3L,4L,5L))
    assertEquals(first.targetCount,4)
    val last=data.batch(1)
    assertEquals(last.input.toLongArray.toVector,Vector(5L,0L,0L,0L))
    assertEquals(last.targets.toLongArray.toVector,Vector(6L,-100L,-100L,-100L))
    assertEquals(last.targetCount,1)
    assertEquals(data.batch(2).input.toLongArray.toVector,first.input.toLongArray.toVector)
    assertEquals(data.batch(Long.MaxValue).targets.toLongArray.toVector,last.targets.toLongArray.toVector)
    intercept[IllegalArgumentException](data.batch(-1))
  }
  test("bounded UTF-8 documents tokenize with EOS and a stable corpus fingerprint") {
    val resource=getClass.getResourceAsStream("/tokenizer/synthetic.json")
    val tokenizer=try ByteBpeTokenizer.fromJson(new String(resource.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)) finally resource.close()
    val path=Files.createTempFile("training-data-",".txt")
    try
      val text="ababa\n\n日本語\n"
      Files.writeString(path,text)
      val eos=tokenizer.vocabSize-1L
      val expected=tokenizer.encode("ababa")++Array(eos)++tokenizer.encode("日本語")++Array(eos)
      val actual=TextDataset.read(path,tokenizer,eos,4,2)
      val direct=TextDataset.fromTokens(expected,tokenizer.vocabSize,eos,4,2)
      assertEquals(actual.fingerprint,direct.fingerprint)
      assertEquals(actual.batch(0).targets.toLongArray.toVector,direct.batch(0).targets.toLongArray.toVector)
      intercept[IllegalArgumentException](TextDataset.read(path,tokenizer,eos,4,2,maxBytes=2))
      intercept[IllegalArgumentException](TextDataset.read(path,tokenizer,eos,4,2,maxTokens=2))
      Files.write(path,Array(0xc3.toByte,0x28.toByte))
      intercept[IllegalArgumentException](TextDataset.read(path,tokenizer,eos,4,2))
      Files.writeString(path,"\n\n")
      intercept[IllegalArgumentException](TextDataset.read(path,tokenizer,eos,4,2))
    finally Files.deleteIfExists(path)
  }
  test("invalid dimensions and token limits reject before constructing batches") {
    for values <- Seq(Array.emptyLongArray,Array(1L),Array(-1L,2L),Array(1L,8L)) do
      intercept[IllegalArgumentException](TextDataset.fromTokens(values,8,0,2,1))
    intercept[IllegalArgumentException](TextDataset.fromTokens(Array(1L,2L),8,0,0,1))
    intercept[IllegalArgumentException](TextDataset.fromTokens(Array(1L,2L),8,0,2,0))
    intercept[IllegalArgumentException](TextDataset.fromTokens(Array(1L,2L),8,8,2,1))
    intercept[IllegalArgumentException](TextDataset.fromTokens(Array(1L,2L),8,0,Int.MaxValue,2))
  }
