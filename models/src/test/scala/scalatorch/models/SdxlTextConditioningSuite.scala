package scalatorch.models

import scalatorch.*
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}

class SdxlTextConditioningSuite extends munit.FunSuite:
  test("dual CLIP contexts and pooled text match NUI including weights, chunks and empty negatives") {
    val dir=sys.props.get("scalatorch.sdxl.dir")
    assume(dir.nonEmpty,"tiny SDXL required")
    val root=Path.of(dir.get)
    def tokenizer(part: String)=ClipFastTokenizer.load(root.resolve(part).resolve("vocab.json"),root.resolve(part).resolve("merges.txt"))
    val first=tokenizer("tokenizer");val second=tokenizer("tokenizer_2")
    val encoder=PretrainedClip.load(root.resolve("text_encoder"))
    val encoder2=PretrainedClip.load(root.resolve("text_encoder_2"))
    val model=new SdxlTextConditioning(first,encoder,second,encoder2)
    val split=ClipFastTokenizer.fromText(Files.readString(root.resolve("tokenizer_2/vocab.json")),"#version: 0.2\n")
    val padded=ClipFastTokenizer.load(root.resolve("tokenizer_2/vocab.json"),root.resolve("tokenizer_2/merges.txt"),padToken="!")
    def bytes(name: String): Array[Byte] =
      val stream=getClass.getResourceAsStream(s"/tiny-sdxl-prompt/$name")
      try stream.readAllBytes() finally stream.close()
    val cases=TokenizerJson.parse(new String(bytes("manifest.json"),java.nio.charset.StandardCharsets.UTF_8)).obj("cases").arr
    for c <- cases do
      val f=c.obj
      val selected=f("name").str match
        case "split" =>
          assert(f("encoder_chunks").arr.map(_.int).distinct.size==2)
          new SdxlTextConditioning(first,encoder,split,encoder2)
        case "pad" => new SdxlTextConditioning(first,encoder,padded,encoder2)
        case _ => model
      val actual=selected.encode(f("prompt").str,f("negative").str,f("clip_skip").int)
      for (name,tensor) <- Seq("positive"->actual.positive,"negative"->actual.negative,
          "pooled"->actual.pooled,"negative-pooled"->actual.negativePooled) do
        val ref=f("tensors").obj(name).obj
        assertEquals(tensor.shape.dims,ref("shape").arr.map(_.int.toLong))
        val data=ByteBuffer.wrap(bytes(ref("file").str)).order(ByteOrder.LITTLE_ENDIAN)
        val expected=Array.fill(data.remaining()/4)(data.getFloat())
        val error=tensor.toFloatArray.zip(expected).map((a,b) => math.abs(a.toDouble-b)).max
        println(s"SDXL prompt ${f("name").str} $name maxAbsError=$error")
        assert(error<3e-5)
    intercept[IllegalArgumentException](new SdxlTextConditioning(first,encoder,second,encoder))
    val shortVocabulary=ClipFastTokenizer.fromText("""{"<|startoftext|>":0,"<|endoftext|>":1}""","#version: 0.2\n")
    intercept[IllegalArgumentException](new SdxlTextConditioning(shortVocabulary,encoder,second,encoder2))
    val shortContext=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt"),maxLength=6)
    intercept[IllegalArgumentException](new SdxlTextConditioning(shortContext,encoder,second,encoder2))
    intercept[IllegalArgumentException](new SdxlTextConditioning(first,encoder,second,encoder2,maxElements=1).encode("a",""))
    intercept[IllegalArgumentException](model.encode("a","",0))
    intercept[IllegalArgumentException](model.encode("a","",encoder.layers.size+2))
    intercept[IllegalArgumentException](model.encode("a"," "*1048577))
    assertEquals(padded.batch(Seq("apple")).ids.toLongArray.last,padded.padId)
  }
