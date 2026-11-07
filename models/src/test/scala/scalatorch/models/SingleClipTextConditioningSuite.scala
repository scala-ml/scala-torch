package scalatorch.models

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.Path
import java.nio.charset.StandardCharsets
import java.util.concurrent.CancellationException

class SingleClipTextConditioningSuite extends munit.FunSuite:
  test("single CLIP conditioning matches NUI for weights, chunks, clip skip, padding and truncation") {
    val directory=sys.props.get("scalatorch.sd15.dir")
    assume(directory.nonEmpty,"set scalatorch.sd15.dir to the pinned tiny SD1.5 Diffusers model")
    val root=Path.of(directory.get)
    val tokenizerRoot=root.resolve("tokenizer")
    val encoder=PretrainedClip.load(root.resolve("text_encoder"))
    val manifestStream=getClass.getResourceAsStream("/tiny-sd15-prompt/manifest.json")
    val manifestBytes=try manifestStream.readAllBytes() finally manifestStream.close()
    val manifest=TokenizerJson.parse(new String(manifestBytes,StandardCharsets.UTF_8)).obj
    val cases=manifest("cases").arr

    for entry <- cases do
      val c=entry.obj
      val customPad=c("custom_pad") match
        case TokenizerJson.Value.Bool(value) => value
        case _ => fail("custom_pad must be a boolean")
      val truncate=c("truncate") match
        case TokenizerJson.Value.Bool(value) => value
        case _ => fail("truncate must be a boolean")
      val tokenizer=if customPad then
        ClipFastTokenizer.load(tokenizerRoot.resolve("vocab.json"),tokenizerRoot.resolve("merges.txt"),padToken="!")
      else ClipFastTokenizer.loadDirectory(tokenizerRoot)
      val conditioning=new SingleClipTextConditioning(tokenizer,encoder)
      val pair=conditioning.encode(c("prompt").str,c("negative").str,c("clip_skip").int,
        truncatePrompt=truncate)
      for (name,actual) <- Seq("positive"->pair.positive,"negative"->pair.negative.get) do
        val spec=c("shapes").obj(name).arr.map(_.int.toLong).toSeq
        assertEquals(actual.shape.dims,spec,s"${c("name").str} $name shape")
        val file=c("files").obj(name).obj("name").str
        val stream=getClass.getResourceAsStream(s"/tiny-sd15-prompt/$file")
        val bytes=try stream.readAllBytes() finally stream.close()
        val buffer=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val expected=Array.fill(buffer.remaining()/4)(buffer.getFloat())
        val actualValues=actual.toFloatArray
        val maxError=actualValues.zip(expected).map((a,b) => math.abs(a.toDouble-b)).max
        println(s"SD1 prompt ${c("name").str} $name maxAbsError=$maxError")
        assert(maxError<3e-5,s"${c("name").str} $name differs from NUI")
      assertEquals(pair.positive.shape.dims(1),pair.negative.get.shape.dims(1))

    val tokenizer=ClipFastTokenizer.loadDirectory(tokenizerRoot)
    val conditioning=new SingleClipTextConditioning(tokenizer,encoder)
    val unguided=conditioning.encode("apple","\uD800",encodeNegative=false)
    assertEquals(unguided.negative,None)
    intercept[IllegalArgumentException](conditioning.encode("apple","",encoder.layers.size+2))
    intercept[IllegalArgumentException](new SingleClipTextConditioning(tokenizer,encoder,maxElements=1)
      .encode("apple",""))
    var cancellationChecks=0
    intercept[CancellationException](conditioning.encode("apple, "*40,"blur",cancelled=() => {
      cancellationChecks+=1;cancellationChecks>=4
    }))
    assertEquals(cancellationChecks,4)
  }
