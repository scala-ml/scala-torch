package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Autograd
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path,StandardCopyOption}

class SdxlClipSuite extends munit.FunSuite:
  private def root(): Path =
    val dir=sys.props.get("scalatorch.sdxl.dir")
    assume(dir.nonEmpty,"tiny SDXL required")
    Path.of(dir.get)
  private val ids=Tensor.fromLongs(Array[Long](0,17,22,2,1,1,0,29,2,1,1,1),Shape.from(Seq(2L,6L)))
  private def reference(name: String): Array[Float] =
    val stream=getClass.getResourceAsStream(s"/tiny-sdxl/$name.bin")
    val bytes=try ByteBuffer.wrap(stream.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN) finally stream.close()
    Array.fill(bytes.remaining()/4)(bytes.getFloat())
  private def closeTo(tensor: Tensor[Float],name: String): Unit =
    val expected=reference(name);val actual=tensor.toFloatArray
    assertEquals(actual.length,expected.length)
    val error=actual.zip(expected).map((a,e) => math.abs(a.toDouble-e)).max
    println(s"SDXL CLIP $name maxAbsError=$error")
    assert(error<2e-5)

  test("raw selected CLIP features and projected pooling match independent Transformers") {
    for part <- Seq("text_encoder","text_encoder_2") do
      val model=PretrainedClip.load(root().resolve(part))
      val output=Autograd.noGrad(model.encode(ids))
      closeTo(output.hidden.data,s"$part-penultimate")
      assertEquals(output.finalHidden.data.toFloatArray.toVector,Autograd.noGrad(model(ids)).data.toFloatArray.toVector)
      assert(output.hidden.data.toFloatArray.toVector!=Autograd.noGrad(model(ids,1)).data.toFloatArray.toVector)
      if part=="text_encoder" then
        closeTo(output.finalHidden.data,s"$part-final")
        assertEquals(output.projected,None)
      else closeTo(output.projected.get.data,s"$part-projected")
      intercept[IllegalArgumentException](model.encode(ids,skipLastLayers = -1))
      intercept[IllegalArgumentException](model.encode(ids,skipLastLayers=model.layers.size+1))
      intercept[IllegalArgumentException](model.copy(maxHiddenElements=1).encode(ids))
  }

  test("pooling handles legacy EOS override and preserves projection gradients") {
    val model=PretrainedClip.load(root().resolve("text_encoder_2"),trainable=true)
    val output=model.encode(ids,legacyEosOverride=Some(2L))
    val expected=scalatorch.autograd.Variable.cat(Seq(
      output.finalHidden.narrow(0,0,1).narrow(1,3,1).reshape(Shape.from(Seq(1L,32L))),
      output.finalHidden.narrow(0,1,1).narrow(1,2,1).reshape(Shape.from(Seq(1L,32L)))),0)
    assertEquals(output.pooled.data.toFloatArray.toVector,expected.data.toFloatArray.toVector)
    val projected=output.projected.get
    closeTo(projected.data,"text_encoder_2-eos-override")
    projected.backward(Tensor.fromFloats(Array.fill(projected.data.shape.numel.toInt)(1f),projected.data.shape))
    assert(model.textProjection.get.weight.grad.exists(_.toFloatArray.exists(math.abs(_)>1e-6)))
    assert(model.tokenEmbedding.weight.grad.exists(_.toFloatArray.exists(math.abs(_)>1e-6)))
  }

  test("modern first EOS and missing EOS fallback match Transformers and skip bounds include embeddings") {
    val model=PretrainedClip.load(root().resolve("text_encoder_2"))
    val modernIds=Tensor.fromLongs(Array[Long](0,29,17,29,1,1,0,22,17,1,1,1),Shape.from(Seq(2L,6L)))
    closeTo(Autograd.noGrad(model.copy(eosTokenId=29).encode(modernIds)).projected.get.data,"text_encoder_2-modern-eos")
    val all=Autograd.noGrad(model.encode(ids,skipLastLayers=0))
    assertEquals(Autograd.noGrad(model.finalNorm(all.hidden)).data.toFloatArray.toVector,all.finalHidden.data.toFloatArray.toVector)
    val raw=Autograd.noGrad(model.encode(ids,skipLastLayers=model.layers.size)).hidden
    assertEquals(Autograd.noGrad(model.finalNorm(raw)).data.toFloatArray.toVector,
      Autograd.noGrad(model(ids,model.layers.size)).data.toFloatArray.toVector)
    val empty=Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(0L,6L)))
    assertEquals(Autograd.noGrad(model.encode(empty)).projected.get.data.shape,Shape.from(Seq(0L,32L)))
  }

  test("projected CLIP loader rejects wrong dimensions missing projection and undeclared projection") {
    val directory=root()
    val temp=Files.createTempDirectory("projected-clip")
    try
      def load(part: String,change: String => String): Unit =
        Files.writeString(temp.resolve("config.json"),change(Files.readString(directory.resolve(part).resolve("config.json"))))
        Files.copy(directory.resolve(part).resolve("model.safetensors"),temp.resolve("model.safetensors"),StandardCopyOption.REPLACE_EXISTING)
        PretrainedClip.load(temp)
        ()
      val missing=intercept[IllegalArgumentException](load("text_encoder",_.replace("CLIPTextModel","CLIPTextModelWithProjection")))
      assert(missing.getMessage.contains("missing CLIP weight: text_projection.weight"))
      val shape=intercept[IllegalArgumentException](load("text_encoder_2",_.replace("\"projection_dim\": 32","\"projection_dim\": 33")))
      assert(shape.getMessage.contains("CLIP weight shape mismatch: text_projection.weight"))
      val undeclared=intercept[IllegalArgumentException](load("text_encoder_2",_.replace("CLIPTextModelWithProjection","CLIPTextModel")))
      assert(undeclared.getMessage.contains("unused CLIP weights: text_projection.weight"))
    finally
      val paths=Files.list(temp)
      try paths.forEach(p => {Files.delete(p);()}) finally paths.close()
      Files.delete(temp)
  }
