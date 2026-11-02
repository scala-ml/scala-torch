package scalatorch.models

import scalatorch.*
import scalatorch.nn.*

class LlamaLoRASuite extends munit.FunSuite:
  private def base = LlamaModel.init(LlamaConfig(8,8,16,1,2,1,maxPositions=16,initializerRange=.1,tieWordEmbeddings=true),7)
  private val input = Tensor.fromLongs(Array(1L,2L,3L,4L),Shape.from(Seq(1L,4L)))
  private val targets = Tensor.fromLongs(Array(2L,3L,4L,1L),Shape.from(Seq(1L,4L)))

  test("pretrained SmolLM2 last-layer adapters reduce loss without changing base storage") {
    val dir = sys.props.get("scalatorch.smollm.model.dir").getOrElse(
      assume(false,"set scalatorch.smollm.model.dir for trained-model gate").toString)
    val original = LlamaModel.load(java.nio.file.Path.of(dir))
    val adapted = LlamaLoRA.attach(original,4,8.0,42,targetLayers=Some(Set(original.layers.size-1)))
    assertEquals(Module.parameters(adapted).map(_.data.shape.numel).sum, 7680L)
    val snapshots = adapted.model.namedWeights.map((_,p) => p -> p.data)
    val optimizer = new AdamW(Module.parameters(adapted),learningRate=.01)
    val initial = adapted.model.forward(input).crossEntropy(targets).data.toFloatArray.head
    for _ <- 0 until 30 do
      optimizer.zeroGrad()
      adapted.model.forward(input).crossEntropy(targets).backward()
      assert(Module.parameters(adapted).forall(_.grad.exists(_.toFloatArray.forall(_.isFinite))))
      optimizer.step()
    val last = adapted.model.forward(input).crossEntropy(targets).data.toFloatArray.head
    println(s"LoRA pretrained SmolLM2 synthetic-token loss: $initial -> $last")
    assert(last < initial * .9)
    // Tensor data is immutable: retaining the identical snapshot proves no base replacement.
    assert(snapshots.forall((p,t) => (p.data eq t) && !p.trainable && p.grad.isEmpty))
    val path=java.nio.file.Files.createTempFile("smollm-lora-",".stl")
    try
      LoRACheckpoint.save(adapted,optimizer,path,30)
      // A fresh initialization over the same frozen base must restore both factors and Adam slots.
      val restored=LlamaLoRA.attach(original,4,8.0,999,targetLayers=Some(Set(original.layers.size-1)))
      val resumed=new AdamW(Module.parameters(restored),learningRate=.01)
      assertEquals(LoRACheckpoint.load(restored,resumed,path),30L)
      assertEquals(restored.model.logits(input).toFloatArray.toVector,adapted.model.logits(input).toFloatArray.toVector)
      for (model,opt) <- Seq(adapted -> optimizer,restored -> resumed) do
        opt.zeroGrad();model.model.forward(input).crossEntropy(targets).backward();opt.step()
      assertEquals(Module.parameters(restored).map(_.data.toFloatArray.toVector),
        Module.parameters(adapted).map(_.data.toFloatArray.toVector))
      assert(java.nio.file.Files.size(path) < 200000)
    finally java.nio.file.Files.deleteIfExists(path)
  }

  test("selected adapters train while all base weights remain frozen and tied") {
    val original = base
    val adapted = LlamaLoRA.attach(original, 2, 4.0, 19L, Set("q_proj", "v_proj", "down_proj"))
    assert(adapted.model.embedding.weight eq adapted.model.lmHead.weight)
    assert(adapted.model.embedding.weight ne original.embedding.weight)
    assert(original.embedding.weight.trainable)
    assertEquals(Module.parameters(adapted).size, 6)
    assertEquals(Module.parameters(adapted.model).filter(_.trainable).toSet, Module.parameters(adapted).toSet)
    val before = adapted.model.namedWeights.map((n,p) => n -> p.data.toFloatArray.toVector).toMap
    assertEquals(adapted.model.logits(input).toFloatArray.toVector, original.logits(input).toFloatArray.toVector)
    val optimizer = new AdamW(Module.parameters(adapted), learningRate=.03)
    val initial = adapted.model.forward(input).crossEntropy(targets).data.toFloatArray.head
    val cache = adapted.model.newCache(capacity=8)
    try
      cache.decode(input)
      for _ <- 0 until 50 do
        optimizer.zeroGrad()
        adapted.model.forward(input).crossEntropy(targets).backward()
        optimizer.step()
      intercept[IllegalArgumentException](cache.decode(Tensor.fromLongs(Array(1L),Shape.from(Seq(1L,1L)))))
    finally cache.close()
    val last = adapted.model.forward(input).crossEntropy(targets).data.toFloatArray.head
    println(s"LoRA tiny Llama loss: $initial -> $last")
    assert(last < initial * .9)
    assertEquals(adapted.model.namedWeights.map((n,p) => n -> p.data.toFloatArray.toVector).toMap,before)
    assert(adapted.model.namedWeights.forall((_,p) => !p.trainable && p.grad.isEmpty))
  }

  test("selection and allocation limits are validated before attaching") {
    val model = base
    intercept[IllegalArgumentException](LlamaLoRA.attach(model,2,4,0,Set.empty))
    intercept[IllegalArgumentException](LlamaLoRA.attach(model,2,4,0,Set("unknown")))
    intercept[IllegalArgumentException](LlamaLoRA.attach(model,2,4,0,targetLayers=Some(Set(1))))
    intercept[IllegalArgumentException](LlamaLoRA.attach(model,2,4,0,targetLayers=Some(Set.empty)))
    intercept[IllegalArgumentException](LlamaLoRA.attach(model,2,4,0,maxAdapterParameters=1))
    val adapted=LlamaLoRA.attach(model,2,4,0)
    intercept[IllegalArgumentException](LlamaLoRA.attach(adapted.model,2,4,0))
    assert(model.namedWeights.forall((_,p) => p.trainable))
  }
