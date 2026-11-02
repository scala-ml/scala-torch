package scalatorch.models

import scalatorch.*
import scalatorch.nn.*
import java.nio.file.Files

class LoRACheckpointSuite extends munit.FunSuite:
  private val config = LlamaConfig(8,8,16,1,2,1,maxPositions=16,initializerRange=.1)
  private val input = Tensor.fromLongs(Array(1L,2L,3L),Shape.from(Seq(1L,3L)))
  private val targets = Tensor.fromLongs(Array(2L,3L,4L),Shape.from(Seq(1L,3L)))
  private def adapted(seed: Long = 7, alpha: Double = 4) = LlamaLoRA.attach(LlamaModel.init(config,seed),2,alpha,42)
  private def weights(a: LlamaLoRA) = Module.parameters(a).map(_.data.toFloatArray.toVector)
  private def step(a: LlamaLoRA, o: AdamW): Unit =
    o.zeroGrad()
    a.model.forward(input).crossEntropy(targets).backward()
    o.step()
  private def temporary(body: java.nio.file.Path => Unit): Unit =
    val dir=Files.createTempDirectory("lora-checkpoint-test")
    try body(dir.resolve("adapter.stl"))
    finally
      val paths=Files.list(dir)
      try paths.forEach(p => Files.delete(p)) finally paths.close()
      Files.delete(dir)

  test("adapter bundle restores weights and exact AdamW continuation") {
    temporary { path =>
      val source=adapted();val optimizer=new AdamW(Module.parameters(source),.03)
      for _ <- 0 until 4 do step(source,optimizer)
      LoRACheckpoint.save(source,optimizer,path,4)
      val restored=adapted();val resumed=new AdamW(Module.parameters(restored),.03)
      assertEquals(LoRACheckpoint.load(restored,resumed,path,maxBytes=Files.size(path)),4L)
      assertEquals(weights(restored),weights(source))
      assertEquals(restored.model.logits(input).toFloatArray.toVector,source.model.logits(input).toFloatArray.toVector)
      for _ <- 0 until 3 do
        step(source,optimizer);step(restored,resumed)
        assertEquals(weights(restored),weights(source))
    }
  }

  test("different base, alpha and config reject before changing parameters or optimizer") {
    temporary { path =>
      val source=adapted();val optimizer=new AdamW(Module.parameters(source),.03)
      step(source,optimizer)
      LoRACheckpoint.save(source,optimizer,path,1)
      val differentConfig=LlamaLoRA.attach(LlamaModel.init(config.copy(ropeTheta=20000),7),2,4,42)
      for target <- Seq(adapted(seed=8),adapted(alpha=8),differentConfig) do
        val opt=new AdamW(Module.parameters(target),.03)
        val before=weights(target)
        intercept[IllegalArgumentException](LoRACheckpoint.load(target,opt,path))
        assertEquals(weights(target),before)
        assert(Module.parameters(target).forall(_.grad.isEmpty))
        val control=if target eq differentConfig then LlamaLoRA.attach(LlamaModel.init(config.copy(ropeTheta=20000),7),2,4,42)
          else if target.adapters.head._2.alpha==8 then adapted(alpha=8) else adapted(seed=8)
        step(target,opt);step(control,new AdamW(Module.parameters(control),.03))
        assertEquals(weights(target),weights(control))
    }
  }

  test("corruption, truncation, trailing bytes and byte budgets reject without mutation") {
    temporary { path =>
      val source=adapted();val opt=new AdamW(Module.parameters(source),.03)
      step(source,opt);LoRACheckpoint.save(source,opt,path,1)
      val bytes=Files.readAllBytes(path)
      val corrupt=bytes.clone();corrupt(corrupt.length-1)=(corrupt.last ^ 1).toByte
      val target=adapted();val optimizer=new AdamW(Module.parameters(target),.03)
      val control=adapted();val controlOptimizer=new AdamW(Module.parameters(control),.03)
      step(target,optimizer);step(control,controlOptimizer)
      val before=weights(target)
      val gradients=Module.parameters(target).map(_.grad)
      val excessiveLength=bytes.clone()
      java.nio.ByteBuffer.wrap(excessiveLength).putLong(40,Long.MaxValue)
      for broken <- Seq(corrupt,bytes.take(3),bytes.take(bytes.length-1),bytes ++ Array[Byte](0),excessiveLength) do
        Files.write(path,broken)
        intercept[IllegalArgumentException](LoRACheckpoint.load(target,optimizer,path))
        assertEquals(weights(target),before)
        assertEquals(Module.parameters(target).map(_.grad),gradients)
      Files.write(path,bytes)
      for limit <- Seq(1L,48L,bytes.length.toLong-1) do
        intercept[IllegalArgumentException](LoRACheckpoint.load(target,optimizer,path,maxBytes=limit))
        intercept[IllegalArgumentException](LoRACheckpoint.save(source,opt,path,1,maxBytes=limit))
      assertEquals(Files.readAllBytes(path).toVector,bytes.toVector)
      val listing=Files.list(path.getParent)
      try assertEquals(listing.count(),1L) finally listing.close()
      step(target,optimizer);step(control,controlOptimizer)
      assertEquals(weights(target),weights(control))
    }
  }
