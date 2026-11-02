package scalatorch.models

import scalatorch.*
import scalatorch.nn.{AdamW,Module}

class LanguageTrainingSuite extends munit.FunSuite:
  test("AdamW trains a small Llama on shifted next-token targets") {
    val model=LlamaModel.init(LlamaConfig(8,8,16,1,2,1,maxPositions=16,initializerRange=.1),7)
    val sequence=Tensor.fromLongs(Array(1L,2L,3L,4L,1L,4L,1L,2L,3L,4L),Shape.from(Seq(2L,5L)))
    val input=sequence.slice(1,0,4);val targets=sequence.slice(1,1,5)
    val optimizer=new AdamW(Module.parameters(model),learningRate=.03)
    val initial=model.forward(input).crossEntropy(targets).data.toFloatArray.head
    for _ <- 0 until 60 do
      optimizer.zeroGrad()
      val loss=model.forward(input).crossEntropy(targets)
      assert(loss.data.toFloatArray.head.isFinite)
      loss.backward()
      assert(Module.parameters(model).forall(_.grad.exists(_.toFloatArray.forall(_.isFinite))))
      optimizer.step()
    val result=model.forward(input)
    val last=result.crossEntropy(targets).data.toFloatArray.head
    println(s"Llama next-token loss: $initial -> $last")
    assert(last < .15 && last < initial*.1)
    val predictions=result.data.toFloatArray.grouped(8).map(row => row.indices.maxBy(row).toLong).toVector
    assertEquals(predictions,targets.toLongArray.toVector)
  }
