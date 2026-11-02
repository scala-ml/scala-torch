package scalatorch.models

import scalatorch.*
import scalatorch.autograd.{Autograd,Variable}
import scalatorch.nn.*

class ClipTextSuite extends munit.FunSuite:
  private def model(): ClipTextModel =
    var seed=200L
    def linear(in: Int,out: Int): Linear={seed+=1;Linear.init(in,out,seed)}
    def layer(activation: ClipActivation)=ClipTextLayer(LayerNorm.init(4),linear(4,4),linear(4,4),linear(4,4),
      linear(4,4),LayerNorm.init(4),linear(4,7),linear(7,4),2,activation)
    ClipTextModel(Embedding.init(11,4,1),Embedding.init(8,4,2),
      Vector(layer(ClipActivation.QuickGelu),layer(ClipActivation.Gelu)),LayerNorm.init(4))
  private def tokens(ids: Long*)=Tensor.fromLongs(ids.toArray,Shape.from(Seq(1,ids.size.toLong)))
  test("CLIP is causal, supports batched variable-length sequences and normalized earlier layers") {
    val m=model()
    val original=m(tokens(1,2,3,4)).data.toFloatArray
    val changed=m(tokens(1,2,8,9)).data.toFloatArray
    assertEquals(original.take(8).toVector,changed.take(8).toVector)
    assert(original.drop(8).toVector!=changed.drop(8).toVector)
    val short=m(tokens(1,2)).data.toFloatArray
    assertEquals(short.toVector,original.take(8).toVector)
    val batch=m(Tensor.fromLongs(Array(1L,2L,1L,2L),Shape.from(Seq(2,2)))).data.toFloatArray
    assertEquals(batch.toVector,short.toVector++short.toVector)
    val ids=tokens(1,2,3,4)
    val pos=tokens(0,1,2,3)
    val embedded=m.tokenEmbedding(ids)+m.positionEmbedding(pos)
    val expected=m.finalNorm(m.layers.head(embedded,m.maxScoreElements)).data
    assertEquals(m(ids,skipLastLayers=1).data.toFloatArray.toVector,expected.toFloatArray.toVector)
    assertEquals(m(ids,skipLastLayers=2).data.toFloatArray.toVector,m.finalNorm(embedded).data.toFloatArray.toVector)
  }
  test("CLIP backward reaches embeddings and every layer with directional gradient agreement") {
    val m=model();val ids=tokens(1,2,2,4)
    val result=m(ids)
    val seed=Tensor.fromFloats(Array.tabulate(16)(i => math.cos(i*.37).toFloat),result.data.shape)
    result.backward(seed)
    val params=Module.parameters(m)
    assert(params.forall(p => p.grad.exists(_.toFloatArray.forall(_.isFinite))))
    assert(m.tokenEmbedding.weight.grad.get.toFloatArray.slice(2*4,3*4).exists(math.abs(_)>1e-6))
    val originals=params.map(p => p->p.data).toMap
    val gradients=params.map(p => p->p.grad.get.toFloatArray).toMap
    for n <- 0 until 3 do
      val random=new java.util.Random(n+700)
      val directions=params.map(p => p->Array.fill(p.data.shape.numel.toInt)((random.nextDouble()*.2-.1).toFloat)).toMap
      val expected=params.map(p => gradients(p).zip(directions(p)).map((a,b) => a.toDouble*b).sum).sum
      def score(step: Float): Double =
        params.foreach(p => p.load(Tensor.fromFloats(originals(p).toFloatArray.zip(directions(p)).map((a,b) => a+step*b),p.data.shape)))
        Autograd.noGrad(m(ids)).data.toFloatArray.zip(seed.toFloatArray).map((a,b) => a.toDouble*b).sum
      val numerical=(score(.002f)-score(-.002f))/.004
      params.foreach(p => p.load(originals(p)))
      assertEqualsDouble(expected,numerical,.002)
  }
  test("CLIP rejects bad shapes, token IDs, context, skip counts and workspace budgets") {
    val m=model()
    intercept[IllegalArgumentException](m(tokens(11)))
    intercept[IllegalArgumentException](m(tokens(-1)))
    intercept[IllegalArgumentException](m(tokens()))
    intercept[IllegalArgumentException](m(tokens(1,2,3,4,5,6,7,8,9)))
    intercept[IllegalArgumentException](m(tokens(1),skipLastLayers=3))
    intercept[IllegalArgumentException](m(tokens(1),skipLastLayers= -1))
    intercept[IllegalArgumentException](m.copy(maxScoreElements=1)(tokens(1,2)))
    intercept[IllegalArgumentException](m.copy(maxHiddenElements=1)(tokens(1,2)))
    intercept[IllegalArgumentException](m.copy(layers=Vector.empty))
    intercept[IllegalArgumentException](m.layers.head.copy(heads=3))
  }
  test("empty batches still respect positional embedding allocation budget") {
    val m=model()
    val empty=Tensor.fromLongs(Array.emptyLongArray,Shape.from(Seq(0,8)))
    assertEquals(m(empty).data.shape,Shape.from(Seq(0,8,4)))
    intercept[IllegalArgumentException](m.copy(maxHiddenElements=31)(empty))
    intercept[IllegalArgumentException](m.copy(maxHiddenElements=0)(empty,skipLastLayers=2))
    assertEquals(m.copy(maxHiddenElements=32)(empty).data.shape,Shape.from(Seq(0,8,4)))
  }
