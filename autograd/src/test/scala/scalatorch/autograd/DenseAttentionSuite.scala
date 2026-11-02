package scalatorch.autograd

import scalatorch.*

class DenseAttentionSuite extends munit.FunSuite:
  private def v(a: Array[Float],s: Long*)=Variable(Tensor.fromFloats(a,Shape.from(s)),true)
  test("noncausal queries see all keys and retain grouped heads with unequal lengths") {
    val q=v(Array.fill(4)(0f),1,2,2,1)
    val k=v(Array(1f,2f,3f),1,1,3,1)
    val value=v(Array(2f,4f,9f),1,1,3,1)
    val out=q.scaledDotProductAttention(k,value)
    out.data.toFloatArray.foreach(a => assertEqualsDouble(a,5.0,1e-6))
    out.sum().backward()
    value.grad.get.toFloatArray.foreach(a => assertEqualsDouble(a,4.0/3,1e-6))
    assertEquals(q.causalAttention(k,value).data.toFloatArray.toVector,Vector(2f,3f,2f,3f))
    intercept[IllegalArgumentException](q.scaledDotProductAttention(k,value,maxScoreElements=11))
    val empty=v(Array.emptyFloatArray,1,2,0,1)
    empty.scaledDotProductAttention(k,value).sum().backward()
    assertEquals(empty.grad.get.shape,empty.data.shape)
  }
  test("dense GQA pullbacks match every query, key and value finite difference") {
    val qa=Array.tabulate(8)(i => math.sin(i*.3).toFloat)
    val ka=Array.tabulate(6)(i => math.cos(i*.2).toFloat)
    val va=Array.tabulate(9)(i => math.sin(i*.7).toFloat)
    def args()=(v(qa,1,2,2,2),v(ka,1,1,3,2),v(va,1,1,3,3))
    val (q,k,value)=args()
    val y=q.scaledDotProductAttention(k,value)
    val seed=Tensor.fromFloats(Array.tabulate(12)(i => math.cos(i*.4).toFloat),y.data.shape)
    y.backward(seed)
    def score(): Double =
      val (a,b,c)=args()
      Autograd.noGrad(a.scaledDotProductAttention(b,c)).data.toFloatArray.zip(seed.toFloatArray).map((x,g) => x.toDouble*g).sum
    for (values,grad) <- Seq(qa -> q.grad.get,ka -> k.grad.get,va -> value.grad.get) do
      for i <- values.indices do
        val saved=values(i);val e=.001f
        values(i)=saved+e;val plus=score()
        values(i)=saved-e;val minus=score()
        values(i)=saved
        assertEqualsDouble(grad.toFloatArray(i),(plus-minus)/(2*e),.0002)
  }
