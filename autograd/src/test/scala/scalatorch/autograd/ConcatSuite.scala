package scalatorch.autograd

import scalatorch.*

class ConcatSuite extends munit.FunSuite:
  private def v(values: Array[Float],dims: Long*)=Variable(Tensor.fromFloats(values,Shape.from(dims)),true)
  test("concatenation respects strided logical order and repeated-input gradients") {
    val x=v(Array(1f,2f,3f,4f,5f,6f),2,3).transpose(0,1)
    val out=Variable.cat(Seq(x,x),1)
    assertEquals(out.data.shape,Shape.from(Seq(3,4)))
    assertEquals(out.data.toFloatArray.toVector,Vector(1f,4f,1f,4f,2f,5f,2f,5f,3f,6f,3f,6f))
    val leaf=v(Array(1f,2f,3f,4f),2,2)
    Variable.cat(Seq(leaf,leaf),0).sum().backward()
    assertEquals(leaf.grad.get.toFloatArray.toVector,Vector.fill(4)(2f))
  }
  test("concatenation slices unequal channel gradients and handles empty inputs") {
    val a=v(Array(1f,2f),1,1,2);val b=v(Array(3f,4f,5f,6f),1,2,2)
    val out=Variable.cat(Seq(a,b),1)
    out.backward(Tensor.fromFloats(Array(10f,20f,30f,40f,50f,60f),out.data.shape))
    assertEquals(a.grad.get.toFloatArray.toVector,Vector(10f,20f))
    assertEquals(b.grad.get.toFloatArray.toVector,Vector(30f,40f,50f,60f))
    val empty=v(Array.emptyFloatArray,1,0,2)
    assertEquals(Variable.cat(Seq(empty,a),1).data.toFloatArray.toVector,a.data.toFloatArray.toVector)
    Variable.cat(Seq(empty,empty),1).sum().backward()
    assertEquals(empty.grad.get.shape,empty.data.shape)
    intercept[IllegalArgumentException](Variable.cat(Seq.empty,0))
    intercept[IllegalArgumentException](Variable.cat(Seq(a,b),0))
    intercept[IllegalArgumentException](Variable.cat(Seq(a),3))
  }
