package scalatorch.autograd

import scalatorch.*

class NarrowSuite extends munit.FunSuite:
  test("overlapping selections accumulate gradients in original logical coordinates") {
    val x=Variable(Tensor.fromFloats((1 to 6).map(_.toFloat).toArray,Shape.from(Seq(2,3))),true)
    val y=x.transpose(0,1).narrow(0,1,2)
    assertEquals(y.data.toFloatArray.toVector,Vector(2f,5f,3f,6f))
    (y.sum()+x.narrow(1,0,2).sum()).backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(1f,2f,1f,1f,2f,1f))
  }
  test("empty selections have zero gradients and overflow-sized bounds reject") {
    val x=Variable(Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(3))),true)
    x.narrow(0,3,0).sum().backward()
    assertEquals(x.grad.get.toFloatArray.toVector,Vector(0f,0f,0f))
    intercept[IllegalArgumentException](x.narrow(0,1,Long.MaxValue))
    intercept[IllegalArgumentException](x.narrow(1,0,1))
    intercept[IllegalArgumentException](x.narrow(0,-1,1))
  }
