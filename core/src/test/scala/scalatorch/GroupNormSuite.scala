package scalatorch

class GroupNormSuite extends munit.FunSuite:
  test("groups normalize independently with biased variance") {
    val x=Tensor.fromFloats(Array(1f,3f,10f,14f,5f,7f,20f,24f),Shape.from(Seq(2,2,2)))
    val y=GroupNormOps.forward(x,2,1.0)
    val expected=Vector(-1/math.sqrt(2),1/math.sqrt(2),-2/math.sqrt(5),2/math.sqrt(5))
    y.toFloatArray.zip(expected++expected).foreach((a,e) => assertEqualsDouble(a,e,1e-7))
    val view=x.transpose(1,2)
    assertEquals(GroupNormOps.forward(view,1).toFloatArray.toVector,
      GroupNormOps.forward(view.contiguous,1).toFloatArray.toVector)
  }
  test("singleton and large finite groups remain stable; invalid geometry rejects") {
    val x=Tensor.fromFloats(Array(Float.MaxValue,-Float.MaxValue),Shape.from(Seq(1,2)))
    assertEquals(GroupNormOps.forward(x,2).toFloatArray.toVector,Vector(0f,0f))
    GroupNormOps.forward(x,1).toFloatArray.zip(Array(1f,-1f)).foreach((a,e) => assertEqualsDouble(a,e,1e-6))
    val empty=Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,2,3)))
    assertEquals(GroupNormOps.forward(empty,1).shape,empty.shape)
    for groups <- Seq(0,3) do intercept[IllegalArgumentException](GroupNormOps.forward(x,groups))
    for epsilon <- Seq(0.0,Double.NaN,Double.PositiveInfinity) do
      intercept[IllegalArgumentException](GroupNormOps.forward(x,1,epsilon))
    intercept[IllegalArgumentException](GroupNormOps.forward(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(1,2,0))),1))
  }
