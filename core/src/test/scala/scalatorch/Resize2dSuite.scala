package scalatorch

class Resize2dSuite extends munit.FunSuite:
  test("floor and exact nearest differ at noninteger ratios") {
    val x=Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(1,1,1,3)))
    assertEquals(Resize2d.nearest(x,1,5).toFloatArray.toVector,Vector(1f,1f,2f,2f,3f))
    assertEquals(Resize2d.nearest(x,1,5,NearestMode.Exact).toFloatArray.toVector,Vector(1f,1f,2f,3f,3f))
    assertEquals(Resize2d.nearest(x,1,2).toFloatArray.toVector,Vector(1f,2f))
    assertEquals(Resize2d.nearest(x,1,2,NearestMode.Exact).toFloatArray.toVector,Vector(1f,3f))
  }
  test("resizing preserves planes, strided values and empty batch shapes") {
    val x=Tensor.fromFloats((1 to 24).map(_.toFloat).toArray,Shape.from(Seq(2,2,2,3))).transpose(2,3)
    assertEquals(Resize2d.nearest(x,3,2).toFloatArray.toVector,x.toFloatArray.toVector)
    val y=Resize2d.nearest(x,1,1,NearestMode.Exact)
    assertEquals(y.toFloatArray.toVector,Vector(5f,11f,17f,23f))
    val empty=Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,2,3,4)))
    assertEquals(Resize2d.nearest(empty,7,9).shape,Shape.from(Seq(0,2,7,9)))
    intercept[IllegalArgumentException](Resize2d.nearest(x,0,1))
    intercept[IllegalArgumentException](Resize2d.nearest(x,Int.MaxValue,Int.MaxValue))
    intercept[IllegalArgumentException](Resize2d.nearest(Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,1,Long.MaxValue,1))),1,1))
  }
