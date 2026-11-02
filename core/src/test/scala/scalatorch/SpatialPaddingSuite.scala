package scalatorch

class SpatialPaddingSuite extends munit.FunSuite:
  private val input=Tensor.fromFloats(Array(1f,2f,3f,4f),Shape.from(Seq(1,1,2,2)))
  test("asymmetric circular columns and zero rows preserve logical noncontiguous order") {
    val out=SpatialPadding.forward(input.transpose(2,3),Padding2d(1,0,1,0,horizontal=PaddingMode.Circular))
    assertEquals(out.shape,Shape.from(Seq(1,1,3,3)))
    assertEquals(out.toFloatArray.toVector,Vector(0f,0f,0f,3f,1f,3f,4f,2f,4f))
  }
  test("circular corners wrap both axes and padding limits reject before allocation") {
    val c=Padding2d(1,1,1,1,PaddingMode.Circular,PaddingMode.Circular)
    assertEquals(SpatialPadding.forward(input,c).toFloatArray.toVector,
      Vector(4f,3f,4f,3f,2f,1f,2f,1f,4f,3f,4f,3f,2f,1f,2f,1f))
    intercept[IllegalArgumentException](Padding2d(-1,0,0,0))
    intercept[IllegalArgumentException](SpatialPadding.forward(input,c.copy(left=3)))
    intercept[IllegalArgumentException](SpatialPadding.forward(input,c.copy(left=Int.MaxValue,horizontal=PaddingMode.Zero)))
    val empty=Tensor.fromFloats(Array.emptyFloatArray,Shape.from(Seq(0,1,2,2)))
    assertEquals(SpatialPadding.forward(empty,c).shape,Shape.from(Seq(0,1,4,4)))
  }
