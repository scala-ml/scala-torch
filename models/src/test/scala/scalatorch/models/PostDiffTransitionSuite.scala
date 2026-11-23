package scalatorch.models

import scalatorch.*

class PostDiffTransitionSuite extends munit.FunSuite:
  test("bilinear enlargement and Euler noise preserve separate batch members") {
    val clean=Tensor.fromFloats(Array(1f,2f,3f,4f, 10f,20f,30f,40f),Shape.from(Seq(2,1,2,2)))
    val noise=Tensor.fromFloats(Array.fill(16)(2f) ++ Array.fill(16)(-3f),Shape.from(Seq(2,1,4,4)))
    val output=PostDiffTransition.euler(clean,noise,0.5)
    assertEquals(output.shape.dims,Seq(2L,1L,4L,4L))
    val values=output.toFloatArray
    assertEqualsDouble(values(0).toDouble,2.0,1e-6)
    assertEqualsDouble(values(5).toDouble,2.75,1e-6)
    assertEqualsDouble(values(15).toDouble,5.0,1e-6)
    assertEqualsDouble(values(16).toDouble,8.5,1e-6)
    assertEqualsDouble(values(31).toDouble,38.5,1e-6)
  }
  test("Euler and VP transitions obey their respective noise coordinates") {
    val clean=Tensor.fromFloats(Array(0.75f),Shape.from(Seq(1,1,1,1)))
    val noise=Tensor.fromFloats(Array(-0.4f),Shape.from(Seq(1,1,1,1)))
    for sigma <- Seq(0.0,0.2,1.0,3.0) do
      val euler=PostDiffTransition.euler(clean,noise,sigma).toFloatArray.head
      val vp=PostDiffTransition.vp(clean,noise,sigma).toFloatArray.head
      assertEqualsDouble(euler.toDouble,0.75-0.4*sigma,1e-6)
      assertEqualsDouble(vp.toDouble,euler.toDouble/math.hypot(1,sigma),1e-6)
  }
  test("rectangular bilinear upsampling matches PyTorch align_corners=false") {
    // torch 2.7.1 F.interpolate on a 1x1x2x3 tensor, size=(4,6), align_corners=false.
    val clean=Tensor.fromFloats(Array(0f,1f,2f,3f,4f,5f),Shape.from(Seq(1,1,2,3)))
    val noise=Tensor.fromFloats(Array.fill(24)(0f),Shape.from(Seq(1,1,4,6)))
    val expected=Vector(0f,.25f,.75f,1.25f,1.75f,2f,.75f,1f,1.5f,2f,2.5f,2.75f,
      2.25f,2.5f,3f,3.5f,4f,4.25f,3f,3.25f,3.75f,4.25f,4.75f,5f)
    assertEquals(PostDiffTransition.euler(clean,noise,1).toFloatArray.toVector,expected)
  }
  test("invalid noise shapes and sigmas are rejected") {
    val clean=Tensor.fromFloats(Array(1f),Shape.from(Seq(1,1,1,1)))
    val wrong=Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(2,1,1,1)))
    intercept[IllegalArgumentException](PostDiffTransition.euler(clean,wrong,1))
    for sigma <- Seq(-1.0,Double.NaN,Double.PositiveInfinity) do
      intercept[IllegalArgumentException](PostDiffTransition.euler(clean,clean,sigma))
  }
  test("transition step is aligned to the predicted clean latent at the same sigma") {
    val plan=PostDiffTransition.plan(128,96,20,0.75,0.20,8)
    assertEquals(plan.lowWidth,96)
    assertEquals(plan.lowHeight,72)
    assertEquals(plan.transitionIndex,3)
    assertEquals(plan.lowPredictionCount,4)
    intercept[IllegalArgumentException](PostDiffTransition.plan(128,96,20,0.0,0.2,8))
    intercept[IllegalArgumentException](PostDiffTransition.plan(128,96,20,0.75,1.0,8))
    intercept[IllegalArgumentException](PostDiffTransition.plan(128,96,20,0.70,0.2,8))
  }
