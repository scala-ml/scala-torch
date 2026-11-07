package scalatorch.models.worker

import scalatorch.{Shape,Tensor}

class NativeHiresSuite extends munit.FunSuite:
  test("all latent interpolation modes preserve constants and output NCHW geometry") {
    val input=Tensor.fromFloats(Array.fill(4)(2.5f),Shape.from(Seq(1L,1L,2L,2L)))
    for mode <- NativeHires.LatentModes do
      val output=NativeHires.resize(input,4,6,mode)
      assertEquals(output.shape,Shape.from(Seq(1L,1L,6L,4L)),mode)
      assert(output.toFloatArray.forall(x => math.abs(x-2.5f)<1e-5f),s"$mode")
  }

  test("nearest and nearest-exact retain their distinct coordinate conventions") {
    val input=Tensor.fromFloats(Array(0f,1f,2f,3f),Shape.from(Seq(1L,1L,2L,2L)))
    val nearest=NativeHires.resize(input,3,3,"Latent (nearest)").toFloatArray
    val exact=NativeHires.resize(input,3,3,"Latent (nearest-exact)").toFloatArray
    assertEquals(nearest(4),0f,"nearest center")
    assertEquals(exact(4),3f,"nearest-exact center")
  }

  test("all six interpolation modes match checked-in PyTorch golden values") {
    val input=Tensor.fromFloats((0 until 12).map(_.toFloat).toArray,Shape.from(Seq(1L,1L,3L,4L)))
    val expected=Map(
      "Latent"->Vector(.166667f,1.5f,2.833333f,1.766667f,3.1f,4.433333f,4.166667f,5.5f,6.833333f,6.566668f,7.900001f,9.233335f,8.166666f,9.5f,10.833333f),
      "Latent (antialiased)"->Vector(.3f,1.5f,2.7f,1.9f,3.1f,4.3f,4.3f,5.5f,6.7f,6.700001f,7.900001f,9.1f,8.3f,9.5f,10.700001f),
      "Latent (bicubic)"->Vector(-.257843f,1.116001f,2.489844f,1.390157f,2.763999f,4.137843f,4.126158f,5.5f,6.873844f,6.862159f,8.236001f,9.609847f,8.510161f,9.884005f,11.257851f),
      "Latent (bicubic antialiased)"->Vector(-.111956f,1.198113f,2.508181f,1.413813f,2.723881f,4.033950f,4.189931f,5.5f,6.810068f,6.966052f,8.276120f,9.586189f,8.491818f,9.801888f,11.111956f),
      "Latent (nearest)"->Vector(0f,1f,2f,0f,1f,2f,4f,5f,6f,4f,5f,6f,8f,9f,10f),
      "Latent (nearest-exact)"->Vector(0f,2f,3f,0f,2f,3f,4f,6f,7f,8f,10f,11f,8f,10f,11f))
    for mode <- NativeHires.LatentModes do
      val actual=NativeHires.resize(input,3,5,mode).toFloatArray.toVector
      assertEquals(actual.size,expected(mode).size,mode)
      val maxError=actual.zip(expected(mode)).map((a,b)=>math.abs(a-b)).max
      assert(maxError<2e-5f,s"$mode max absolute error $maxError")
  }

  test("target and upscale sizes follow NUI integer truncation and centered crop rules") {
    assertEquals(NativeHires.targetSize(65,49,2.0,0,0),(128,96),"scaled target")
    assertEquals(NativeHires.targetSize(65,49,2.0,96,96),(96,96),"explicit target")
    assertEquals(NativeHires.upscaleSize(65,49,96,96,cover=true),(120,96),"covering intermediate")
    assertEquals(NativeHires.upscaleSize(65,49,128,96,cover=false),(128,96),"scale-derived target")
    assertEquals(NativeHires.centerCrop(16,12,12,12),(2,0),"crop offset")
  }

  test("latent crop is centered and preserves channel and batch layout") {
    val input=Tensor.fromFloats((0 until 16).map(_.toFloat).toArray,Shape.from(Seq(1L,1L,4L,4L)))
    val (left,top)=NativeHires.centerCrop(4,4,2,2)
    assertEquals((left,top),(1,1),"crop origin")
    assertEquals(NativeHires.crop(input,2,2,left,top).toFloatArray.toVector,Vector(5f,6f,9f,10f),"cropped values")
  }

  test("request parser accepts Hires overrides and rejects malformed options") {
    val active=NativeJson.Obj(Map("enabled"->NativeJson.Bool(true),"scale"->NativeJson.Number(2),
      "upscaler"->NativeJson.Str("Latent (bicubic)"),"second_pass_steps"->NativeJson.Number(8),
      "denoising_strength"->NativeJson.Number(BigDecimal("0.4"))))
    assertEquals(NativeHires.parse(Some(active)),Some(NativeHires.Settings(2.0,"Latent (bicubic)",8,0,0,0.4)))
    val fields=active.asInstanceOf[NativeJson.Obj].fields
    val overridden=NativeHires.parse(Some(NativeJson.Obj(fields++Map("prompt"->NativeJson.Str("red"),
      "negative_prompt"->NativeJson.Str("blur"),"cfg_scale"->NativeJson.Number(0),
      "sampler_name"->NativeJson.Str("Euler"),"scheduler"->NativeJson.Str("Uniform"))))).get
    assertEquals(overridden.prompt,"red")
    assertEquals(overridden.negativePrompt,"blur")
    assertEquals(overridden.cfgScale,Some(0f))
    assertEquals(overridden.sampler,Some("Euler"))
    assertEquals(overridden.scheduler,Some("Uniform"))
    for invalid <- Vector(
      NativeJson.Obj(fields.updated("upscaler",NativeJson.Str("unavailable-upscaler"))),
      NativeJson.Obj(fields.updated("mystery",NativeJson.Bool(true)))) do
      intercept[NativeBackendError](NativeHires.parse(Some(invalid)))
  }

  test("hires step schedule preserves requested active steps at strength boundaries") {
    assertEquals(NativeHires.steps(10,0.55),(19,0.55,10),"normal strength")
    assertEquals(NativeHires.steps(10,0.0),(10,0.1,1),"zero strength")
    assertEquals(NativeHires.steps(10,1.0),(11,1.0,11),"full strength")
  }

  test("unknown and image upscalers are rejected rather than treated as latent") {
    intercept[NativeBackendError](NativeHires.resize(
      Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L,1L,1L))),2,2,"unavailable-upscaler"))
    intercept[NativeBackendError](NativeHires.resize(
      Tensor.fromFloats(Array(1f),Shape.from(Seq(1L,1L,1L,1L))),2,2,"not-a-mode"))
  }
