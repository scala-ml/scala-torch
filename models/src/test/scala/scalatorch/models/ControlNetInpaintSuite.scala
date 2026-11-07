package scalatorch.models

import scalatorch.{Shape,Tensor}

class ControlNetInpaintSuite extends munit.FunSuite:
  test("ControlNet takes the latent channels of each inpaint batch item") {
    val sample=Tensor.fromFloats(Array.tabulate(2*9*2)(_.toFloat),Shape.from(Seq(2L,9L,1L,2L)))
    val latent=UNetExecution.controlInput(sample)
    assertEquals(latent.shape,Shape.from(Seq(2L,4L,1L,2L)))
    assertEquals(latent.toFloatArray.toVector,(0 until 8).map(_.toFloat).toVector ++ (18 until 26).map(_.toFloat))
    assertEquals(UNetExecution.controlInput(latent).toFloatArray.toVector,latent.toFloatArray.toVector)
    intercept[IllegalArgumentException] {
      UNetExecution.controlInput(Tensor.fromFloats(new Array[Float](5),Shape.from(Seq(1L,5L,1L,1L))))
    }
  }
  test("standard SD1 ControlNet accepts four- and nine-channel compatible bases only") {
    val config=SingleFileSd1Config.unetConfig
    val control=ControlNetConfig(config,3,Vector(16,32,96,256),"rgb",false)
    control.requireCompatible(config)
    control.requireCompatible(config.copy(inChannels=9))
    intercept[IllegalArgumentException](control.requireCompatible(config.copy(inChannels=5)))
    intercept[IllegalArgumentException](control.requireCompatible(config.copy(inChannels=9,contextWidth=1024)))
  }
