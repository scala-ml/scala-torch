package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.{Conv2d,Parameter}

class TinyAutoencoderBlockSuite extends munit.FunSuite:
  test("tiny autoencoder residual block matches Diffusers reference") {
    def conv(index: Int): Conv2d =
      val weights=Array.tabulate(36)(i=>((i%11)-5)*(.005f*(index+1)))
      val bias=Array(.02f*(index+1),-.03f*(index+1))
      Conv2d(Parameter(Tensor.fromFloats(weights,Shape.from(Seq(2,2,3,3))),false),
        Some(Parameter(Tensor.fromFloats(bias,Shape.from(Seq(2))),false)),Conv2dConfig(padding=(1,1)))
    val block=TinyAutoencoderBlock(conv(0),conv(1),conv(2))
    val input=Variable(Tensor.fromFloats(Array.tabulate(12)(i=>(i-5)/4f),Shape.from(Seq(1,2,2,3))),false)
    val actual=block(input).data.toFloatArray
    val expected=Array(0f,0f,0f,0f,0f,.052756499f,.152690694f,.404119253f,.65724808f,.90921551f,1.163396f,1.413060546f)
    actual.zip(expected).foreach((a,b)=>assert(math.abs(a-b)<1e-6,s"$a != $b"))
  }
