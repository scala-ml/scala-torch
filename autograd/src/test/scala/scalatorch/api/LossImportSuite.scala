package example {
  import scalatorch.*
  import scalatorch.autograd.*

  object ImportedLoss {
    def value: Float = {
      val reduction: LossReduction=LossReduction.Mean
      val x=Variable(Tensor.fromFloats(Array(0f,0f),Shape.from(Seq(2L))),true)
      x.crossEntropy(Tensor.fromLongs(Array(0L),Shape.from(Seq.empty)),reduction).data.toFloatArray.head
    }
  }
}

package scalatorch.autograd {
  class LossImportSuite extends munit.FunSuite {
    test("existing external core and autograd wildcard imports preserve loss settings") {
      assertEqualsDouble(example.ImportedLoss.value,math.log(2),1e-7)
    }
  }
}
