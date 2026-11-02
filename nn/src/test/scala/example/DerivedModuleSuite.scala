package example

import scalatorch.*
import scalatorch.nn.*

class DerivedModuleSuite extends munit.FunSuite:
  case class Stack(layers: Vector[Linear], extra: Option[Parameter]) derives Module
  test("external case classes derive nested vector and optional parameter trees") {
    val model = Stack(Vector(Linear.init(2, 3, seed = 1), Linear.init(3, 1, seed = 2, bias = false)), None)
    assertEquals(Module.namedParameters(model).map(_._1), Vector("layers.0.weight", "layers.0.bias", "layers.1.weight"))
    assertEquals(Module.parameters(model).size, 3)
  }
