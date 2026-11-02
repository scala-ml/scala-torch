package scalatorch.nn

import scalatorch.*
import scalatorch.io.SafeTensors
import scalatorch.autograd.Variable
import java.nio.file.Files
import scala.util.Using

class SafeWeightsSuite extends munit.FunSuite:
  test("upstream safetensors weights load directly into a derived Scala layer") {
    val path = Files.createTempFile("scala-torch-safeweights-", ".safetensors")
    try
      Using.resource(getClass.getResourceAsStream("/safetensors/upstream-0.6.2.safetensors")) { input =>
        assert(input != null)
        Files.copy(input, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      }
      val layer = Linear.init(3, 2, seed = 1)
      Using.resource(SafeTensors.open(path)) { file =>
        val state = Module.namedParameters(layer).map((name, _) => name -> file.readFloat(name)).toMap
        Module.loadStateDict(layer, state)
      }
      val output = layer(Variable(Tensor.fromFloats(Array(1f, 2f, 3f), Shape.from(Seq(1L, 3L))), false))
      assertEquals(output.data.toFloatArray.toSeq, Seq(24f, 52f))
    finally Files.deleteIfExists(path)
  }
