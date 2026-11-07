package scalatorch.models

import scalatorch.io.PyTorchArchiveLimits

class PyTorchCompatibilitySuite extends munit.FunSuite:
  test("CRC compatibility is explicit and preserves the archive limits") {
    val base=PyTorchArchiveLimits(maxElements=17)
    assert(!PyTorchCompatibility.limits(base,Map.empty).allowMissingCrc)
    assert(!PyTorchCompatibility.limits(base,Map("SDFX_SCALATORCH_ALLOW_MISSING_CRC"->"true")).allowMissingCrc)
    val enabled=PyTorchCompatibility.limits(base,Map("SDFX_SCALATORCH_ALLOW_MISSING_CRC"->"1"))
    assert(enabled.allowMissingCrc)
    assertEquals(enabled.maxElements,17L)
  }
