package scalatorch.models

import java.nio.file.Path
import scalatorch.Shape

class VaeAttentionBudgetInvestigationSuite extends munit.FunSuite:
  test("CUDA VAE default work limit admits the tested untiled shape and respects explicit limits") {
    assertEquals(scalatorch.models.worker.NativeWorkerMain.vaeAttentionLimit(134217728L,false,true),268435456L)
    assertEquals(scalatorch.models.worker.NativeWorkerMain.vaeAttentionLimit(134217728L,false,false),134217728L)
    assertEquals(scalatorch.models.worker.NativeWorkerMain.vaeAttentionLimit(12345L,true,true),12345L)
  }
  test("tiny model untiled 256px decoder exceeds configured attention work limit") {
    val configured=sys.env.get("SCALATORCH_TINY_MODEL")
    assume(configured.nonEmpty,"tiny model fixture required")
    val vae=PretrainedVae.load(Path.of(configured.get).resolve("vae"))
    val error=intercept[IllegalArgumentException](vae.decoder.preflight(Shape.from(Seq(1L,4L,128L,128L))))
    assert(error.getMessage.contains("VAE attention score budget exceeded"))
    vae.decoder.preflight(Shape.from(Seq(1L,4L,16L,16L)))
  }
