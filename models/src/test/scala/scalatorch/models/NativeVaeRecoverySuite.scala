package scalatorch.models.worker

import scalatorch.*
import java.nio.file.Files

class NativeVaeRecoverySuite extends munit.FunSuite:
  test("failed decoder inputs are preserved exactly in readable safetensors") {
    val root=Files.createTempDirectory("vae-recovery-test")
    try
      val input=Tensor.fromFloats(Array(.25f,-.5f,1f,-2f),Shape.from(Seq(1,4,1,1)))
      val path=NativeVaeRecovery.save(input,root)
      scala.util.Using.resource(scalatorch.io.SafeTensors.open(path)) { archive =>
        assertEquals(archive.readFloat("tensor").toFloatArray.toVector,input.toFloatArray.toVector)
      }
      assert(Files.exists(path))
    finally
      val files=Files.walk(root)
      try
        import scala.jdk.CollectionConverters.*
        files.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_))
      finally files.close()
  }
