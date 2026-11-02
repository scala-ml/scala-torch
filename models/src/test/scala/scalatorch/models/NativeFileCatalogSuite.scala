package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scala.jdk.CollectionConverters.*

class NativeFileCatalogSuite extends munit.FunSuite:
  test("catalog filters extensions deduplicates roots and collapses directory models") {
    val root=Files.createTempDirectory("native-catalog")
    try
      Files.write(root.resolve("B.SAFETENSORS"),Array[Byte](1,2))
      Files.writeString(root.resolve("notes.txt"),"ignored")
      val sub=Files.createDirectory(root.resolve("sub"))
      Files.write(sub.resolve("a.pt"),Array[Byte](3))
      val hidden=Files.createDirectory(root.resolve(".hidden"))
      Files.write(hidden.resolve("x.pt"),Array[Byte](4))
      val model=Files.createDirectory(root.resolve("control"))
      Files.writeString(model.resolve("config.json"),"{}")
      Files.write(model.resolve("weights.safetensors"),Array[Byte](5))
      def names(kind: String)=NativeFileCatalog.scan(kind,Vector(root.toString,root.toString),() => ())
        .map(_.fields("name").asInstanceOf[NativeJson.Str].value)
      assertEquals(names("controlnet"),Vector("B","control","sub/a"))
      assertEquals(names("lora"),Vector("B","control/weights","sub/a"))
      assertEquals(NativeFileCatalog.scan("vae",Vector("",root.resolve("missing").toString),() => ()),Vector.empty)
      intercept[IllegalArgumentException](names("unknown"))
      intercept[java.util.concurrent.CancellationException](NativeFileCatalog.scan("lora",Vector(root.toString),() => throw new java.util.concurrent.CancellationException()))
    finally
      val paths=Files.walk(root)
      try paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.delete(_)) finally paths.close()
  }
