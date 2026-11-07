package scalatorch.models.worker

import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,FileAlreadyExistsException}

class NativeKeywordCatalogSuite extends munit.FunSuite:
  private val bytes="abc,red|blue,style.safetensors\n".getBytes(UTF_8)
  private val hash=java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes))
  private def context(publish: () => Boolean=() => true)=new NativeWorkContext(() => false,publish,(_,_) => ())
  private def input()=new ByteArrayInputStream(bytes)
  test("confirmed catalog installation verifies bytes and refuses replacement") {
    val dir=Files.createTempDirectory("native-catalog")
    try
      val result=NativeKeywordCatalog.install(dir,true,() => input(),bytes.length,hash,context())
      assertEquals(Files.readAllBytes(result).toVector,bytes.toVector)
      intercept[FileAlreadyExistsException](NativeKeywordCatalog.install(dir,true,() => input(),bytes.length,hash,context()))
      assertEquals(Files.readAllBytes(result).toVector,bytes.toVector)
    finally
      Files.deleteIfExists(dir.resolve("lora-keyword.txt"));Files.delete(dir)
  }
  test("failed verification or publication leaves no output or temporary file") {
    val dir=Files.createTempDirectory("native-catalog-invalid")
    def names()=scala.util.Using.resource(Files.list(dir))(_.count())
    try
      intercept[IllegalArgumentException](NativeKeywordCatalog.install(dir,false,() => input(),bytes.length,hash,context()))
      intercept[IllegalArgumentException](NativeKeywordCatalog.install(dir,true,() => input(),bytes.length,"0"*64,context()))
      intercept[IllegalArgumentException](NativeKeywordCatalog.install(dir,true,() => input(),bytes.length-1,hash,context()))
      intercept[java.util.concurrent.CancellationException](NativeKeywordCatalog.install(dir,true,() => input(),bytes.length,hash,context(() => false)))
      assertEquals(names(),0L)
      val racing=context(() => { Files.writeString(dir.resolve("lora-keyword.txt"),"other");true })
      intercept[FileAlreadyExistsException](NativeKeywordCatalog.install(dir,true,() => input(),bytes.length,hash,racing))
      assertEquals(Files.readString(dir.resolve("lora-keyword.txt")),"other")
      assertEquals(names(),1L)
    finally
      Files.deleteIfExists(dir.resolve("lora-keyword.txt"));Files.delete(dir)
  }
  test("the NUI action requires explicit confirmation before downloading") {
    import NativeJson.*
    val dir=Files.createTempDirectory("native-catalog-action")
    try
      intercept[IllegalArgumentException](NativeKeywordMappings.execute(Map("action"->Str("install_upstream_lora_catalog"),
        "mapping_dir"->Str(dir.toString),"confirmed"->Bool(false)),dir,() => "a"*64,context()))
      assert(!Files.exists(dir.resolve("lora-keyword.txt")))
    finally Files.delete(dir)
  }
  test("pinned upstream catalog downloads through the native NUI action") {
    assume(sys.env.get("SCALATORCH_TEST_CATALOG_DOWNLOAD").contains("1"),"network catalog test is opt-in")
    import NativeJson.*
    val dir=Files.createTempDirectory("native-catalog-live")
    try
      val result=NativeKeywordMappings.execute(Map("action"->Str("install_upstream_lora_catalog"),
        "mapping_dir"->Str(dir.toString),"confirmed"->Bool(true)),dir,() => "a"*64,context())
      val path=java.nio.file.Path.of(result("path").asInstanceOf[Str].value)
      assertEquals(path,dir.resolve("lora-keyword.txt"))
      assertEquals(Files.size(path),31395702L)
      val model=dir.resolve("sample.safetensors")
      Files.write(model,Array[Byte](1,2,3))
      val checked=NativeKeywordMappings.execute(Map("kind"->Str("lora"),"action"->Str("check"),
        "model_path"->Str(model.toString),"mapping_dir"->Str(dir.toString)),dir,() => "a"*64,context())
      assertEquals(checked("default_catalog_available"),Bool(true))
    finally
      Files.deleteIfExists(dir.resolve("sample.safetensors"));Files.deleteIfExists(dir.resolve("lora-keyword.txt"));Files.delete(dir)
  }
