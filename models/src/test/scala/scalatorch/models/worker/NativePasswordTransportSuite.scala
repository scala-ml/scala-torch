package scalatorch.models.worker

class NativePasswordTransportSuite extends munit.FunSuite:
  test("VAE request password is accepted without appearing in request diagnostics") {
    val request=NativeRequest.parse("""{"id":"r1","type":"vae_tools","params":{},"image_password":"secret-test-value"}""")
    assertEquals(request.imagePassword,Some("secret-test-value"))
    assert(!request.toString.contains("secret-test-value"))
    intercept[IllegalArgumentException](NativeRequest.parse("""{"id":"r2","type":"txt2img","image_password":"x"}"""))
    intercept[IllegalArgumentException](NativeRequest.parse("""{"id":"r3","type":"vae_tools","image_password":""}"""))
    val root=java.nio.file.Files.createTempDirectory("native-password-transport")
    val scheduler=root.resolve("scheduler.json")
    java.nio.file.Files.writeString(scheduler,"{}")
    val backend=new NativeDiffusionBackend(root,scheduler)
    val error=intercept[NativeBackendError](backend.execute(request,new NativeWorkContext(()=>false,()=>true,(_,_)=>())))
    assertEquals(error.code,"no_model")
    assert(!error.getMessage.contains("secret-test-value"))
  }
