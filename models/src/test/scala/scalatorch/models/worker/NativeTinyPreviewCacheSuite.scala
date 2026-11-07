package scalatorch.models.worker

import java.nio.file.{Files,Path,StandardCopyOption}
import scalatorch.{Shape,Tensor}
import scalatorch.models.VaeExecution

class NativeTinyPreviewCacheSuite extends munit.FunSuite:
  test("changed or damaged TAESD files do not reuse cached weights") {
    val source=Vector(Path.of(".superpowers/models/taesd/diffusion_pytorch_model.safetensors"),
      Path.of("../.superpowers/models/taesd/diffusion_pytorch_model.safetensors")).find(Files.isRegularFile(_))
    assume(source.nonEmpty,"trained TAESD reference weights required")
    val dir=Files.createTempDirectory("native-taesd-cache")
    val weights=dir.resolve("diffusion_pytorch_model.safetensors")
    Files.copy(source.get,weights)
    val preview=new NativeTinyPreview(Some(dir),None,VaeExecution.Cuda(offloadWeights=true))
    val latent=Tensor.fromFloats(Array.tabulate(24)(i=>math.sin(i*.37).toFloat),Shape.from(Seq(1,4,2,3)))
    val first=preview.render(latent,false,()=>())
    assert(first.nonEmpty)
    Files.write(weights,Array[Byte](1,2,3))
    intercept[Exception](preview.render(latent,false,()=>()))
    Files.copy(source.get,weights,StandardCopyOption.REPLACE_EXISTING)
    assertEquals(preview.render(latent,false,()=>()),first)
    var checks=0
    intercept[java.util.concurrent.CancellationException] {
      preview.render(latent,false,()=>{checks+=1;if checks==12 then throw new java.util.concurrent.CancellationException()})
    }
    assertEquals(preview.render(latent,false,()=>()),first)
  }
