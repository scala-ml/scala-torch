package scalatorch.models.worker

import java.awt.image.BufferedImage
import munit.FunSuite
import scalatorch.models.worker.NativeJson.*

final class NativeImg2ImgHiresSuite extends FunSuite:
  test("defaults and ratio target match NUI img2img Hires Fix") {
    val config=NativeImg2ImgHires.parse(Obj(Map("enabled"->Bool(true),"upscaler"->Str("Lanczos"))))
    assertEquals(config.ratio,2.0)
    assertEquals(config.upscaler,"Lanczos")
    assertEquals((config.steps,config.prompt,config.negativePrompt,config.cfg),(0,"","",0.0))
    assertEquals((config.sampler,config.scheduler),("Use same sampler","Use same scheduler"))
    assertEquals(config.target(73,65),(144,128))
    assertEquals(config.target(16,16),(64,64))
  }

  test("explicit target dimensions are floored to multiples of eight with a 64 pixel minimum") {
    val config=NativeImg2ImgHires.parse(Obj(Map("width"->Number(136),"height"->Number(72),
      "upscaler"->Str("Lanczos"))))
    assertEquals(config.target(512,512),(136,72))
  }

  test("nearest pixel upscale maps each destination pixel to its source cell") {
    val source=new BufferedImage(2,2,BufferedImage.TYPE_INT_RGB)
    source.setRGB(0,0,0xff110000);source.setRGB(1,0,0xff220000)
    source.setRGB(0,1,0xff330000);source.setRGB(1,1,0xff440000)
    val resized=NativeImg2ImgHires.upscale(source,4,4,"Nearest",() => false)
    assertEquals(resized.getRGB(0,0),source.getRGB(0,0))
    assertEquals(resized.getRGB(1,0),source.getRGB(0,0))
    assertEquals(resized.getRGB(2,0),source.getRGB(1,0))
    assertEquals(resized.getRGB(3,3),source.getRGB(1,1))
  }

  test("NUI model-upscaler default is rejected explicitly while built-in zero-denoise is accepted") {
    val config=NativeImg2ImgHires.parse(Obj(Map("denoise"->Number(0),"upscaler"->Str("Lanczos"))))
    assertEquals(config.denoise,0.0)
    val unavailable=intercept[NativeBackendError](NativeImg2ImgHires.parse(Obj(Map.empty)))
    assertEquals(unavailable.code,"unsupported")
  }

  test("catalog Anime 6B model name resolves to native RRDB Hires upscaling") {
    val model="RealESRGAN_x4plus_anime_6B"
    val path="C:/models/RealESRGAN_x4plus_anime_6B.pth"
    val config=NativeImg2ImgHires.parse(Obj(Map("upscaler"->Str(model))),Map(model->path))
    assertEquals(config.upscaler,"R-ESRGAN 4x+")
    assertEquals(config.upscalerPath,Some(path))
  }

  test("invalid ratios, dimensions and config types are rejected") {
    intercept[NativeBackendError](NativeImg2ImgHires.parse(Obj(Map("ratio"->Number(0)))))
    intercept[NativeBackendError](NativeImg2ImgHires.parse(Obj(Map("ratio"->Number(5)))))
    intercept[NativeBackendError](NativeImg2ImgHires.parse(Obj(Map("width"->Number(64)))))
    intercept[NativeBackendError](NativeImg2ImgHires.parse(Obj(Map("upscaler"->Str("Lanczos"),"sampler"->Str("unknown-sampler")))))
    intercept[NativeBackendError](NativeImg2ImgHires.parse(Str("bad")))
  }
