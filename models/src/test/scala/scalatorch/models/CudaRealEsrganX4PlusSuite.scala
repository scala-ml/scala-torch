package scalatorch.models

import java.awt.image.BufferedImage
import java.nio.file.{Files,Path}

class CudaRealEsrganX4PlusSuite extends munit.FunSuite:
  test("Real-ESRGAN Anime 6B checkpoint is recognized as a six-block RRDB model") {
    val candidate=sys.env.get("SCALATORCH_REAL_ESRGAN_ANIME_MODEL").map(Path.of(_)).filter(Files.isRegularFile(_))
    assume(candidate.nonEmpty,"set SCALATORCH_REAL_ESRGAN_ANIME_MODEL to the official Anime 6B checkpoint")
    val weights=RealEsrganX4PlusWeights.read(candidate.get)
    assertEquals(weights.blockCount,6)
  }

  test("fast CUDA RRDB inference stays within one 8-bit level of direct convolution") {
    def setting(name: String)=sys.env.get(name).orElse(sys.props.get(name))
    assume(setting("SCALATORCH_CUDA_TESTS").contains("1"),"set SCALATORCH_CUDA_TESTS=1 for hardware tests")
    val candidate=setting("SCALATORCH_REAL_ESRGAN_MODEL").map(Path.of(_)).filter(Files.isRegularFile(_))
    assume(candidate.nonEmpty,"set SCALATORCH_REAL_ESRGAN_MODEL to the official RealESRGAN_x4plus checkpoint")
    val modelPath=candidate.get
    val source=new BufferedImage(32,32,BufferedImage.TYPE_INT_RGB)
    for y <- 0 until source.getHeight; x <- 0 until source.getWidth do
      val red=(x*17+y*3)&255
      val green=(x*5+y*11)&255
      val blue=(x*7+y*13)&255
      source.setRGB(x,y,(red<<16)|(green<<8)|blue)
    val direct=CudaRealEsrganX4Plus.upscale(source,modelPath,deviceIndex=0,maxCudaBytes=1073741824L,
      fastConvolution=false)
    val fast=CudaRealEsrganX4Plus.upscale(source,modelPath,deviceIndex=0,maxCudaBytes=1073741824L,
      fastConvolution=true)
    try
      assertEquals((fast.getWidth,fast.getHeight),(128,128))
      var maxDifference=0
      for y <- 0 until direct.getHeight; x <- 0 until direct.getWidth do
        val expected=direct.getRGB(x,y);val actual=fast.getRGB(x,y)
        maxDifference=math.max(maxDifference,math.abs(((expected>>>16)&255)-((actual>>>16)&255)))
        maxDifference=math.max(maxDifference,math.abs(((expected>>>8)&255)-((actual>>>8)&255)))
        maxDifference=math.max(maxDifference,math.abs((expected&255)-(actual&255)))
      assert(maxDifference<=1,s"fast RRDB output differs by $maxDifference 8-bit levels")
    finally
      fast.flush()
      direct.flush()
      source.flush()
  }
