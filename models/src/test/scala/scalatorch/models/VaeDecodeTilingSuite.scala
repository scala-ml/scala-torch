package scalatorch.models

class VaeDecodeTilingSuite extends munit.FunSuite:
  test("decoder tiles automatically when requested latent size exceeds the VAE tile") {
    assert(EulerImagePipeline.shouldTileDecode(96,96,512,512,8,configured=false))
    assert(!EulerImagePipeline.shouldTileDecode(64,64,512,512,8,configured=false))
    assert(EulerImagePipeline.shouldTileDecode(64,32,512,512,8,configured=true))
  }
