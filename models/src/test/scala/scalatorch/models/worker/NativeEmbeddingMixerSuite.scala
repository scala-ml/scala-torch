package scalatorch.models.worker

class NativeEmbeddingMixerSuite extends munit.FunSuite:
  test("preset transforms are applied before combination and zero filtering") {
    val values=Map("clip_l"->Vector(Vector(-0.2f,0.1f,0.2f,0.3f),Vector(0.2f,-0.1f,0.4f,0.1f)))
    val binary=NativeEmbeddingMixer.mix(Vector(values->1.0),false,true,Some("=(1*(v>=0)-1*(v<0))/50"))
    assertEquals(binary("clip_l"),Vector(Vector(0f,0f,0.04f,0.04f)))
    val comb=NativeEmbeddingMixer.mix(Vector(values->1.0),false,false,Some("=v*((i%2)==0)"))
    assertEquals(comb("clip_l").head,Vector(-0.2f,0f,0.2f,0f))
    val digitize=NativeEmbeddingMixer.mix(Vector(values->1.0),false,false,Some("=math.ceil(v*8)/8"))
    assertEquals(digitize("clip_l").head,Vector(-0.125f,0.125f,0.25f,0.375f))
    val cropped=NativeEmbeddingMixer.mix(Vector(values->1.0),false,false,Some("=v*(i>=maxi//2)"))
    assertEquals(java.lang.Float.floatToRawIntBits(cropped("clip_l").head.head),java.lang.Float.floatToRawIntBits(-0.0f))
    intercept[IllegalArgumentException](NativeEmbeddingMixer.mix(Vector(values->1.0),false,false,Some("__import__('os')")))
  }
  test("mixes weighted vectors with padding, concatenation and paired SDXL zero removal") {
    val first=Map("clip_l"->Vector(Vector(1f,2f),Vector(0f,0f)),"clip_g"->Vector(Vector(0f,0f,0f),Vector(1f,0f,0f)))
    val second=Map("clip_l"->Vector(Vector(2f,4f)),"clip_g"->Vector(Vector(0f,1f,0f)))
    val result=NativeEmbeddingMixer.mix(Vector(first->1.0,second->0.5),concat=false,combine=false)
    assertEquals(result("clip_l"),Vector(Vector(2f,4f),Vector(0f,0f)))
    assertEquals(result("clip_g"),Vector(Vector(0f,0.5f,0f),Vector(1f,0f,0f)))
    val combined=NativeEmbeddingMixer.mix(Vector(first->1.0,second->0.5),concat=true,combine=true)
    assertEquals(combined("clip_l"),Vector(Vector(2f,4f)))
    assertEquals(combined("clip_g"),Vector(Vector(1f,0.5f,0f)))
  }
  test("rejects incompatible encoder sets and zero results") {
    val left=Map("clip_l"->Vector(Vector(1f)))
    val pair=left.updated("clip_g",Vector(Vector(1f)))
    intercept[IllegalArgumentException](NativeEmbeddingMixer.mix(Vector(left->1.0,pair->1.0),false,false))
    intercept[IllegalArgumentException](NativeEmbeddingMixer.mix(Vector(left->1.0,left-> -1.0),false,false))
  }
