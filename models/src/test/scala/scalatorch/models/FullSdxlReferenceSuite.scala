package scalatorch.models

import scalatorch.*
import scalatorch.ffm.CudaSession
import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scala.util.Using

/** Optional numeric comparison with independently produced full-model fixtures. */
class FullSdxlReferenceSuite extends munit.FunSuite:
  private def paths(): (Path,Path) =
    val root=sys.props.get("scalatorch.full.sdxl.dir")
    val fixtures=sys.props.get("scalatorch.full.sdxl.reference")
    assume(root.nonEmpty && fixtures.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),
      "full SDXL, independent fixtures and CUDA required")
    (Path.of(root.get),Path.of(fixtures.get))
  private def tensor(dir: Path,name: String,dims: Long*): Tensor[Float] =
    val bytes=ByteBuffer.wrap(Files.readAllBytes(dir.resolve(name+".bin"))).order(ByteOrder.LITTLE_ENDIAN)
    Tensor.fromFloats(Array.fill(bytes.remaining()/4)(bytes.getFloat()),Shape.from(dims))
  private def compare(actual: Tensor[Float],expected: Tensor[Float]): Unit =
    assertEquals(actual.shape,expected.shape)
    val differences=actual.toFloatArray.zip(expected.toFloatArray).map((a,b)=>math.abs(a.toDouble-b))
    val max=differences.max
    val mean=differences.sum/differences.length
    val index=differences.indexOf(max)
    println(s"full SDXL reference: max error=$max, mean error=$mean, index=$index, expected=${expected.toFloatArray(index)}, actual=${actual.toFloatArray(index)}")
    assert(max<0.001,s"maximum error $max")
  test("full SDXL prompt conditioning agrees with NUI reference") {
    val (root,ref)=paths()
    val model=new SdxlTextConditioning(ClipFastTokenizer.loadDirectory(root.resolve("tokenizer")),
      PretrainedClip.load(root.resolve("text_encoder")),ClipFastTokenizer.loadDirectory(root.resolve("tokenizer_2")),
      PretrainedClip.load(root.resolve("text_encoder_2")))
    val output=model.encode("a photograph of a red apple on a wooden table","")
    compare(output.pooled,tensor(ref,"prompt-pooled",1,1280))
    compare(output.positive,tensor(ref,"prompt-positive",1,77,2048))
  }
  test("full SDXL VAE CUDA decode agrees with independent CPU reference") {
    val (root,ref)=paths()
    val vae=PretrainedVae.load(root.resolve("vae"))
    val input=tensor(ref,"sample",1,4,8,8)
    compare(VaeExecution.Cuda(offloadWeights=true).decode(vae.decoder,input,()=>false),
      tensor(ref,"vae-output",1,3,64,64))
  }
  test("full SDXL U-Net CUDA agrees with independent CPU reference") {
    val (root,ref)=paths()
    val unet=PretrainedUNet.load(root.resolve("unet"),maxParameters=3000000000L)
    val sample=tensor(ref,"sample",1,4,8,8)
    val context=tensor(ref,"context",1,8,2048)
    val added=DiffusionAddedConditioning(tensor(ref,"pooled",1,1280),tensor(ref,"ids",1,6))
    val time=Tensor.fromFloats(Array(500f),Shape.from(Seq(1L)))
    CudaSession.withDevice(maxBytes=1073741824L) { session =>
      Using.Manager { use =>
        val model=use(CudaUNet.upload(session,unet,offloadWeights=true))
        val x=use(session.upload(sample)); val c=use(session.upload(context))
        val output=use(model(x,time,c,Some(added)))
        compare(output.toCpu,tensor(ref,"unet-output",1,4,8,8))
      }.get
    }
  }
