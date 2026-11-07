package scalatorch.models.worker

import java.nio.{ByteBuffer,ByteOrder}
import java.nio.file.{Files,Path}
import scalatorch.models.*

class NativeDdimSamplerTrajectorySuite extends munit.FunSuite:
  test("DDIM leading trajectory matches Diffusers") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"tiny model and CUDA required")
    val root=Path.of(configured.get).getParent
    val tok=ClipFastTokenizer.load(root.resolve("tokenizer/vocab.json"),root.resolve("tokenizer/merges.txt")).withNuiFallbackNormalization
    val pipe=new EulerImagePipeline(tok,PretrainedClip.load(root.resolve("text_encoder")),PretrainedUNet.load(root.resolve("unet")),
      PretrainedVae.load(root.resolve("vae")),vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
    val req=DiffusionImageRequest("a red cat","",64,64,42,guidanceScale=7.5f,torchCpuNoise=true)
    val schedule=DdimSchedule(DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true),4,DiffusionTimestepSpacing.Leading)
    val trace=scala.collection.mutable.ArrayBuffer.empty[Array[Float]]
    pipe.generateDdim(req,schedule,false,1f,onProgress=p=>trace+=p.latents.toFloatArray)
    val stream=this.getClass.getResourceAsStream("/ddim-sampler-trace.f32")
    val raw=try stream.readAllBytes() finally stream.close()
    val bytes=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
    trace.zipWithIndex.foreach { (arr,index) =>
      val err=arr.map(x=>math.abs(x.toDouble-bytes.getFloat())).max
      println(s"DDIM step ${index+1} max latent error $err")
      assert(err<0.0001,s"step=$index error=$err")
    }
  }
