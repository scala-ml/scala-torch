package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scalatorch.models.{PretrainedClip,PyTorchEmbeddingFixture,UNetExecution,VaeExecution}
import NativeJson.*

class NativeBinEmbeddingSuite extends munit.FunSuite:
  test("BIN textual inversion changes native generation without warnings") {
    val configured=sys.props.get("scalatorch.clip.dir")
    assume(configured.nonEmpty,"tiny SD model required")
    val root=Path.of(configured.get).getParent
    val dir=Files.createTempDirectory("native-bin-embedding")
    val scheduler=dir.resolve("scheduler.json")
    Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
    val width=PretrainedClip.load(root.resolve("text_encoder")).tokenEmbedding.weight.data.shape.dims(1).toInt
    val embedding=dir.resolve("legacybin.bin")
    PyTorchEmbeddingFixture.write(embedding,"string_to_param",("*",Vector.fill(width)(-0.75f),Vector(1,width)))
    val devices=if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then Vector(false,true) else Vector(false)
    devices.foreach { cuda =>
      val backend=if cuda then new NativeDiffusionBackend(root,scheduler,
        vaeExecution=VaeExecution.Cuda(offloadWeights=true),unetExecution=UNetExecution.Cuda(offloadWeights=true))
      else new NativeDiffusionBackend(root,scheduler)
      val context=new NativeWorkContext(()=>false,()=>true,(_,_)=>())
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      def generate(paths: Vector[NativeJson]) =
        val result=backend.execute(NativeRequest("generate","txt2img",Map("output_dir"->Str(dir.toString),
          "embedding_paths"->Arr(paths),"params"->Obj(Map("prompt"->Str("a red cat legacybin"),
            "width"->Number(16),"height"->Number(16),"steps"->Number(2),"seed"->Number(42)))),None),context)
        val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
        (ImageIO.read(path.toFile).getRGB(0,0,16,16,null,0,16).toVector,result("warnings"))
      val baseline=generate(Vector.empty)._1
      val (active,warnings)=generate(Vector(Str(embedding.toString)))
      assertEquals(warnings,Arr(Vector.empty))
      assertNotEquals(active,baseline)
    }
  }
