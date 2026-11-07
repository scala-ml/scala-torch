package scalatorch.models

import java.nio.file.Path
import scalatorch.{Shape,Tensor}
import scalatorch.autograd.Variable
import scalatorch.ffm.CudaSession
import scalatorch.nn.{Conv2d,LinearAdapterScope,Parameter}
import scalatorch.models.worker.NativeJson
import NativeJson.*

class RealDynamicLoraSuite extends munit.FunSuite:
  test("trained LCM LoRA dynamic ranks match the Python NUI transformation") {
    val file=sys.props.get("scalatorch.lcm.rank.fixture")
    assume(file.nonEmpty,"exported trained LoRA layer required")
    val key="unet.down_blocks.0.attentions.0.proj_in"
    val weight=Parameter(Tensor.fromFloats(new Array[Float](320*320),Shape.from(Seq(320L,320L,1L,1L))),false)
    val layer=Conv2d(weight,None)
    val x=Tensor.fromFloats(Array.tabulate(320)(i=>math.sin(i*.013).toFloat),Shape.from(Seq(1L,320L,1L,1L)))
    val stream=getClass.getResourceAsStream("/lcm-dynamic-rank-reference.json")
    val refs=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    for rank <- Vector(0,1,8,64,-1) do
      val adapters=DiffusionLoRA.load(Path.of(file.get),Map(key->weight),dynamicRank=Some(rank))
      val expected=refs(rank.toString).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      def compare(actual: Array[Float],place: String): Unit =
        val error=actual.zip(expected).map((a,b)=>math.abs(a.toDouble-b)).max
        println(s"trained dynamic rank $rank $place max error $error")
        assert(error<1e-4)
      LinearAdapterScope.withAdapters(adapters) {
        compare(layer(Variable(x)).data.toFloatArray,"CPU")
        if sys.env.get("SCALATORCH_CUDA_TESTS").contains("1") then CudaSession.withDevice() { session =>
          val runtime=CudaDiffusionRuntime.upload(session,layer,offloadWeights=true)
          val input=session.upload(x);val output=runtime.conv(input,layer)
          compare(output.toCpu.toFloatArray,"CUDA")
          output.close();input.close();runtime.close()
          assertEquals(session.allocatedBytes,0L)
        }
      }
  }
