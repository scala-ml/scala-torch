package scalatorch.models

import scalatorch.{Shape,Tensor}
import scalatorch.autograd.Variable
import scalatorch.models.worker.NativeJson.*

class DdimSchedulerSuite extends munit.FunSuite:
  test("DDIM updates match independent Diffusers references for prediction and clipping modes") {
    val stream=this.getClass.getResourceAsStream("/ddim-sampler-reference.json")
    val fixture=try scalatorch.models.worker.NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    val shape=Shape.from(Seq(1L,4L))
    val sample=Variable(Tensor.fromFloats(Array(-2.5f,-.5f,.5f,2.5f),shape))
    val model=Variable(Tensor.fromFloats(Array(.1f,-.4f,.3f,-.2f),shape))
    for (spacing,name)<-Vector((DiffusionTimestepSpacing.Linspace,"linspace"),(DiffusionTimestepSpacing.Leading,"leading"),
      (DiffusionTimestepSpacing.Trailing,"trailing")); clip<-Vector(false,true);
      (prediction,pred)<-Vector((DiffusionPrediction.Epsilon,"epsilon"),(DiffusionPrediction.Sample,"sample"),
        (DiffusionPrediction.VPrediction,"v_prediction")) do
      val reference=fixture(s"$name:${if clip then "True" else "False"}:$pred").asInstanceOf[Obj].fields
      val expectedTimes=reference("timesteps").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
      val expected=reference("values").asInstanceOf[Arr].values.map(_.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat))
      val schedule=DdimSchedule(noise,4,spacing,1)
      assertEquals(schedule.layout.timesteps,expectedTimes)
      val solver=DdimScheduler(schedule,prediction,clip)
      expected.indices.foreach { i =>
        val actual=solver.step(i,model,sample).data.toFloatArray
        actual.zip(expected(i)).foreach((a,b)=>assertEqualsFloat(a,b,0.00002f))
        val negative=reference("negative_eta").asInstanceOf[Arr].values(i).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toFloat)
        val actualNegative=solver.step(i,model,sample,eta= -.5).data.toFloatArray
        actualNegative.zip(negative).foreach((a,b)=>assertEqualsFloat(a,b,0.00002f))
      }
  }
