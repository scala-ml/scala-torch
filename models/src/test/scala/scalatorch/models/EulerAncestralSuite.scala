package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.models.worker.NativeJson
import NativeJson.*
import java.nio.charset.StandardCharsets.UTF_8

class EulerAncestralSuite extends munit.FunSuite:
  private def obj(value: NativeJson)=value.asInstanceOf[Obj].fields
  private def arr(value: NativeJson)=value.asInstanceOf[Arr].values
  private def number(value: NativeJson)=value.asInstanceOf[Number].value.toDouble
  private def tensor(values: Vector[NativeJson]): Variable =
    Variable(Tensor.fromFloats(values.map(v=>number(v).toFloat).toArray,Shape.from(Seq(values.size.toLong))))
  test("Euler a steps match independent Diffusers fixtures with explicit noise") {
    val stream=getClass.getResourceAsStream("/euler-ancestral-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),
      "0996e3d2bcb2ba43c08f739cf6cb158007a7d24adc336050b7a4f4138dd821b9")
    val root=NativeJson.parse(new String(bytes,UTF_8))
    for item <- arr(obj(root)("cases")) do
      val fields=obj(item)
      val prediction=if fields("prediction")==Str("epsilon") then DiffusionPrediction.Epsilon else DiffusionPrediction.VPrediction
      val schedule=EulerSchedule(arr(fields("timesteps")).map(v=>number(v).toFloat),
        arr(fields("sigmas")).map(number),number(fields("initial_noise_sigma")))
      val scheduler=EulerAncestralScheduler(schedule,prediction)
      arr(fields("steps")).zipWithIndex.foreach { (item,index) =>
        val values=obj(item)
        val result=scheduler.step(index,tensor(arr(values("model_output"))),tensor(arr(values("sample"))),tensor(arr(values("noise"))))
        for (actual,name) <- Seq(result.previous.data->"previous",result.predictedOriginal.data->"original") do
          val expected=arr(values(name)).map(v=>number(v).toFloat)
          actual.toFloatArray.zip(expected).foreach((a,b) => assertEqualsFloat(a,b,2e-6f))
      }
  }
  test("Euler a validates prediction, noise shape and index") {
    val schedule=EulerSchedule(Vector(1f,0f),Vector(2.0,1.0,0.0),2.0)
    val a=Variable(Tensor.fromFloats(Array(1f,2f),Shape.from(Seq(2L))))
    val short=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq(1L))))
    intercept[IllegalArgumentException](EulerAncestralScheduler(schedule,DiffusionPrediction.Sample))
    val scheduler=EulerAncestralScheduler(schedule)
    intercept[IllegalArgumentException](scheduler.step(0,a,a,short))
    intercept[IllegalArgumentException](scheduler.step(2,a,a,a))
  }
