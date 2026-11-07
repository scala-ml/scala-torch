package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*

class LoraBlockWeightsSuite extends munit.FunSuite:
  test("block layouts and presets match NUI for SD1 and SDXL") {
    val in=getClass.getResourceAsStream("/lora-block-weights-reference.json")
    val cases=try NativeJson.parse(new String(in.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally in.close()
    def number(value: NativeJson)=value.asInstanceOf[Number].value.toDouble
    def groups(value: NativeJson)=value.asInstanceOf[Arr].values.map(_.asInstanceOf[Arr].values.map(number))
    def near(a: Double,b: Double)=math.abs(a-b)<1e-12
    for row <- cases do
      val f=row.asInstanceOf[Obj].fields
      val spec=f("spec").asInstanceOf[Str].value;val sdxl=f("sdxl").asInstanceOf[Bool].value
      val weights=LoraBlockWeights.parse(spec,sdxl,Map("mixed"->"0.7,0.2,-0.4"))
      val expected=f("weights").asInstanceOf[Obj].fields.view.mapValues(number).toMap
      assertEquals(weights,expected,s"$spec SDXL=$sdxl")
      val plan=LoraBlockWeights.plan(weights,sdxl,.7,.4)
      assert(near(plan.text,number(f("text"))) && near(plan.mid,number(f("mid"))))
      assert(plan.down.zip(groups(f("down"))).forall((a,b)=>a.size==b.size && a.zip(b).forall(near)))
      assert(plan.up.zip(groups(f("up"))).forall((a,b)=>a.size==b.size && a.zip(b).forall(near)))
      for (target,value) <- f("targets").asInstanceOf[Obj].fields do
        val actual=LoraBlockWeights.targetScale(plan,sdxl,target)
        value match
          case Null => assertEquals(actual,None,s"$spec $target")
          case _ => assert(actual.exists(near(_,number(value))),s"$spec $target: $actual expected $value")
      assert(LoraBlockWeights.targetScale(plan,sdxl,"text_encoder.encoder.layers.0.fc1").exists(near(_,plan.text)))
    intercept[IllegalArgumentException](LoraBlockWeights.parse("NaN",false))
    intercept[IllegalArgumentException](LoraBlockWeights.parse("bogus",false))
  }
