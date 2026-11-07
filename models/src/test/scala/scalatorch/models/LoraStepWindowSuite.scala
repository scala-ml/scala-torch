package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*

class LoraStepWindowSuite extends munit.FunSuite:
  test("step windows match NUI Python reference including fractional ties") {
    val input=getClass.getResourceAsStream("/lora-step-window-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    def optional(value: NativeJson): Option[Double] = value match
      case Null => None
      case Number(number) => Some(number.toDouble)
      case _ => fail("invalid reference value")
    for item <- cases do
      val fields=item.asInstanceOf[Obj].fields
      val total=fields("total").asInstanceOf[Number].value.toInt
      val window=LoraStepWindow(optional(fields("start")),optional(fields("stop")))
      val expected=fields("active").asInstanceOf[Arr].values.map(_.asInstanceOf[Bool].value)
      assertEquals((1 to total).map(window.active(_,total)).toVector,expected,s"$window total=$total")
    intercept[IllegalArgumentException](LoraStepWindow(start=Some(Double.NaN)))
    intercept[IllegalArgumentException](LoraStepWindow().active(0,4))
  }
