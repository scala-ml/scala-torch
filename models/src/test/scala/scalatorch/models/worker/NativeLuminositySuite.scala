package scalatorch.models.worker

import NativeJson.*

class NativeLuminositySuite extends munit.FunSuite:
  test("luminosity blend matches NUI reference pixels") {
    val input=this.getClass.getResourceAsStream("/nui-luminosity-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    cases.foreach { entry =>
      val row=entry.asInstanceOf[Obj].fields
      def bytes(key: String)=row(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt).toArray
      val actual=NativeLuminosity.blend(bytes("background"),bytes("foreground"),() => false)
      val expected=bytes("output")
      val differences=actual.indices.filter(i=>actual(i)!=expected(i))
      assert(differences.isEmpty,s"${differences.size} mismatches: ${differences.take(8).map(i=>s"$i:${actual(i)}!=${expected(i)}").mkString(",")}")
    }
  }
