package scalatorch.models.worker

import NativeJson.*

class NativeHistogramMatchSuite extends munit.FunSuite:
  test("LAB channel histogram transfer matches scikit-image byte references") {
    val stream=this.getClass.getResourceAsStream("/nui-a1111-histogram-reference.json")
    val cases=try NativeJson.parse(new String(stream.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally stream.close()
    cases.foreach { entry =>
      val row=entry.asInstanceOf[Obj].fields
      def bytes(key: String)=row(key).asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt).toArray
      val actual=NativeHistogramMatch.rgbChannels(bytes("source_lab"),bytes("reference_lab"),() => false)
      val expected=bytes("matched")
      val differences=actual.indices.filter(i=>actual(i)!=expected(i))
      assert(differences.isEmpty,s"mismatch ${differences.take(10).map(i=>s"$i:${actual(i)}!=${expected(i)}").mkString(", ")}")
    }
  }
