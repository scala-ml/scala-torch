package scalatorch.models

import java.nio.charset.StandardCharsets.UTF_8
import scalatorch.models.worker.NativeJson
import NativeJson.*

class TorchCpuNormalSuite extends munit.FunSuite:
  test("CPU normal generator follows PyTorch across sizes, seeds and repeated draws") {
    val stream=getClass.getResourceAsStream("/torch-cpu-normal-reference.json")
    val fixture=try NativeJson.parse(new String(stream.readAllBytes(),UTF_8)).asInstanceOf[Obj].fields finally stream.close()
    fixture("cases").asInstanceOf[Arr].values.foreach { value =>
      val fields=value.asInstanceOf[Obj].fields
      val seed=fields("seed").asInstanceOf[Str].value.toLong
      val rng=new TorchCpuNormal(seed)
      val batches=fields("values").asInstanceOf[Arr].values
      val sizes=fields("sizes").asInstanceOf[Arr].values
      sizes.zip(batches).foreach { (size,batch) =>
        val count=size.asInstanceOf[Number].value.toInt
        val expected=batch.asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
        rng.normal(count).zip(expected).zipWithIndex.foreach { case ((actual,want),index) =>
          assertEqualsDouble(actual.toDouble,want,2e-6,s"seed=$seed count=$count index=$index")
        }
      }
    }
  }
