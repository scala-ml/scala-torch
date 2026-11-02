package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable

class DpmReferenceSuite extends munit.FunSuite:
  test("DPM++ trajectories match pinned Diffusers CPU outputs for all prediction modes") {
    val stream=getClass.getResourceAsStream("/dpm/trajectory.tsv")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),
      "ee9b33bc17640039c2d93e118ebf380103fb4ba0bc37809057037583f1a0a188")
    val rows=new String(bytes,java.nio.charset.StandardCharsets.UTF_8).linesIterator.drop(1)
      .map(_.split("\t").toVector).toVector
    def variable(values: Vector[Float])=Variable(Tensor.fromFloats(values.toArray,Shape.from(Seq(2))),false)
    for (name,mode) <- Vector("epsilon"->DiffusionPrediction.Epsilon,"sample"->DiffusionPrediction.Sample,
        "v_prediction"->DiffusionPrediction.VPrediction) do
      val cases=rows.filter(_.head==name)
      assertEquals(cases.size,4)
      val sigmas=cases.map(_(3).toDouble) :+ cases.last(4).toDouble
      val solver=DpmPlusPlus2M(sigmas,mode)
      var sample=variable(cases.head.slice(5,7).map(_.toFloat))
      var history: Option[Variable]=None
      for (row,i) <- cases.zipWithIndex do
        assertEquals(row(1).toInt,i)
        val result=solver.step(i,variable(row.slice(7,9).map(_.toFloat)),sample,history)
        result.previous.data.toFloatArray.zip(row.slice(9,11).map(_.toFloat)).foreach { (actual,expected) =>
          assertEqualsFloat(actual,expected,1e-5f,s"$name step $i")
        }
        sample=result.previous
        history=Some(result.predictedOriginal)
      val schedule=DpmSchedule.karras(DiffusionNoiseSchedule.linear(1000,.00085,.012,true),4)
      assertEquals(schedule.timesteps,cases.map(_(2).toFloat))
      schedule.sigmas.zip(sigmas).foreach { (actual,expected) =>
        assertEqualsDouble(actual,expected,3e-5)
      }
  }
