package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*

class NuiLongEulerScheduleSuite extends munit.FunSuite:
  test("Euler schedules above training length match NUI") {
    val input=this.getClass.getResourceAsStream("/long-euler-schedules-reference.json")
    val rows=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    for row <- rows do
      val fields=row.asInstanceOf[Obj].fields
      val count=fields("count").asInstanceOf[Number].value.toInt
      val name=fields("schedule").asInstanceOf[Str].value
      val schedule=name match
        case "Uniform" | "Normal" => noise.normalEuler(count)
        case "Simple" => noise.simpleEuler(count)
        case "SGM Uniform" => noise.sgmUniformEuler(count)
        case "Karras" => noise.karrasEuler(count)
        case _ => noise.exponentialEuler(count)
      val indices=fields("indices").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toInt)
      val times=fields("timesteps").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      val sigmas=fields("sigmas").asInstanceOf[Arr].values.map(_.asInstanceOf[Number].value.toDouble)
      for i <- indices.indices do
        assertEqualsDouble(schedule.timesteps(indices(i)).toDouble,times(i),.003,s"$name $count timestep")
        assertEqualsDouble(schedule.sigmas(indices(i)),sigmas(i),.0002,s"$name $count sigma")
  }
