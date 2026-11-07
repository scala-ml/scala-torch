package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*

class NuiExtraSchedulesSuite extends munit.FunSuite:
  test("Euler ancestral and DPM fallback schedules match NUI reference") {
    val input=this.getClass.getResourceAsStream("/nui-extra-schedules-reference.json")
    val cases=try NativeJson.parse(new String(input.readAllBytes(),java.nio.charset.StandardCharsets.UTF_8)).asInstanceOf[Arr].values finally input.close()
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    cases.foreach { entry =>
      val row=entry.asInstanceOf[Obj].fields
      val count=row("count").asInstanceOf[Number].value.toInt
      val trailing=row("schedule")!=Str("Normal")
      val schedule=if row("sampler")==Str("Euler a") then
        noise.euler(count,if trailing then DiffusionTimestepSpacing.Trailing else DiffusionTimestepSpacing.Linspace)
      else if trailing then DpmSchedule.trailing(noise,count).layout else DpmSchedule.uniform(noise,count).layout
      val times=row("timesteps").asInstanceOf[Arr].values
      val levels=row("sigmas").asInstanceOf[Arr].values
      assertEquals(schedule.timesteps.length,times.length)
      assertEquals(schedule.sigmas.length,levels.length)
      schedule.timesteps.zip(times).foreach((a,b)=>assertEqualsDouble(a.toDouble,b.asInstanceOf[Number].value.toDouble,.001))
      schedule.sigmas.zip(levels).foreach((a,b)=>assertEqualsDouble(a,b.asInstanceOf[Number].value.toDouble,.0002))
    }
  }
