package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*
import java.nio.charset.StandardCharsets.UTF_8

class NuiEulerScheduleSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toDouble
  test("native Euler schedules match NUI's independent reference") {
    val stream=getClass.getResourceAsStream("/nui-euler-schedule-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),
      "b8064517ac4472849514a99cd4c73d73127fe7fc86eedcfbfaeb5236fcab56e5")
    val cases=arr(obj(NativeJson.parse(new String(bytes,UTF_8)))("cases"))
    val noise=DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
    cases.foreach { value =>
      val fields=obj(value);val count=num(fields("count")).toInt
      val schedule=fields("schedule") match
        case Str("Uniform") => noise.nuiUniformEuler(count)
        case Str("Karras") => noise.karrasEuler(count)
        case Str("Exponential") => noise.exponentialEuler(count)
        case Str("Normal") => noise.normalEuler(count)
        case Str("Simple") => noise.simpleEuler(count)
        case Str("SGM Uniform") => noise.sgmUniformEuler(count)
        case _ => fail("unexpected reference schedule")
      assertEquals(schedule.timesteps.length,arr(fields("timesteps")).length)
      assertEquals(schedule.sigmas.length,arr(fields("sigmas")).length)
      schedule.timesteps.zip(arr(fields("timesteps"))).foreach((a,b)=>assertEqualsDouble(a.toDouble,num(b),.003))
      schedule.sigmas.zip(arr(fields("sigmas"))).foreach((a,b)=>assertEqualsDouble(a,num(b),.0002))
      assertEqualsDouble(schedule.initialNoiseSigma,num(fields("initial_noise_sigma")),.0002)
    }
  }
