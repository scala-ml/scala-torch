package scalatorch.models

import scalatorch.models.worker.NativeJson
import NativeJson.*
import java.nio.charset.StandardCharsets.UTF_8

class NuiDpmExponentialSuite extends munit.FunSuite:
  private def obj(v: NativeJson)=v.asInstanceOf[Obj].fields
  private def arr(v: NativeJson)=v.asInstanceOf[Arr].values
  private def num(v: NativeJson)=v.asInstanceOf[Number].value.toDouble
  test("DPM exponential and Euler a fallback match NUI scheduler references") {
    val stream=getClass.getResourceAsStream("/nui-dpm-exponential-reference.json")
    val bytes=try stream.readAllBytes() finally stream.close()
    assertEquals(java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)),"9110c044df609a25a43ec863814c7c4af4983455352bd1c608ad865485c16bf3")
    for value <- arr(obj(NativeJson.parse(new String(bytes,UTF_8)))("cases")) do
      val fields=obj(value);val count=num(fields("count")).toInt
      val noise=num(fields("profile")).toInt match
        case 0 => DiffusionNoiseSchedule.linear(1000,.00085,.012,scaled=true)
        case 1 => DiffusionNoiseSchedule.linear(1000,.0001,.02)
        case 2 => DiffusionNoiseSchedule.fromBetas((1 to 8).map(_ * .001).toVector)
      val (times,sigmas)=if fields("sampler")==Str("DPM++ 2M") then
        val result=DpmSchedule.exponential(noise,count);(result.timesteps,result.sigmas)
      else
        assert(arr(fields("warnings")).nonEmpty)
        val result=noise.euler(count,DiffusionTimestepSpacing.Leading,1);(result.timesteps,result.sigmas)
      times.zip(arr(fields("timesteps"))).zipWithIndex.foreach { case ((a,b),index) =>
        val tolerance=if fields("profile")==Number(0) then 0.0 else 1.0
        assert(math.abs(a.toDouble-num(b))<=tolerance,
          s"profile ${fields("profile")} sampler ${fields("sampler")} count $count index $index: $a vs ${num(b)}")
      }
      sigmas.zip(arr(fields("sigmas"))).foreach((a,b)=>assertEqualsDouble(a,num(b),.0002))
  }
