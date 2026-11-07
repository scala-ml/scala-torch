package scalatorch.models.worker

import NativeJson.*

class NativeTemperatureGuardSuite extends munit.FunSuite:
  test("temperature hysteresis reports pause and resume and observes interval") {
    var now=0.0;var reads=0
    val readings=Iterator(85.0,79.0,69.0)
    val events=scala.collection.mutable.ArrayBuffer.empty[(String,Double)]
    val guard=new NativeTemperatureGuard.Guard(NativeTemperatureGuard.Config(),()=>{reads+=1;readings.next()},
      ()=>(),(k,v)=>events+=((k,v)),()=>now,s=>now+=s)
    guard.step()
    assertEquals(events.map(_._1).toVector,Vector("pause","reading","reading","resume"))
    assertEquals(reads,3);assert(math.abs(now-2)<1e-8)
    guard.step();assertEquals(reads,3)
  }
  test("timeout, disabled guard, invalid values, and interruption are explicit") {
    var now=0.0
    val events=scala.collection.mutable.ArrayBuffer.empty[String]
    val guard=new NativeTemperatureGuard.Guard(NativeTemperatureGuard.Config(delay=10,limit=.3),()=>90,()=>(),
      (k,_)=>events+=k,()=>now,s=>now+=s)
    guard.step();assert(now<=.300001);assertEquals(events.last,"timeout")
    val disabled=new NativeTemperatureGuard.Guard(NativeTemperatureGuard.Config(interval=0),()=>throw AssertionError("read"),()=>(),(_,_)=>())
    disabled.step()
    intercept[IllegalArgumentException](NativeTemperatureGuard.parse(Obj(Map("sleep_time"->Number(0)))))
    val badSensor=NativeTemperatureGuard.parse(Obj(Map("provider"->Str("OpenHardwareMonitor"),"sensor_id"->Str("/intelcpu/0/temperature/0"))))
    intercept[IllegalArgumentException](NativeTemperatureGuard.read(badSensor))
    var waits=0
    val interrupted=new NativeTemperatureGuard.Guard(NativeTemperatureGuard.Config(),()=>90,
      ()=>if waits>0 then throw new java.util.concurrent.CancellationException(),(_,_)=>(),()=>0,s=>waits+=1)
    intercept[java.util.concurrent.CancellationException](interrupted.step());assertEquals(waits,1)
  }
  test("sensor responses select the requested device and reject unavailable readings") {
    val nvidia=NativeTemperatureGuard.Config()
    assertEquals(NativeTemperatureGuard.decode(nvidia,"42"),42.0)
    for value <- Vector("N/A","NaN","Infinity","151") do
      intercept[IllegalArgumentException](NativeTemperatureGuard.decode(nvidia,value))
    val rocm=NativeTemperatureGuard.Config(provider="ROCm",device=1)
    val cards="""{"card0":{"Temperature (Sensor edge) (C)":"40"},"card1":{"Temperature (Sensor edge) (C)":"54.5"}}"""
    assertEquals(NativeTemperatureGuard.decode(rocm,cards),54.5)
    intercept[IllegalArgumentException](NativeTemperatureGuard.decode(rocm.copy(device=2),cards))
    val ohm=NativeTemperatureGuard.Config(provider="OpenHardwareMonitor",sensor="/amdgpu/0/temperature/0")
    def response(value: String)=s"""[{"Identifier":"/amdgpu/0/temperature/0","SensorType":"Temperature","Value":$value}]"""
    assertEquals(NativeTemperatureGuard.decode(ohm,response("55")),55.0)
    for value <- Vector("0","null","true","\"N/A\"") do
      intercept[IllegalArgumentException](NativeTemperatureGuard.decode(ohm,response(value)))
    intercept[IllegalArgumentException](NativeTemperatureGuard.decode(ohm.copy(sensor="/amdgpu/1/temperature/0"),response("55")))
  }
  test("sensor command drains output before waiting for process exit") {
    assume(sys.props("os.name").startsWith("Windows"))
    val output=NativeTemperatureGuard.run(Seq("powershell.exe","-NoProfile","-NonInteractive","-Command","[Console]::Out.Write(('x' * 100000))"),10)
    assertEquals(output.length,100000)
  }
