package scalatorch.models.worker

import java.util.concurrent.TimeUnit
import NativeJson.*

/** Interruptible sampling-step temperature guard. */
private[worker] object NativeTemperatureGuard:
  final case class Config(provider: String="NVIDIA",sensor: String="",device: Int=0,hot: Double=80,cool: Double=70,
      interval: Double=1,delay: Double=1,limit: Double=0,printReadings: Boolean=false)
  def parse(value: NativeJson): Config =
    val f=value match
      case Obj(fields) => fields
      case _ => throw IllegalArgumentException("temperature guard must be an object")
    require(f.keySet.subsetOf(Set("enabled","provider","sensor_id","device","sleep_temperature","wake_temperature","interval","sleep_time","max_sleep","print_readings")),"unknown temperature guard option")
    def text(k: String,d: String): String=f.get(k) match
      case None => d
      case Some(Str(v)) => v
      case _ => throw IllegalArgumentException(s"invalid $k")
    def num(k: String,d: Double): Double=f.get(k) match
      case None => d
      case Some(Number(v)) if v.toDouble.isFinite => v.toDouble
      case _ => throw IllegalArgumentException(s"invalid $k")
    val device=f.get("device") match
      case None => 0
      case Some(Number(v)) if v.isValidInt && v>=0 && v<=31 => v.toInt
      case _ => throw IllegalArgumentException("invalid temperature device")
    val readings=f.get("print_readings") match
      case None => false
      case Some(Bool(v)) => v
      case _ => throw IllegalArgumentException("invalid print_readings")
    val c=Config(text("provider","NVIDIA"),text("sensor_id",""),device,num("sleep_temperature",80),
      num("wake_temperature",70),num("interval",1),num("sleep_time",1),num("max_sleep",0),readings)
    require(Set("NVIDIA","ROCm","OpenHardwareMonitor")(c.provider),"unknown temperature provider")
    require(c.interval>=0 && c.delay>0 && c.limit>=0,"invalid temperature guard timing")
    c

  final class Guard(c: Config,read: () => Double,check: () => Unit,event: (String,Double) => Unit,
      clock: () => Double=()=>System.nanoTime()/1e9,wait: Double => Unit=s=>Thread.sleep((s*1000).ceil.toLong)):
    private var last: Option[Double]=None
    def step(): Unit =
      check()
      if c.interval==0 || c.cool>c.hot then return
      val now=clock()
      if last.exists(now-_<c.interval) then return
      last=Some(now)
      def temperature(): Double =
        val v=read();require(v.isFinite,"temperature reading must be finite");v
      var t=temperature()
      if t<=c.hot then return
      val started=now;event("pause",t)
      var timeout=false
      while t>=c.cool && !timeout do
        check()
        if c.limit>0 && clock()-started>=c.limit then timeout=true
        else
          var remaining=if c.limit>0 then math.min(c.delay,math.max(0,c.limit-(clock()-started))) else c.delay
          while remaining>1e-9 do
            check();val chunk=math.min(.1,remaining);wait(chunk);remaining-=chunk
          check();t=temperature();event("reading",t)
      event(if timeout then "timeout" else "resume",t);last=Some(clock())

  def read(c: Config): Double =
    if c.provider=="OpenHardwareMonitor" then require((c.sensor.startsWith("/amdgpu/") || c.sensor.startsWith("/nvidiagpu/")) && c.sensor.contains("/temperature/"),"an exact GPU temperature sensor ID is required")
    val (cmd,timeout)=c.provider match
      case "NVIDIA" => (Seq("nvidia-smi",s"--id=${c.device}","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"),2L)
      case "ROCm" => (Seq("rocm-smi","--showtemp","--json"),2L)
      case _ => (Seq("powershell.exe","-NoProfile","-NonInteractive","-Command",
        "$ErrorActionPreference='Stop'; Get-CimInstance -Namespace root/OpenHardwareMonitor -ClassName Sensor | Select-Object Identifier,SensorType,Value | ConvertTo-Json -Compress"),3L)
    decode(c,run(cmd,timeout))

  private[worker] def run(cmd: Seq[String],timeout: Long): String =
    val deadline=System.nanoTime()+TimeUnit.SECONDS.toNanos(timeout)
    val process=new ProcessBuilder(cmd*).redirectErrorStream(true).start()
    process.getOutputStream.close()
    val data=new java.util.concurrent.atomic.AtomicReference[Array[Byte]]()
    val failure=new java.util.concurrent.atomic.AtomicReference[Throwable]()
    val reader=Thread.ofVirtual().start(() =>
      try data.set(process.getInputStream.readNBytes(1048577))
      catch case e: java.io.IOException => failure.set(e))
    try
      if !process.waitFor(timeout,TimeUnit.SECONDS) then throw IllegalStateException("temperature sensor timed out")
      val remaining=deadline-System.nanoTime()
      if remaining>0 then reader.join(math.max(1,TimeUnit.NANOSECONDS.toMillis(remaining)))
      if reader.isAlive then throw IllegalStateException("temperature sensor output timed out")
      if failure.get()!=null then throw IllegalStateException("temperature sensor output failed",failure.get())
      val bytes=data.get()
      require(bytes!=null && bytes.length<=1048576 && process.exitValue()==0,"temperature sensor failed")
      new String(bytes,java.nio.charset.StandardCharsets.UTF_8).trim
    finally
      if process.isAlive then process.destroyForcibly()
      process.getInputStream.close()

  private[worker] def decode(c: Config,output: String): Double =
      val v=c.provider match
        case "NVIDIA" => output.toDouble
        case "ROCm" => NativeJson.parse(output) match
          case Obj(cards) => cards.get(s"card${c.device}") match
            case Some(Obj(card)) => card.get("Temperature (Sensor edge) (C)") match
              case Some(Str(s)) => s.toDouble
              case Some(Number(n)) => n.toDouble
              case _ => throw IllegalArgumentException("ROCm temperature sensor unavailable")
            case _ => throw IllegalArgumentException("ROCm device unavailable")
          case _ => throw IllegalArgumentException("invalid ROCm sensor response")
        case _ =>
          val rows=NativeJson.parse(output) match
            case Arr(values) => values
            case o: Obj => Vector(o)
            case _ => Vector.empty
          val matches=rows.collect { case Obj(row) if row.get("Identifier").contains(Str(c.sensor)) && row.get("SensorType").contains(Str("Temperature")) => row.get("Value") }
          require(matches.size==1,"temperature sensor unavailable or ambiguous")
          matches.head match
            case Some(Number(n)) if n!=0 => n.toDouble
            case _ => throw IllegalArgumentException("temperature sensor unavailable")
      require(v.isFinite && v>= -50 && v<=150,"invalid GPU temperature reading");v
