package scalatorch.models.worker

import java.lang.management.ManagementFactory
import java.nio.file.{Files,Path}
import scala.util.Using
import scala.util.control.NonFatal

/** Best-effort provenance for NUI's hardware-info metadata. */
private[worker] object NativeHardwareInfo:
  import NativeJson.*
  def options(value: Option[NativeJson]): Option[String] = value match
    case None | Some(Null) => None
    case Some(Obj(fields)) =>
      require(fields.keySet.subsetOf(Set("enabled","remark")),"unsupported hardware_info fields")
      val remark=fields.get("remark") match
        case None => ""
        case Some(Str(value)) => value
        case _ => throw IllegalArgumentException("hardware_info remark must be a string")
      fields.get("enabled") match
        case None | Some(Bool(false)) => None
        case Some(Bool(true)) => Some(remark)
        case _ => throw IllegalArgumentException("hardware_info enabled must be boolean")
    case _ => throw IllegalArgumentException("hardware_info must be an object")
  def record(profile: String,remark: String,seconds: Double): Obj =
    Obj(Map("hardware_info"->Str(profile),"time_taken"->Str(duration(seconds)),
      "remark"->(if remark.isEmpty then Null else Str(remark))))
  private def cpuName: Option[String] =
    sys.env.get("PROCESSOR_IDENTIFIER").map(_.trim).filter(_.nonEmpty).orElse {
      val path=Path.of("/proc/cpuinfo")
      if !Files.isRegularFile(path) then None
      else Using.resource(Files.lines(path)) { lines =>
        import scala.jdk.CollectionConverters.*
        lines.iterator().asScala.find(_.startsWith("model name"))
          .map(_.split(":",2)).filter(_.length==2).map(_(1).trim).filter(_.nonEmpty)
      }
    }
  private val vendors=Vector("nvidia","geforce","cpu","gpu","gen","amd","vega","ryzen","radeon","intel","core","arc")
    .map(word=>s"\\b${java.util.regex.Pattern.quote(word)}\\b").mkString("(?i)","|","")
  private val markers="(?i)\\(r\\)|\\(tm\\)|\\(c\\)|@"
  private def roundedGiB(bytes: Long): String=math.rint(bytes.toDouble/1073741824.0).toLong.toString
  private[worker] def compact(cpu: String,ramBytes: Long,gpu: Option[(String,Long)]): String =
    require(cpu.trim.nonEmpty && ramBytes>0 && gpu.forall((name,bytes)=>name.trim.nonEmpty && bytes>0),"invalid hardware facts")
    val parts=gpu.map((name,bytes)=>s"$name ${roundedGiB(bytes)}GB").toVector ++ Vector(cpu,s"${roundedGiB(ramBytes)}GB RAM")
    parts.mkString(", ").replaceAll(vendors,"").replaceAll(markers,"").replaceAll("(?i)\\d+th|\\d+\\.\\d+ghz","")
      .replaceAll("\\s+"," ").trim.replaceAll("\\s+,",",")
  def profile(device: String): String =
    try
      val cpu=cpuName.getOrElse(throw IllegalStateException("CPU identification unavailable"))
      val ram=ManagementFactory.getOperatingSystemMXBean match
        case os: com.sun.management.OperatingSystemMXBean => os.getTotalMemorySize
        case _ => 0L
      val gpu=if device.startsWith("cuda:") then
        val info=scalatorch.ffm.CudaProbe.memoryInfo(device.drop(5).toInt)
        Some(info.deviceName -> info.totalBytes)
      else None
      compact(cpu,ram,gpu)
    catch case NonFatal(_) => "unknown"
  private[worker] def duration(seconds: Double): String =
    val elapsed=math.max(0.0,seconds)
    val minutes=(elapsed/60).toLong
    val rest=elapsed-minutes*60
    val formatted=new java.math.BigDecimal(rest).setScale(1,java.math.RoundingMode.HALF_EVEN).toPlainString
    (if minutes>0 then s"$minutes min. " else "")+s"$formatted sec."
