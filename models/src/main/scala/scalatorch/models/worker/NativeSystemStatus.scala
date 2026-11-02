package scalatorch.models.worker

import java.lang.management.ManagementFactory
import java.time.Instant

/** JVM/OS observations only: heap use is deliberately not reported as process RSS. */
object NativeSystemStatus:
  import NativeJson.*
  def snapshot(model: Map[String,NativeJson]=Map.empty,vaeDevice: String="cpu"): Map[String,NativeJson]=
    def mb(bytes: Long): NativeJson=Number(BigDecimal(bytes)/BigDecimal(1048576))
    val runtime=ManagementFactory.getRuntimeMXBean
    val heap=ManagementFactory.getMemoryMXBean.getHeapMemoryUsage
    val base=Map("jvm_heap_used_mb"->mb(heap.getUsed),"jvm_heap_committed_mb"->mb(heap.getCommitted),
      "ram_status"->Str("process RSS unavailable; JVM heap reported separately"),
      "gpu_status"->Str(if vaeDevice=="cpu" then "CUDA not active in this engine"
        else "CUDA VAE decode configured; allocations scoped to decode, GPU telemetry unavailable"))
    val memory=ManagementFactory.getOperatingSystemMXBean match
      case os: com.sun.management.OperatingSystemMXBean =>
        val total=os.getTotalMemorySize
        val free=os.getFreeMemorySize
        if total>0 && free>=0 && free<=total then base++Map("ram_total_mb"->mb(total),"ram_available_mb"->mb(free),
          "ram_percent"->Number(BigDecimal(total-free)*100/BigDecimal(total)))
        else base
      case _ => base
    Map("timestamp"->Str(Instant.now().toString),"started_at"->Str(Instant.ofEpochMilli(runtime.getStartTime).toString),
      "uptime_seconds"->Number(BigDecimal(runtime.getUptime)/1000),"memory"->Obj(memory),"model"->Obj(model),
      "runtime"->Obj(Map("pipeline"->Str("scala-torch"),"backend"->Str("cpu"),"dtype"->Str("float32"),
        "vae_device"->Str(vaeDevice),
        "vram_mode"->Str("cpu"),"vae_tiling"->Bool(false),"dry_run"->Bool(false),
        "flags"->Arr(Vector(Str("cpu"),Str("float32"))),"attention"->Str("scala-torch CPU"))),
      "device"->Obj(Map("active"->Str("cpu"),"dtype"->Str("float32"))),
      "platform"->Obj(Map("arch"->Str(sys.props("os.arch")),"system"->Str(sys.props("os.name")),
        "release"->Str(sys.props("os.version")),"java"->Str(sys.props("java.version")),"python"->Str(""))))
