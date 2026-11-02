package scalatorch.models.worker

import java.lang.management.ManagementFactory
import java.time.Instant

/** JVM/OS observations only: heap use is deliberately not reported as process RSS. */
object NativeSystemStatus:
  import NativeJson.*
  def snapshot(model: Map[String,NativeJson]=Map.empty,vaeDevice: String="cpu",unetDevice: String="cpu",
      vaeOffloadWeights: Boolean=false,unetOffloadWeights: Boolean=false,
      maxAttentionScoreElements: Long=16777216L): Map[String,NativeJson]=
    val cuda=vaeDevice!="cpu" || unetDevice!="cpu"
    def mb(bytes: Long): NativeJson=Number(BigDecimal(bytes)/BigDecimal(1048576))
    val runtime=ManagementFactory.getRuntimeMXBean
    val heap=ManagementFactory.getMemoryMXBean.getHeapMemoryUsage
    val base=Map("jvm_heap_used_mb"->mb(heap.getUsed),"jvm_heap_committed_mb"->mb(heap.getCommitted),
      "ram_status"->Str("process RSS unavailable; JVM heap reported separately"),
      "gpu_status"->Str(if !cuda then "CUDA not active in this engine"
        else "CUDA inference configured; phase-scoped allocations, GPU telemetry unavailable"))
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
      "runtime"->Obj(Map("pipeline"->Str("scala-torch"),"backend"->Str(if cuda then "hybrid" else "cpu"),"dtype"->Str("float32"),
        "vae_device"->Str(vaeDevice),"unet_device"->Str(unetDevice),
        "vae_offload_weights"->Bool(vaeOffloadWeights),"unet_offload_weights"->Bool(unetOffloadWeights),
        "max_attention_score_elements"->Number(BigDecimal(maxAttentionScoreElements)),
        "text_encoder_device"->Str("cpu"),"scheduler_device"->Str("cpu"),
        "vram_mode"->Str("cpu"),"vae_tiling"->Bool(false),"dry_run"->Bool(false),
        "flags"->Arr((Vector("cpu","float32")++(if cuda then Vector("cuda") else Vector.empty)).map(Str.apply)),
        "attention"->Str(if unetDevice=="cpu" then "scala-torch CPU" else "scala-torch CUDA"))),
      "device"->Obj(Map("active"->Str(unetDevice),"dtype"->Str("float32"))),
      "platform"->Obj(Map("arch"->Str(sys.props("os.arch")),"system"->Str(sys.props("os.name")),
        "release"->Str(sys.props("os.version")),"java"->Str(sys.props("java.version")),"python"->Str(""))))
