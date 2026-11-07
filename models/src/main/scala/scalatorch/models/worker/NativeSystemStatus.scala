package scalatorch.models.worker

import java.lang.management.ManagementFactory
import java.time.Instant
import scala.util.control.NonFatal

/** JVM/OS observations: heap use is never reported as process RSS. */
object NativeSystemStatus:
  import NativeJson.*
  @volatile private var cudaMemorySnapshot: Map[String,NativeJson]=Map.empty
  def clearCudaMemory(): Unit=cudaMemorySnapshot=Map.empty
  /** Refreshes telemetry from a backend operation thread, never from the protocol reader. */
  def captureCudaMemory(deviceIndex: Int): Unit =
    try
      val info=scalatorch.ffm.CudaProbe.memoryInfo(deviceIndex)
      cudaMemorySnapshot=Map("cuda_total_mb"->Number(BigDecimal(info.totalBytes)/BigDecimal(1048576)),
        "cuda_available_mb"->Number(BigDecimal(info.freeBytes)/BigDecimal(1048576)),
        "cuda_used_mb"->Number(BigDecimal(info.usedBytes)/BigDecimal(1048576)),
        "gpu"->Obj(Map("total_mb"->Number(BigDecimal(info.totalBytes)/BigDecimal(1048576)),
          "free_mb"->Number(BigDecimal(info.freeBytes)/BigDecimal(1048576)),
          "used_mb"->Number(BigDecimal(info.usedBytes)/BigDecimal(1048576)),
          "scope"->Str("whole device"))),
        "cuda_percent"->Number(BigDecimal(info.usedBytes)*100/BigDecimal(info.totalBytes)),
        "cuda_device"->Str(info.deviceName),"gpu_status"->Str("CUDA driver memory telemetry available"))
    catch case NonFatal(error) =>
      cudaMemorySnapshot=Map("gpu_status"->Str(s"CUDA memory telemetry unavailable: ${Option(error.getMessage).getOrElse(error.getClass.getSimpleName)}"))
  def snapshot(model: Map[String,NativeJson]=Map.empty,vaeDevice: String="cpu",unetDevice: String="cpu",
      vaeOffloadWeights: Boolean=false,unetOffloadWeights: Boolean=false,
      maxAttentionScoreElements: Long=16777216L,vaeTiling: Boolean=false,
      maxVaeAttentionScoreElements: Long= -1L): Map[String,NativeJson]=
    val cuda=vaeDevice!="cpu" || unetDevice!="cpu"
    def mb(bytes: Long): NativeJson=Number(BigDecimal(bytes)/BigDecimal(1048576))
    val runtime=ManagementFactory.getRuntimeMXBean
    val heap=ManagementFactory.getMemoryMXBean.getHeapMemoryUsage
    val gpuMemory=if cuda then Some(if cudaMemorySnapshot.nonEmpty then cudaMemorySnapshot else
      Map("gpu_status"->Str("CUDA configured; telemetry pending model load"))) else None
    val process=scalatorch.ffm.ProcessMemory.current().map { info =>
      Map("process_rss_mb"->mb(info.workingSetBytes),"process_peak_rss_mb"->mb(info.peakWorkingSetBytes),
        "process_private_mb"->mb(info.privateBytes),"ram_status"->Str("Windows process working set and private commit"))
    }.getOrElse(Map("ram_status"->Str("process RSS unavailable; JVM heap reported separately")))
    val base=Map("jvm_heap_used_mb"->mb(heap.getUsed),"jvm_heap_committed_mb"->mb(heap.getCommitted)) ++ process ++
      gpuMemory.getOrElse(Map("gpu_status"->Str("CUDA not active in this engine")))
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
        "max_vae_attention_score_elements"->Number(BigDecimal(if maxVaeAttentionScoreElements<0 then maxAttentionScoreElements else maxVaeAttentionScoreElements)),
        "text_encoder_device"->Str("cpu"),"scheduler_device"->Str("cpu"),"batch_execution"->Str("sequential"),
        "vram_mode"->Str("cpu"),"vae_tiling"->Bool(vaeTiling),"dry_run"->Bool(false),
        "flags"->Arr((Vector("cpu","float32")++(if cuda then Vector("cuda") else Vector.empty)).map(Str.apply)),
        "attention"->Str(if unetDevice=="cpu" then "scala-torch CPU" else "scala-torch CUDA"))),
      "device"->Obj(Map("active"->Str(unetDevice),"dtype"->Str("float32"))),
      "platform"->Obj(Map("arch"->Str(sys.props("os.arch")),"system"->Str(sys.props("os.name")),
        "release"->Str(sys.props("os.version")),"java"->Str(sys.props("java.version")),"python"->Str(""))))
