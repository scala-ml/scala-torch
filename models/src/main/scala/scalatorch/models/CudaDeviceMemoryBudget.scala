package scalatorch.models

import scalatorch.ffm.CudaProbe

/** Conservative per-session CUDA allocation budget, derived from currently free device memory. */
private[models] object CudaDeviceMemoryBudget:
  private val Maximum=2L*1024L*1024L*1024L

  def fromAvailableBytes(freeBytes: Long): Long =
    require(freeBytes>0,"CUDA reports no free device memory")
    math.max(1L,math.min(Maximum,freeBytes/2))

  def withAutomaticBudgets(environment: Map[String,String],devices: Seq[(String,String)]): Map[String,String] =
    devices.foldLeft(environment) { case (current,(deviceKey,budgetKey)) =>
      val index=current.get(deviceKey).flatMap {
        case "cuda" => Some(0)
        case value if value.matches("cuda:[0-9]+") => Some(value.drop(5).toInt)
        case _ => None
      }
      if current.contains(budgetKey) || index.isEmpty then current
      else current.updated(budgetKey,fromAvailableBytes(CudaProbe.memoryInfo(index.get).freeBytes).toString)
    }
