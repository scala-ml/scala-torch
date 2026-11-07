package scalatorch.models

class CudaDeviceMemoryBudgetSuite extends munit.FunSuite:
  test("automatic CUDA budget uses half of free VRAM up to two GiB") {
    val gib=1024L*1024L*1024L
    assertEquals(CudaDeviceMemoryBudget.fromAvailableBytes(4*gib),2*gib)
    assertEquals(CudaDeviceMemoryBudget.fromAvailableBytes(6*gib),2*gib)
    assertEquals(CudaDeviceMemoryBudget.fromAvailableBytes(1*gib),512L*1024L*1024L)
    assertEquals(CudaDeviceMemoryBudget.fromAvailableBytes(1L),1L)
    intercept[IllegalArgumentException](CudaDeviceMemoryBudget.fromAvailableBytes(0L))
    val configured=Map("CUDA_DEVICE"->"cuda","CUDA_BUDGET"->"1234")
    assertEquals(CudaDeviceMemoryBudget.withAutomaticBudgets(configured,Seq("CUDA_DEVICE"->"CUDA_BUDGET")),configured)
    val cpu=Map("CUDA_DEVICE"->"cpu")
    assertEquals(CudaDeviceMemoryBudget.withAutomaticBudgets(cpu,Seq("CUDA_DEVICE"->"CUDA_BUDGET")),cpu)
  }

  test("offload cache stays at one quarter for small budgets and caps at 512 MiB") {
    val gib=1024L*1024L*1024L
    val mib=1024L*1024L
    assertEquals(CudaDiffusionRuntime.offloadedWeightLimit(1024L),256L)
    assertEquals(CudaDiffusionRuntime.offloadedWeightLimit(gib),256L*mib)
    assertEquals(CudaDiffusionRuntime.offloadedWeightLimit(2*gib),512L*mib)
    assertEquals(CudaDiffusionRuntime.offloadedWeightLimit(8*gib),512L*mib)
  }
