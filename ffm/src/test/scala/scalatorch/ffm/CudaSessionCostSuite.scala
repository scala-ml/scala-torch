package scalatorch.ffm

import munit.FunSuite

/** Prints what opening a CUDA session costs (context, PTX modules, stream, cuBLAS); only with SCALATORCH_CUDA_TESTS=1. */
class CudaSessionCostSuite extends FunSuite:
  test("session open/close cost") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    val times=(0 until 6).map { _ =>
      val t=System.nanoTime()
      CudaSession.withDevice(maxBytes=1L<<30) { s => s.hasCublas }
      (System.nanoTime()-t)/1e6
    }
    println("CUDA session open+close ms: "+times.map(t => f"$t%.0f").mkString(", "))
  }

  test("small kernel launch cost") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    CudaSession.withDevice(maxBytes=1L<<30) { s =>
      val a=s.uploadHalf(scalatorch.Tensor.fromFloats(Array.fill(4096)(1f),scalatorch.Shape.from(Seq(4096L))))
      val b=s.uploadHalf(scalatorch.Tensor.fromFloats(Array.fill(4096)(2f),scalatorch.Shape.from(Seq(4096L))))
      for _ <- 0 until 2000 do s.add(a,b).close()
      val t=System.nanoTime()
      for _ <- 0 until 20000 do s.add(a,b).close()
      println(f"half add launch+alloc+free: ${(System.nanoTime()-t)/20000/1000.0}%.1f us per op")
    }
  }

  test("session open cost parts") {
    assume(sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"CUDA gate disabled")
    for _ <- 0 until 3 do
      val t0=System.nanoTime()
      CudaDriver.withDevice(0) { (driver,_,_) => () }
      val t1=System.nanoTime()
      CudaDriver.withDevice(0) { (driver,_,_) =>
        val src=scala.io.Source.fromInputStream(getClass.getResourceAsStream("/scalatorch/transformer.ptx")).mkString
        val a=System.nanoTime(); val m=driver.loadModule(src); val b=System.nanoTime()
        val src2=scala.io.Source.fromInputStream(getClass.getResourceAsStream("/scalatorch/gemm.ptx")).mkString
        val m2=driver.loadModule(src2); val c=System.nanoTime()
        println(f"context only ${(t1-t0)/1e6}%.0f ms; transformer.ptx load ${(b-a)/1e6}%.0f ms; gemm.ptx load ${(c-b)/1e6}%.0f ms")
      }
  }
