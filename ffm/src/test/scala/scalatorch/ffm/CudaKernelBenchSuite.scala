package scalatorch.ffm

import scalatorch.*

/** Kernel timings on the shapes of an SD1.5 U-Net pass (batch of two, Float16). Run with SCALATORCH_CUDA_BENCH=1; prints microseconds per
  * call, the fastest of seven runs of many queued calls with one synchronisation at the end. */
class CudaKernelBenchSuite extends munit.FunSuite:
  override val munitTimeout=scala.concurrent.duration.Duration(600,"s")
  private def tensor(dims: Long*): Tensor[Float] =
    val shape=Shape.from(dims)
    Tensor.fromFloats(Array.tabulate(shape.numel.toInt)(i => math.sin(i*.37).toFloat),shape)

  test("kernel timings") {
    assume(sys.env.get("SCALATORCH_CUDA_BENCH").contains("1"),"benchmark disabled")
    CudaSession.withDevice(maxBytes=6L<<30) { s =>
      def half(dims: Long*): CudaTensor = s.uploadHalf(tensor(dims*))
      def time(label: String,rounds: Int)(body: => CudaTensor): Unit =
        def run(n: Int): Double =
          val started=System.nanoTime()
          var i=0
          while i<n do { body.close();i+=1 }
          s.allFinite(s.fillHalf(Shape.from(Seq(1L)),0f)) // downloads: waits for the queue
          (System.nanoTime()-started)/1e3/n
        run(20)
        val micros=Seq.fill(7)(run(rounds)).min // the GPU is shared with the desktop: the fastest of several runs
        println(f"[bench] $label%-46s $micros%8.1f us")
      for (n,c,side) <- Seq((2L,320L,64L),(8L,320L,64L),(2L,640L,32L),(8L,640L,32L),(8L,1280L,16L),(8L,1280L,8L)) do
        val x=half(n,c,side,side)
        val w=half(c,c,3,3)
        time(s"conv 3x3           [$n,$c,$side,$side]",60)(s.conv2dFast(x,w,Conv2dConfig(padding=(1,1))))
        x.close();w.close()
      for (c,side) <- Seq((320L,64L),(640L,32L),(1280L,16L),(1280L,8L)) do
        val x=half(2,c,side,side)
        val channelBias=half(1,c,1,1)
        val gamma=half(c);val beta=half(c)
        time(s"add channel bias   [2,$c,$side,$side]",300)(s.addBroadcast(x,channelBias))
        time(s"add same shape     [2,$c,$side,$side]",300)(s.add(x,x))
        time(s"silu               [2,$c,$side,$side]",300)(s.silu(x))
        time(s"group norm 32      [2,$c,$side,$side]",300)(s.groupNorm(x,32,1e-5,Some(gamma),Some(beta)))
        val tokens=half(2,side*side,c)
        val lastBias=half(c)
        time(s"add last-axis bias [2,${side*side},$c]",300)(s.addBroadcast(tokens,lastBias))
        val heads=half(2,side*side,8,c/8)
        time(s"transpose heads    [2,${side*side},8,${c/8}]",300)(s.transpose(heads,1,2))
        Seq(x,channelBias,gamma,beta,tokens,lastBias,heads).foreach(_.close())
    }
  }
