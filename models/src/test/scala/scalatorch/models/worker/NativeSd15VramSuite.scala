package scalatorch.models.worker

import java.nio.file.{Files,Path}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scalatorch.models.{UNetExecution,VaeExecution}
import NativeJson.*

/** Opt-in real-model measurement; nvidia-smi reports whole-device usage, including the desktop. */
class NativeSd15VramSuite extends munit.FunSuite:
  override val munitTimeout: Duration=15.minutes
  test("SD1.5 CUDA 512 generation with weight offloading") {
    val dir=sys.props.get("scalatorch.sd15.dir")
    assume(dir.nonEmpty && sys.env.get("SCALATORCH_CUDA_TESTS").contains("1"),"SD1.5 directory and CUDA required")
    val root=Path.of(dir.get)
    val output=Files.createTempDirectory("native-sd15-vram")
    val stop=new java.util.concurrent.atomic.AtomicBoolean(false)
    val peak=new java.util.concurrent.atomic.AtomicLong(0)
    val sampler=new Thread(() => {
      while !stop.get() do
        try
          val process=new ProcessBuilder("nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits").start()
          val value=scala.io.Source.fromInputStream(process.getInputStream)
          val text=try value.mkString.trim.linesIterator.nextOption().getOrElse("") finally value.close()
          if process.waitFor()==0 then text.toLongOption.foreach(n=>peak.accumulateAndGet(n,java.lang.Math.max))
          Thread.sleep(500)
        catch case _: InterruptedException => ()
    })
    sampler.setDaemon(true);sampler.start()
    try
      val scheduler=output.resolve("scheduler.json")
      Files.writeString(scheduler,"""{"num_train_timesteps":1000,"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear"}""")
      val backend=new NativeDiffusionBackend(root,scheduler,vaeExecution=VaeExecution.Cuda(offloadWeights=true),
        unetExecution=UNetExecution.Cuda(offloadWeights=true),maxAttentionScoreElements=134217728L)
      val steps=sys.props.get("scalatorch.sd15.steps").fold(1)(_.toInt)
      require(steps>0 && steps<=50)
      val progress=new scala.collection.mutable.ArrayBuffer[Long]()
      val context=new NativeWorkContext(()=>false,()=>true,(kind,_)=>if kind=="progress" then progress += System.nanoTime())
      val started=System.nanoTime()
      backend.execute(NativeRequest("load","load_model",Map.empty,None),context)
      val loaded=System.nanoTime()
      val params=Map[String,NativeJson]("output_dir"->Str(output.toString),"params"->Obj(Map(
        "prompt"->Str("a red apple on a wooden table"),"width"->Number(512),"height"->Number(512),
        "steps"->Number(steps),"seed"->Number(42),"sampler_name"->Str("Euler"),"scheduler"->Str("Uniform"))))
      val result=backend.execute(NativeRequest("generate","txt2img",params,None),context)
      val path=Path.of(result("images").asInstanceOf[Arr].values.head.asInstanceOf[Obj].fields("path").asInstanceOf[Str].value)
      val image=ImageIO.read(path.toFile)
      assertEquals((image.getWidth,image.getHeight),(512,512))
      sys.props.get("scalatorch.sd15.output").foreach(destination=>Files.copy(path,Path.of(destination)))
      println(f"SD1.5 512x512: load ${(loaded-started)/1e9}%.2f s, generation ${(System.nanoTime()-loaded)/1e9}%.2f s, sampled whole-GPU peak ${peak.get()} MiB")
      println("progress elapsed seconds: "+progress.map(t=>f"${(t-loaded)/1e9}%.2f").mkString(", "))
      assertEquals(progress.size,steps)
      assert(peak.get()>0,"nvidia-smi sampling must succeed")
    finally
      stop.set(true);sampler.interrupt();sampler.join(5000)
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(Files.deleteIfExists(_)) finally paths.close()
  }
