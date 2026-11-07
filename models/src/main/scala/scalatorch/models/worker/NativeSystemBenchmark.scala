package scalatorch.models.worker

import java.nio.file.{Files,Path}
import java.util.concurrent.CancellationException
import scala.util.control.NonFatal
import NativeJson.*

/** Real generation benchmark; nested outputs are private and removed on every exit. */
private[worker] object NativeSystemBenchmark:
  val prompt="postapocalyptic steampunk city, exploration, cinematic, realistic, hyper detailed, photorealistic maximum detail, volumetric light, (((focus))), wide-angle, (((brightly lit))), (((vegetation))), lightning, vines, destruction, devastation, wartorn, ruins"
  val negative="(((blurry))), ((foggy)), (((dark))), ((monochrome)), sun, (((depth of field)))"
  private val levels=Map("quick"->Vector(1),"normal"->Vector(1,2,4),"extensive"->Vector(1,2,4,8,16),
    "ludicrous"->Vector(1,2,4,8,16,32,64))
  private val steps=Map("turbo"->10,"default"->20,"long"->50)
  def generationParams(batch: Int,width: Int,height: Int,count: Int): Map[String,NativeJson] =
    Map("prompt"->Str(prompt),"negative_prompt"->Str(negative),"width"->Number(width),"height"->Number(height),
      "steps"->Number(count),"seed"->Number(42),"batch_size"->Number(1),"n_iter"->Number(batch),
      "sampler_name"->Str("Euler a"),"scheduler"->Str("Automatic"),"cfg_scale"->Number(7))
  def run(p: Map[String,NativeJson],system: Map[String,NativeJson],ctx: NativeWorkContext)
      (generate: (Int,Int,Int,Int,Path,NativeWorkContext)=>Int): Map[String,NativeJson] =
    require((p.keySet--Set("level","steps","warmup","width","height","note")).isEmpty,"unknown benchmark parameters")
    def str(key: String,default: String)=p.get(key) match
      case None => default
      case Some(Str(value)) => value
      case _ => throw IllegalArgumentException(s"$key must be a string")
    def dimension(key: String)=p.get(key) match
      case None => 512
      case Some(Number(value)) if value.isWhole && value.isValidInt && value>=64 && value<=8192 && value.toInt%8==0 => value.toInt
      case _ => throw IllegalArgumentException(s"invalid benchmark $key")
    val level=str("level","normal");val preset=str("steps","default")
    require(levels.contains(level) && steps.contains(preset),"unknown benchmark level or step preset")
    val width=dimension("width");val height=dimension("height")
    require(width.toLong*height>=65536,"benchmark image must contain at least 65536 pixels")
    val warmup=p.get("warmup") match
      case None => true
      case Some(Bool(value)) => value
      case _ => throw IllegalArgumentException("warmup must be boolean")
    val note=str("note","");require(note.length<=4096,"benchmark note is too long")
    val batches=levels(level);val count=steps(preset)
    val config=Obj(Map("level"->Str(level),"steps"->Str(preset),"inference_steps"->Number(count),
      "batches"->Arr(batches.map(n=>Number(n))),"batch_execution"->Str("sequential"),"warmup"->Bool(warmup),"sampler"->Str("Euler a"),
      "schedule"->Str("Automatic"),"cfg_scale"->Number(7),"seed"->Number(42),"width"->Number(width),"height"->Number(height)))
    val stable=system.filter((key,_)=>Set("version","platform","libraries","gpu","runtime","model")(key)).map {
      case ("model",Obj(fields)) => "model"->Obj(fields--Set("model_token","path"))
      case pair => pair
    }
    val digest=java.security.MessageDigest.getInstance("SHA-256").digest(NativeJson.render(Obj(stable)).getBytes(java.nio.charset.StandardCharsets.UTF_8))
    val fingerprint=java.util.HexFormat.of().formatHex(digest).take(12)
    val output=Files.createTempDirectory("scalatorch-system-benchmark-")
    val rows=scala.collection.mutable.ArrayBuffer.empty[NativeJson]
    var active=1
    try
      def runOne(batch: Int,measured: Boolean): Unit =
        ctx.checkCancelled()
        val started=System.nanoTime()
        val actual=generate(batch,width,height,count,output,ctx)
        val seconds=(System.nanoTime()-started).toDouble/1e9
        require(actual==batch && seconds>0 && seconds.isFinite,"benchmark returned invalid image count or duration")
        if measured then rows+=Obj(Map("batch"->Number(batch),"batch_execution"->Str("sequential"),"seconds"->Number(BigDecimal.decimal(seconds)),
          "iterations_per_second"->Number(BigDecimal.decimal(count.toDouble*batch/seconds)),
          "megapixels_per_second"->Number(BigDecimal.decimal(width.toDouble*height*batch/1048576/seconds)),"status"->Str("complete")))
      try
        if warmup then runOne(1,false)
        batches.foreach { batch =>
          active=batch
          ctx.event("log",Map("level"->Str("info"),"message"->Str(s"System benchmark batch $batch")))
          runOne(batch,true)
        }
      catch
        case c: CancellationException => throw c
        case error: OutOfMemoryError =>
          ctx.checkCancelled()
          rows+=Obj(Map("batch"->Number(active),"batch_execution"->Str("sequential"),"status"->Str("oom"),"error"->Str(Option(error.getMessage).getOrElse("out of memory").take(2048))))
        case NonFatal(error) =>
          ctx.checkCancelled()
          val message=Option(error.getMessage).getOrElse(error.getClass.getSimpleName)
          var cause: Throwable=error
          var oom=false
          while cause!=null && !oom do
            val text=Option(cause.getMessage).getOrElse("").toLowerCase(java.util.Locale.ROOT)
            oom=cause.isInstanceOf[OutOfMemoryError] || text.contains("out of memory") || text.contains("cuda status 2")
            cause=cause.getCause
          rows+=Obj(Map("batch"->Number(active),"batch_execution"->Str("sequential"),"status"->Str(if oom then "oom" else "error"),"error"->Str(message.take(2048))))
      Map("timestamp"->system.getOrElse("timestamp",Str(java.time.Instant.now().toString)),"fingerprint"->Str(fingerprint),
        "system"->Obj(stable),"config"->config,"results"->Arr(rows.toVector),"note"->Str(note),
        "status"->Str(if rows.forall(_.asInstanceOf[Obj].fields("status")==Str("complete")) then "complete" else "stopped_after_error"))
    finally
      val paths=Files.walk(output)
      try paths.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p=>{Files.deleteIfExists(p);()}) finally paths.close()
