package scalatorch.models.worker

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files,Path,StandardCopyOption,AtomicMoveNotSupportedException}
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID
import java.util.concurrent.atomic.{AtomicBoolean,AtomicLong}
import scala.util.control.NonFatal
import scalatorch.ffm.CudaProbe

/** Profile identity and admission shared by the native worker's VRAM tools. */
private[worker] object NativeVramEstimator:
  import NativeJson.*
  private val modes=Set("txt2img","txt2img_hr","txt2img_hr_latent","img2img")
  final case class CalibrationCase(mode: String,target: Int,width: Int,height: Int,batch: Int,upscaler: Option[String])

  def fingerprint(identity: NativeJson): String =
    val digest=MessageDigest.getInstance("SHA-256").digest(NativeJson.render(identity).getBytes(UTF_8))
    digest.take(12).map(byte=>f"${byte & 0xff}%02x").mkString

  def validate(profile: NativeJson): Unit =
    val fields=profile match
      case Obj(value) => value
      case _ => throw IllegalArgumentException("VRAM estimator: malformed profile")
    require(fields.get("version").contains(Number(1)),"VRAM estimator: unsupported profile version")
    require(fields.get("status").exists {
      case Str(value) => Set("running","complete","interrupted","stopped_after_error").contains(value)
      case _ => false
    },"VRAM estimator: invalid profile status")
    val identity=fields.getOrElse("identity",throw IllegalArgumentException("VRAM estimator: missing identity"))
    val expected=fingerprint(identity)
    require(fields.get("fingerprint").contains(Str(expected)),"VRAM estimator: profile fingerprint does not match identity")
    val rows=fields.getOrElse("rows",throw IllegalArgumentException("VRAM estimator: missing rows")) match
      case Arr(values) if values.size<=4096 => values
      case _ => throw IllegalArgumentException("VRAM estimator: invalid profile rows")
    rows.foreach {
      case Obj(row) =>
        val mode=row.get("mode") match
          case Some(Str(value)) if modes(value) => value
          case _ => throw IllegalArgumentException("VRAM estimator: invalid profile mode")
        val status=row.get("status") match
          case Some(Str(value)) if Set("complete","oom","error")(value) => value
          case _ => throw IllegalArgumentException("VRAM estimator: invalid profile row status")
        if status=="complete" then
          def number(key: String): BigDecimal=row.get(key) match
            case Some(Number(value)) if value.isDecimalDouble => value
            case _ => throw IllegalArgumentException(s"VRAM estimator: invalid $key")
          require(number("pixels")>0 && number("pixels").isWhole,"VRAM estimator: invalid pixels")
          require(number("batch_size")>0 && number("batch_size").isWhole,"VRAM estimator: invalid batch_size")
          require(number("reserved_delta_mb")>=0,"VRAM estimator: invalid reserved memory")
      case _ => throw IllegalArgumentException("VRAM estimator: invalid profile row")
    }

  def profile(model: Map[String,NativeJson],deviceName: String,totalBytes: Long,
      vaeDevice: String,unetDevice: String,profile: Option[NativeJson],supported: Boolean,
      maxAttentionScoreElements: Long=16777216L,vaeOffloadWeights: Boolean=false,
      unetOffloadWeights: Boolean=false,vaeTiling: Boolean=false,maxVaeAttentionScoreElements: Long= -1L): Map[String,NativeJson] =
    profile.foreach(validate)
    val stableModel=Set("name","hash","family","vae_hash","vae_name").flatMap(key=>model.get(key).map(key->_)).toMap
    val identity=Obj(Map(
      "version"->Obj(Map("tag"->Str("scala-torch-0.1.0"))),
      "libraries"->Obj(Map("backend"->Str("scala-torch"))),
      "gpu"->Obj(Map("name"->Str(deviceName),"total_memory_bytes"->Number(BigDecimal(totalBytes)))),
      "runtime"->Obj(Map("backend"->Str("scala-torch"),"dtype"->Str("float32"),
        "vae_device"->Str(vaeDevice),"unet_device"->Str(unetDevice),
        "vae_offload_weights"->Bool(vaeOffloadWeights),"unet_offload_weights"->Bool(unetOffloadWeights),
        "vae_tiling"->Bool(vaeTiling),"pixel_hires_vae_roundtrip"->Bool(true),"batch_execution"->Str("sequential"),
        "max_attention_score_elements"->Number(BigDecimal(maxAttentionScoreElements)),
        "max_vae_attention_score_elements"->Number(BigDecimal(if maxVaeAttentionScoreElements<0 then maxAttentionScoreElements else maxVaeAttentionScoreElements)))),
      "model"->Obj(stableModel)
    ))
    val current=if supported && model.nonEmpty then identity else Null
    val currentHash=if current==Null then Null else Str(fingerprint(current))
    val record=profile.getOrElse(Null)
    val compatible=(record,currentHash) match
      case (Obj(fields),Str(hash)) => fields.get("fingerprint").contains(Str(hash))
      case _ => false
    Map("profile"->record,"current_identity"->current,"current_fingerprint"->currentHash,
      "compatible"->Bool(compatible),"supported"->Bool(supported),"model_loaded"->Bool(model.nonEmpty))

  private def solveLeastSquares(source: Array[Array[Double]],target: Array[Double]): Option[Array[Double]] =
    if source.isEmpty || source.length!=target.length || source.exists(_.length==0) then return None
    val rows=source.length
    val columns=source(0).length
    if source.exists(_.length!=columns) || rows<columns then return None
    val a=source.map(_.clone())
    val b=target.clone()
    val initialScale=math.sqrt(a.iterator.flatMap(_.iterator).map(x=>x*x).max)
    if !initialScale.isFinite || initialScale==0 then return None
    var k=0
    while k<columns do
      val norm=math.sqrt((k until rows).iterator.map(i=>a(i)(k)*a(i)(k)).sum)
      if !norm.isFinite || norm<=initialScale*1e-12 then return None
      val alpha=if a(k)(k)>=0 then -norm else norm
      val v=Array.tabulate(rows-k)(i=>a(k+i)(k))
      v(0)-=alpha
      val vv=v.iterator.map(x=>x*x).sum
      if vv==0 || !vv.isFinite then return None
      val beta=2.0/vv
      var col=k
      while col<columns do
        var dot=0.0
        var i=0
        while i<v.length do { dot+=v(i)*a(k+i)(col);i+=1 }
        i=0
        while i<v.length do { a(k+i)(col)-=beta*v(i)*dot;i+=1 }
        col+=1
      var dot=0.0
      var i=0
      while i<v.length do { dot+=v(i)*b(k+i);i+=1 }
      i=0
      while i<v.length do { b(k+i)-=beta*v(i)*dot;i+=1 }
      a(k)(k)=alpha
      k+=1
    val solution=Array.fill(columns)(0.0)
    var row=columns-1
    while row>=0 do
      var value=b(row)
      var col=row+1
      while col<columns do { value-=a(row)(col)*solution(col);col+=1 }
      if math.abs(a(row)(row))<=initialScale*1e-12 then return None
      solution(row)=value/a(row)(row)
      row-=1
    if solution.forall(_.isFinite) then Some(solution) else None

  private def unavailable(mode: String,reason: String): Map[String,NativeJson]=
    Map("available"->Bool(false),"mode"->Str(mode),"reason"->Str(reason))

  def estimate(profile: NativeJson,mode: String,width: Int,height: Int,batch: Int,
      systemUsedMb: Double,totalMb: Double): Map[String,NativeJson] =
    validate(profile)
    require(modes(mode),"VRAM estimator: unknown mode")
    require(width>0 && height>0 && batch>0,"VRAM estimator: dimensions and batch must be positive")
    require(systemUsedMb.isFinite && totalMb.isFinite && systemUsedMb>=0 && totalMb>0 && systemUsedMb<=totalMb,
      "VRAM estimator: invalid current memory")
    val records=profile.asInstanceOf[Obj].fields("rows").asInstanceOf[Arr].values
    val measuredOom=width==height && records.exists {
      case Obj(row) => row.get("mode").contains(Str(mode)) && row.get("status").contains(Str("oom")) &&
        row.get("pixels").contains(Number(BigDecimal(width)*height)) && row.get("batch_size").contains(Number(batch))
      case _ => false
    }
    if measuredOom then return unavailable(mode,"measured_oom")
    val rows=records.collect {
      case Obj(row) if row.get("mode").contains(Str(mode)) && row.get("status").contains(Str("complete")) =>
        val pixels=row("pixels").asInstanceOf[Number].value.toDouble
        val rowBatch=row("batch_size").asInstanceOf[Number].value.toDouble
        val delta=row("reserved_delta_mb").asInstanceOf[Number].value.toDouble
        (math.sqrt(pixels),rowBatch,delta)
    }
    val point=math.sqrt(width.toDouble*height.toDouble)
    val sameBatch=rows.filter(_._2==batch.toDouble)
    val (prediction,fitted,trainingValues,method,degree,sampleCount,sizeMin,sizeMax,extrapolated)=
      if sameBatch.nonEmpty then
        val xs=sameBatch.map(_._1);val ys=sameBatch.map(_._3)
        val unique=xs.distinct.sorted
        if unique.size==1 then
          if math.abs(point-unique.head)>math.max(1.0,math.abs(unique.head))*1e-12 then
            return unavailable(mode,"insufficient_size_samples")
          val predicted=ys.sum/ys.size
          (predicted,Array.fill(ys.size)(predicted),ys,"measured_point",0,ys.size,xs.min,xs.max,false)
        else
          val fitDegree=math.min(3,unique.size-1)
          val center=xs.sum/xs.size
          val scale=math.sqrt(xs.iterator.map(x=>(x-center)*(x-center)).sum/xs.size)
          val normalized=xs.map(x=>(x-center)/scale)
          val matrix=normalized.map(x=>Array.tabulate(fitDegree+1)(power=>math.pow(x,power)))
          solveLeastSquares(matrix.toArray,ys.toArray) match
            case None => return unavailable(mode,"rank_deficient_samples")
            case Some(coefficients) =>
              def evaluate(x: Double): Double = coefficients.indices.iterator.map(i=>coefficients(i)*math.pow((x-center)/scale,i)).sum
              (evaluate(point),ys.indices.map(i=>evaluate(xs(i))).toArray,ys,"per_batch_polynomial",fitDegree,
                ys.size,xs.min,xs.max,point<xs.min || point>xs.max)
      else
        val coordinates=rows.map(r=>(r._1,r._2));val values=rows.map(_._3)
        if rows.size<10 then return unavailable(mode,"insufficient_surface_samples")
        val centerX=coordinates.map(_._1).sum/coordinates.size
        val centerB=coordinates.map(_._2).sum/coordinates.size
        val scaleX=math.sqrt(coordinates.iterator.map(c=>math.pow(c._1-centerX,2)).sum/coordinates.size)
        val scaleB=math.sqrt(coordinates.iterator.map(c=>math.pow(c._2-centerB,2)).sum/coordinates.size)
        if scaleX==0 || scaleB==0 then return unavailable(mode,"insufficient_surface_samples")
        val normalized=coordinates.map(c=>((c._1-centerX)/scaleX,(c._2-centerB)/scaleB))
        val terms=(0 to 3).flatMap(degree=>(0 to degree).map(power=>(power,degree-power)))
        val matrix=normalized.map((x,b)=>terms.map((i,j)=>math.pow(x,i)*math.pow(b,j)).toArray)
        solveLeastSquares(matrix.toArray,values.toArray) match
          case None => return unavailable(mode,"rank_deficient_surface")
          case Some(coefficients) =>
            def evaluate(x0: Double,b0: Double): Double =
              val x=(x0-centerX)/scaleX;val b=(b0-centerB)/scaleB
              terms.indices.iterator.map(index=>coefficients(index)*math.pow(x,terms(index)._1)*math.pow(b,terms(index)._2)).sum
            val queryX=(point-centerX)/scaleX;val queryB=(batch-centerB)/scaleB
            val lowX=normalized.map(_._1).min;val highX=normalized.map(_._1).max
            val lowB=normalized.map(_._2).min;val highB=normalized.map(_._2).max
            (evaluate(point,batch),coordinates.indices.map(i=>evaluate(coordinates(i)._1,coordinates(i)._2)).toArray,values,
              "cubic_surface",3,rows.size,coordinates.map(_._1).min,coordinates.map(_._1).max,
              queryX<lowX || queryX>highX || queryB<lowB || queryB>highB)
    val fit=fitted
    val fitMethod=method
    val fitDegree=degree
    val samples=sampleCount
    val minSize=sizeMin
    val maxSize=sizeMax
    val outside=extrapolated
    if !prediction.isFinite || prediction<0 then return unavailable(mode,"nonphysical_fit")
    val rmse=math.sqrt(fit.indices.iterator.map(i=>math.pow(fit(i)-trainingValues(i),2)).sum/fit.length)
    val estimateTotal=systemUsedMb+1.1*prediction
    if !estimateTotal.isFinite || !rmse.isFinite then return unavailable(mode,"nonfinite_fit")
    def num(value: Double)=Number(BigDecimal.decimal(value))
    Map("available"->Bool(true),"mode"->Str(mode),"batch_execution"->Str("sequential"),"method"->Str(fitMethod),"degree"->Number(fitDegree),
      "samples"->Number(samples),"incremental_mb"->num(prediction),"system_used_mb"->num(systemUsedMb),
      "estimated_total_mb"->num(estimateTotal),"total_mb"->num(totalMb),"fraction"->num(estimateTotal/totalMb),
      "over_capacity"->Bool(estimateTotal>totalMb),"extrapolated"->Bool(outside),"training_rmse_mb"->num(rmse),
      "measured_size_range"->Arr(Vector(num(minSize),num(maxSize))),
      "shape_assumption"->Str(if width!=height then "square_calibration_by_pixel_area" else "square"),
      "buffer_fraction"->num(0.1))

  private def integer(params: Map[String,NativeJson],key: String,default: Int,low: Int,high: Int): Int =
    val result=params.get(key) match
      case None => default
      case Some(Number(value)) if value.isWhole && value.isValidInt => value.toInt
      case _ => throw IllegalArgumentException(s"VRAM estimator: invalid $key")
    require(result>=low && result<=high,s"VRAM estimator: invalid $key")
    result

  private def memoryMetric(value: Long): NativeJson=Number(BigDecimal(value)/BigDecimal(1048576))
  private def samplePeak(deviceIndex: Int)(generate: => Unit): Map[String,NativeJson] =
    val baseline=CudaProbe.memoryInfo(deviceIndex)
    val peak=new AtomicLong(baseline.usedBytes)
    val running=new AtomicBoolean(true)
    def sample(): Unit =
      try
        val used=CudaProbe.memoryInfo(deviceIndex).usedBytes
        peak.accumulateAndGet(used,Math.max)
      catch case NonFatal(_) => ()
    val sampler=new Thread(() =>
      while running.get() do
        sample()
        try Thread.sleep(15) catch case _: InterruptedException => Thread.currentThread().interrupt()
    ,"scala-torch-vram-sampler")
    sampler.setDaemon(true)
    sampler.start()
    try generate
    finally
      running.set(false)
      sampler.interrupt()
      sampler.join(2000)
    sample()
    val after=CudaProbe.memoryInfo(deviceIndex)
    val peakBytes=math.max(peak.get(),after.usedBytes)
    val delta=math.max(0L,peakBytes-baseline.usedBytes)
    val mib=1048576.0
    Map("reserved_delta_mb"->memoryMetric(delta),"reserved_peak_mb"->memoryMetric(peakBytes),
      "baseline_reserved_mb"->memoryMetric(baseline.usedBytes),
      "system_used_before_mb"->memoryMetric(baseline.usedBytes),
      "system_used_after_mb"->memoryMetric(after.usedBytes),"total_mb"->memoryMetric(after.totalBytes),
      "measurement"->Str("sampled device-wide CUDA used-memory peak; 15ms sampling"))

  private def isOutOfMemory(error: Throwable): Boolean =
    var current=error
    while current!=null do
      val message=Option(current.getMessage).getOrElse("").toLowerCase(java.util.Locale.ROOT)
      if current.isInstanceOf[OutOfMemoryError] || message.contains("out of memory") ||
          message.contains("cuda status 2") then return true
      current=current.getCause
    false

  private def atomicWrite(path: Path,record: NativeJson): Unit =
    val parent=path.getParent
    Files.createDirectories(parent)
    val temp=Files.createTempFile(parent,".vram-profile-",".tmp")
    try
      Files.writeString(temp,NativeJson.render(record)+"\n",UTF_8)
      try Files.move(temp,path,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)
      catch case _: AtomicMoveNotSupportedException => Files.move(temp,path,StandardCopyOption.REPLACE_EXISTING)
    finally Files.deleteIfExists(temp)

  private def deleteTree(root: Path): Unit =
    if Files.exists(root) then
      val stream=Files.walk(root)
      try stream.sorted(java.util.Comparator.reverseOrder()).forEach(path=>Files.deleteIfExists(path))
      finally stream.close()

  /** Serial native calibration. `generate` must use the supplied scoped context and avoid user output paths. */
  def benchmark(params: Map[String,NativeJson],identity: NativeJson,fingerprintValue: String,deviceIndex: Int,
      context: NativeWorkContext,onUpdate: NativeJson=>Unit)(generate: (CalibrationCase,NativeWorkContext,Path,Path)=>Unit): NativeJson =
    val maxSize=integer(params,"max_size",960,256,2048)
    val maxBatch=integer(params,"max_batch",8,1,16)
    val steps=integer(params,"steps",1,1,100)
    require(maxSize%64==0,"VRAM estimator: maximum size must be a multiple of 64")
    require(params.get("save_samples").forall(_==Bool(false)),"native VRAM calibration does not save sample images")
    val session=params.get("session_id") match
      case Some(Str(value)) if value.nonEmpty && value.length<=128 => value
      case _ => throw IllegalArgumentException("VRAM estimator requires a session_id")
    val outputRoot=params.get("output_dir") match
      case Some(Str(value)) if value.trim.nonEmpty => Path.of(value).toAbsolutePath.normalize()
      case _ => throw IllegalArgumentException("VRAM estimator requires an output directory")
    Files.createDirectories(outputRoot)
    require(Files.isDirectory(outputRoot),"VRAM estimator output path is not a directory")
    val runDir=Files.createDirectories(outputRoot.resolve("vram-estimator").resolve("run-"+UUID.randomUUID()))
    val profilePath=runDir.resolve("profile.json")
    val inputPath=runDir.resolve("synthetic-input.png")
    val input=new java.awt.image.BufferedImage(512,512,java.awt.image.BufferedImage.TYPE_INT_RGB)
    require(javax.imageio.ImageIO.write(input,"png",inputPath.toFile()),"PNG writer unavailable for calibration input")
    val config=Obj(Map("max_size"->Number(maxSize),"max_batch"->Number(maxBatch),"steps"->Number(steps),
      "session_id"->Str(session)))
    val assumptions=Obj(Map("sampler"->Str("Euler"),"scheduler"->Str("Uniform"),"cfg_scale"->Number(7),
      "vanilla_hooks"->Bool(true),"input"->Str("synthetic black RGB 512x512"),
      "memory_measurement"->Str("sampled device-wide CUDA usage, 15ms sampling"),
      "upscaler_policy"->Str("Lanczos Hires decodes, resizes pixels and re-encodes; latent Hires uses native Latent")))
    val targets=(256 to maxSize by 64).toVector
    val cases=(for
      mode<-Vector("txt2img","txt2img_hr","txt2img_hr_latent","img2img")
      batch<-1 to maxBatch
      target<-targets
    yield
      val hires=mode=="txt2img_hr" || mode=="txt2img_hr_latent"
      CalibrationCase(mode,target,if hires then target/2 else target,if hires then target/2 else target,batch,
        if mode=="txt2img_hr" then Some("Lanczos") else if mode=="txt2img_hr_latent" then Some("Latent") else None)
    ).toVector
    require(cases.size<=4096,"VRAM estimator calibration plan exceeds profile row limit")
    val rows=scala.collection.mutable.ArrayBuffer.empty[NativeJson]
    val stopped=scala.collection.mutable.Set.empty[(String,Int)]
    val startedAt=Instant.now().toString
    def record(status: String): NativeJson=Obj(Map("version"->Number(1),"status"->Str(status),"config"->config,
      "rows"->Arr(rows.toVector),"planned_cases"->Number(cases.size),"attempted_cases"->Number(rows.size),
      "started_at"->Str(startedAt),"identity"->identity,"fingerprint"->Str(fingerprintValue),
      "profile_path"->Str(profilePath.toString),"assumptions"->assumptions))
    def publish(status: String): NativeJson =
      val value=record(status)
      atomicWrite(profilePath,value)
      onUpdate(value)
      value
    publish("running")
    try
      cases.zipWithIndex.foreach { (calibration,index) =>
        context.checkCancelled()
        val group=(calibration.mode,calibration.batch)
        if !stopped(group) then
          val fields=scala.collection.mutable.Map[String,NativeJson](
            "mode"->Str(calibration.mode),"width"->Number(calibration.width),"height"->Number(calibration.height),
            "pixels"->Number(calibration.target.toLong*calibration.target),"target"->Number(calibration.target),
            "batch_size"->Number(calibration.batch))
          calibration.upscaler.foreach(value=>fields("upscaler")=Str(value))
          val began=System.nanoTime()
          val caseDir=Files.createDirectories(runDir.resolve(f"case-$index%05d"))
          var keepDiagnostics=false
          def recordFailure(error: Throwable): Unit =
            context.checkCancelled()
            val oom=isOutOfMemory(error)
            keepDiagnostics=Option(error.getMessage).exists(_.contains("saved to"))
            fields("status")=Str(if oom then "oom" else "error")
            fields("error")=Str(Option(error.getMessage).getOrElse(error.getClass.getSimpleName).take(2048))
            stopped+=group
          try
            try
              val metrics=samplePeak(deviceIndex)(generate(calibration,context,inputPath,caseDir))
              fields++=metrics
              fields("status")=Str("complete")
            catch
              case error: java.util.concurrent.CancellationException => throw error
              case error: OutOfMemoryError => recordFailure(error)
              case NonFatal(error) => recordFailure(error)
          finally if !keepDiagnostics then deleteTree(caseDir)
          fields("seconds")=Number(BigDecimal((System.nanoTime()-began).toDouble/1e9))
          rows+=Obj(fields.toMap)
          publish("running")
          context.event("progress",Map("step"->Number(index+1),"total"->Number(cases.size),
            "pass"->Str("vram_estimator"),"eta_s"->Number(0)))
      }
      publish(if stopped.nonEmpty then "stopped_after_error" else "complete")
    catch
      case error: java.util.concurrent.CancellationException => publish("interrupted");throw error
    finally Files.deleteIfExists(inputPath)
