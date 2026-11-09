package scalatorch.models.worker

import java.util.UUID
import java.util.concurrent.{CancellationException, ExecutionException, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean

/** Runs the `n_iter` groups of a txt2img request concurrently ("count" = parallel streams, "size" = images per stream).
  * Each group is an ordinary request of its own (n_iter 1, batch_size B, seeds shifted by the group offset, so the images are the same as in a
  * sequential run); the groups share the device-resident weights and run on their own threads and CUDA streams. Only plain requests are split:
  * LoRA, ControlNet, embeddings, hooks, hardware profiling and every other key this class does not know stay on the sequential path, because their
  * state is per request. A group that fails with a memory error is repeated alone after the others have finished. */
private[worker] object NativeParallelCount:
  import NativeJson.*

  /** Parallel streams at most; `SDFX_SCALATORCH_PARALLEL_COUNT_MAX` (default 4, 1 = off). */
  val limit: Int = sys.env.get("SDFX_SCALATORCH_PARALLEL_COUNT_MAX").flatMap(_.trim.toIntOption).map(math.max(1,_)).getOrElse(4)

  private val knownOuter=Set("params","output_dir","vae_recovery_dir","preview_every","preview_method")
  /** Lists of what is available (not what the request uses). */
  private val catalogs=Set("lora_paths","controlnet_paths","upscaler_paths","lbw_presets")
  private val knownParams=Set("prompt","negative_prompt","width","height","steps","seed","subseed","cfg_scale","sampler_name","scheduler",
    "clip_skip","batch_size","n_iter","checkpoint","vae","extra","subseed_strength","styles","loras","hooks","hires")

  private def num(fields: Map[String,NativeJson],key: String): Option[BigDecimal]=fields.get(key) match
    case Some(Number(n)) => Some(n)
    case _ => None
  private def emptyish(v: NativeJson): Boolean=v match
    case Null => true
    case Obj(m) => m.isEmpty || m.get("enabled").contains(Bool(false))
    case Arr(a) => a.isEmpty
    case Bool(false) => true
    case _ => false
  private def text(fields: Map[String,NativeJson],key: String): String=fields.get(key) match
    case Some(Str(s)) => s
    case _ => ""

  private def skipped(request: NativeRequest,why: String): Unit=
    if sys.env.get("SDFX_SCALATORCH_PARALLEL_COUNT_LOG").contains("1") then System.err.println(s"[scala-torch] parallel count not used: $why")

  /** (n_iter, batch_size, params) when the request can be split. */
  private def hookEnabled(value: NativeJson): Boolean=value match
    case Null => false
    case Bool(flag) => flag
    case Number(n) => n!=0
    case Str(text) => !Set("","false","off","no","0","disable","disabled").contains(text.trim.toLowerCase(java.util.Locale.ROOT))
    case Obj(fields) => hookEnabled(fields.getOrElse("enabled",Bool(true)))
    case _ => true

  private def plan(request: NativeRequest,usesCuda: Boolean,keywordInert: (Option[NativeJson],String,Int) => Boolean): Option[(Int,Int,Map[String,NativeJson])]=
    if request.kind!="txt2img" || !usesCuda || limit<=1 then None
    else if request.params.exists((k,v) => !knownOuter(k) && !catalogs(k) && !emptyish(v)) then
      skipped(request,"request keys in use: "+request.params.collect { case (k,v) if !knownOuter(k) && !catalogs(k) && !emptyish(v) => k }.mkString(","))
      None
    else request.params.get("params") match
      case Some(Obj(g)) =>
        val n=num(g,"n_iter").map(_.toInt).getOrElse(1)
        val b=num(g,"batch_size").map(_.toInt).getOrElse(1)
        val prompts=text(g,"prompt")+" "+text(g,"negative_prompt")
        val activeHooks=g.get("hooks") match
          case Some(Obj(h)) => h.filter((_,v) => hookEnabled(v)).keySet
          case _ => Set.empty[String]
        // The model keyword hook is always sent; it only matters when it changes a prompt or the keyword state.
        val hooksPlain=(activeHooks-"model_keyword").isEmpty && (!activeHooks.contains("model_keyword") || keywordInert(g.get("hooks"),text(g,"prompt"),n*b))
        val plain=g.forall((k,v) => knownParams(k) || emptyish(v)) && g.get("loras").forall(emptyish) && hooksPlain &&
          !prompts.contains("<lora:") && !prompts.contains("<lyco:") && !prompts.contains("<hypernet:") && !prompts.contains("<")
        if n>1 && n<=100 && b>=1 && plain then Some((n,b,g))
        else
          if n>1 then skipped(request,"values "+Seq("img2img","controlnet","eta","hooks").flatMap(k => g.get(k).map(v => k+"="+v.toString.take(300))).mkString(" | ")+" parameters "+(g.keySet--knownParams).mkString(",")+(if g.get("loras").exists(!emptyish(_)) then " loras" else "")+(if g.get("hooks").exists(!emptyish(_)) then " hooks" else "")+(if prompts.contains("<") then " prompt-tags" else ""))
          None
      case _ => None

  def run(request: NativeRequest,context: NativeWorkContext,usesCuda: Boolean,keywordInert: (Option[NativeJson],String,Int) => Boolean)(
      runOne: (NativeRequest,NativeWorkContext) => Map[String,NativeJson]): Option[Map[String,NativeJson]]=
    plan(request,usesCuda,keywordInert).map { case (n,b,g) =>
      val started=System.nanoTime()
      val parallel=math.min(n,limit)
      val seed=num(g,"seed").map(_.toLong).getOrElse(-1L)
      val baseSeed=if seed>=0 then seed else java.util.concurrent.ThreadLocalRandom.current().nextLong(0L,1L<<32)
      val subseed=num(g,"subseed").map(_.toLong).getOrElse(-1L)
      val emit=new Object
      val failed=new AtomicBoolean(false)
      def subContext(group: Int): NativeWorkContext=
        new NativeWorkContext(() => failed.get() || context.cancelled(),
          () => try { context.beginPublication();true } catch case _: CancellationException => false,
          (name,fields) => emit.synchronized {
            val remapped=if fields.contains("image") && fields.contains("images") then
              fields.updated("image",Number(group*b+(fields("image") match { case Number(i) => i.toInt; case _ => 0 })))
                .updated("images",Number(n*b))
            else fields
            context.event(name,remapped)
          })
      def groupRequest(group: Int): NativeRequest=
        val shifted=g.updated("n_iter",Number(1)).updated("seed",Number(BigDecimal(baseSeed+group.toLong*b))) ++
          (if subseed>=0 then Map("subseed"->Number(BigDecimal(subseed+group.toLong*b))) else Map.empty)
        NativeRequest(UUID.randomUUID().toString,"txt2img",request.params.updated("params",Obj(shifted)),request.owner)

      val pool=Executors.newFixedThreadPool(parallel,(r: Runnable) => { val t=new Thread(r,"scala-torch-count-stream");t.setDaemon(true);t })
      val results=new Array[Map[String,NativeJson]](n)
      val retry=new java.util.concurrent.ConcurrentLinkedQueue[java.lang.Integer]()
      def memoryError(e: Throwable): Boolean=
        val m=Option(e.getMessage).getOrElse("")+Option(e.getCause).flatMap(c => Option(c.getMessage)).getOrElse("")
        m.contains("allocation budget exceeded") || m.toLowerCase(java.util.Locale.ROOT).contains("out of memory") || m.contains("CUDA_ERROR_OUT_OF_MEMORY")
      try
        val futures=(0 until n).map { group =>
          pool.submit(new java.util.concurrent.Callable[Unit]:
            def call(): Unit=
              if !failed.get() then
                try results(group)=runOne(groupRequest(group),subContext(group))
                catch case e: Throwable if memoryError(e) => retry.add(group))
        }
        var firstError: Throwable=null
        futures.foreach { f =>
          try f.get()
          catch case e: ExecutionException => { failed.set(true);if firstError==null then firstError=e.getCause }
        }
        if firstError!=null then throw firstError
      finally pool.shutdown()
      // groups that did not fit next to the others run alone now
      var again: java.lang.Integer=retry.poll()
      while again!=null && !context.cancelled() do
        results(again.intValue)=runOne(groupRequest(again.intValue),subContext(again.intValue))
        again=retry.poll()
      context.checkCancelled()
      val done=results.toVector
      require(done.forall(_!=null),"parallel generation did not finish")
      val images=done.flatMap(_("images").asInstanceOf[Arr].values)
      val warnings=done.flatMap(_.get("warnings").collect { case Arr(w) => w }.getOrElse(Vector.empty)).distinct
      done.head++Map("images"->Arr(images),"warnings"->Arr(warnings),"timing_s"->Number(BigDecimal((System.nanoTime()-started)/1e9)),
        "parallel_streams"->Number(parallel))
    }
