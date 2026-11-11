package scalatorch.ffm

import scalatorch.*

/** Device-resident weights shared by all sessions on a device (and by the sessions of different threads). A generation opens several sessions
  * and a batch of images many more; uploading the same few GB of weights every time cost more than the sampling of a small image.
  * Entries are keyed by the identity of the parameter and checked against the identity of the host tensor they were uploaded from.
  * They are freed together with the kept CUDA context ([[CudaContextKeeper]]) after its idle time, or explicitly. */
private[ffm] object CudaWeightPool:
  final class Entry(val allocation: CudaAllocation,val shape: Shape,val half: Boolean,val source: AnyRef):
    /** Stamp of the last session that used the entry; guarded by the pool's lock. Old entries are evicted first. */
    var lastUse=0L
  /** One stamp per session, so the weights of one model share a stamp and leave together. */
  val clock=new java.util.concurrent.atomic.AtomicLong()
  private val evictionCount=new java.util.concurrent.atomic.AtomicLong()
  /** True once kept weights had to make room on some device: uploads then go through cached Float16 host copies, which are much faster
    * to send again. */
  def evictedOnce: Boolean = evictionCount.get()>0 || limitBytes.nonEmpty
  @volatile private var lastTight=Long.MinValue
  /** Device memory was short just now (an eviction, or a model that had to be offloaded). */
  def markTight(): Unit = lastTight=System.nanoTime()
  /** True while device memory is known to be short: a limit is set, or it ran short within the last ten minutes. Callers then avoid
    * work that multiplies memory use, such as several generations at the same time. */
  def tight: Boolean = limitBytes.nonEmpty || (lastTight!=Long.MinValue && System.nanoTime()-lastTight<600_000_000_000L)
  /** Device memory this process uses outside the tracked chunks (context, cuBLAS, kernel modules); measured at 300 to 700 MiB. */
  private val overheadBytes=600L*1024L*1024L
  val lock=new Object
  private val entries=new java.util.HashMap[Int,java.util.IdentityHashMap[AnyRef,Array[Entry]]]()
  private val retired=new java.util.HashMap[Int,java.util.ArrayList[Entry]]()
  private val registered=new java.util.HashSet[Int]()

  private def slot(half: Boolean): Int = if half then 1 else 0

  /** Bytes this process holds on each device through sessions and this pool (device blocks, including freed blocks kept for reuse). */
  private val held=new java.util.concurrent.ConcurrentHashMap[Int,java.util.concurrent.atomic.AtomicLong]()
  def heldCounter(device: Int): java.util.concurrent.atomic.AtomicLong =
    held.computeIfAbsent(device,_ => new java.util.concurrent.atomic.AtomicLong())
  private val arenas=new java.util.HashMap[Int,java.util.ArrayList[CudaArena]]()
  /** A new arena for the kept weights that one session uploads. The weights of a model are uploaded by one session and evicted together,
    * so its chunks become empty and return to the device; in one arena for all models, a few surviving tensors kept every chunk alive. */
  def newArena(device: Int): CudaArena = arenas.synchronized {
    val created=new CudaArena(64L*1024L*1024L)
    arenas.computeIfAbsent(device,_ => new java.util.ArrayList[CudaArena]()).add(created)
    created
  }
  private def arenasOf(device: Int): Vector[CudaArena] = arenas.synchronized {
    val list=arenas.get(device)
    if list==null then Vector.empty
    else
      list.removeIf(a => a.ownedBytes==0L && a.liveBytes==0L && a.retired)
      scala.jdk.CollectionConverters.ListHasAsScala(list).asScala.toVector
  }
  /** Returns the empty chunks of every kept-weight arena of `device` to the driver. */
  def trimArenas(device: Int,rawFree: (Long,Long) => Unit): Long = arenasOf(device).map(_.trim(rawFree)).sum
  /** (live bytes, chunk bytes) of the kept weights of `device`. */
  def arenaBytes(device: Int): (Long,Long) = { val all=arenasOf(device);(all.map(_.liveBytes).sum,all.map(_.ownedBytes).sum) }
  /** `SCALATORCH_CUDA_VRAM_LIMIT_MB`: use at most this much device memory (all sessions and kept weights together). */
  val limitBytes: Option[Long] =
    sys.env.get("SCALATORCH_CUDA_VRAM_LIMIT_MB").flatMap(_.trim.toLongOption).filter(_ > 0).map(_*1024L*1024L)
  /** Free device memory as sessions should see it: what the driver reports, capped by the configured limit. */
  def governedFree(device: Int,driverFree: Long): Long =
    limitBytes.fold(driverFree)(limit => math.max(0L,math.min(driverFree,limit-overheadBytes-heldCounter(device).get())))

  /** The entry for `key`, marked as used by the session that owns `touched` (until [[releaseUsers]]). A used entry is never evicted. */
  def acquire(device: Int,key: AnyRef,half: Boolean,source: AnyRef,touched: java.util.Set[CudaAllocation],stamp: Long): Entry =
    entries.synchronized {
      val byKey=entries.get(device)
      if byKey==null then null
      else
        val pair=byKey.get(key)
        val found=if pair==null then null else pair(slot(half))
        if found!=null && (found.source eq source) then
          if touched.add(found.allocation) then found.allocation.users+=1
          found.lastUse=stamp
          found
        else null
    }

  def put(device: Int,key: AnyRef,entry: Entry,touched: java.util.Set[CudaAllocation],stamp: Long): Unit =
    val firstOnDevice=entries.synchronized {
      val byKey=entries.computeIfAbsent(device,_ => new java.util.IdentityHashMap[AnyRef,Array[Entry]]())
      val pair=Option(byKey.get(key)).getOrElse(new Array[Entry](2))
      val old=pair(slot(entry.half))
      if old!=null then retired.computeIfAbsent(device,_ => new java.util.ArrayList[Entry]()).add(old) // may still be in use: freed once unused
      pair(slot(entry.half))=entry
      byKey.put(key,pair)
      if touched.add(entry.allocation) then entry.allocation.users+=1
      entry.lastUse=stamp
      registered.add(device)
    }
    if firstOnDevice then CudaContextKeeper.onRelease(device,() => freeAll(device))

  /** A session ends: its entries may be evicted again. */
  def releaseUsers(touched: java.util.Set[CudaAllocation]): Unit =
    entries.synchronized {
      touched.forEach(a => a.users=math.max(0,a.users-1))
      touched.clear()
    }

  /** Bytes of the kept weights for `keys` that an open session uses right now (they cannot be evicted and need no upload). */
  def residentInUse(device: Int,keys: Iterable[AnyRef],half: Boolean): Long =
    entries.synchronized {
      val byKey=entries.get(device)
      if byKey==null then 0L
      else
        var total=0L
        keys.foreach { key =>
          val pair=byKey.get(key)
          if pair!=null then
            val e=pair(slot(half))
            if e!=null && e.allocation.users>0 then total+=e.allocation.bytes
        }
        total
    }

  /** Bytes of kept weights that no open session uses. */
  def unusedBytes(device: Int): Long =
    entries.synchronized {
      var total=0L
      val byKey=entries.get(device)
      if byKey!=null then byKey.values().forEach(pair => pair.foreach(e => if e!=null && e.allocation.users==0 then total+=e.allocation.bytes))
      val old=retired.get(device)
      if old!=null then old.forEach(e => if e.allocation.users==0 then total+=e.allocation.bytes)
      total
    }

  /** Removes kept weights that no open session uses, least recently used first, until about `wanted` bytes are found, and returns them;
    * the caller frees the device memory (its context is current). This is how a small GPU runs a big model: the text encoders leave
    * when the U-Net needs the room, the U-Net when the VAE does. */
  def evictUnused(device: Int,wanted: Long): Vector[Entry] =
    entries.synchronized {
      val candidates=scala.collection.mutable.ArrayBuffer.empty[(Entry,AnyRef,Int)]
      val byKey=entries.get(device)
      if byKey!=null then byKey.forEach { (key,pair) =>
        var i=0
        while i<pair.length do
          val e=pair(i)
          if e!=null && e.allocation.users==0 then candidates+=((e,key,i))
          i+=1
      }
      val old=retired.get(device)
      if old!=null then old.forEach(e => if e.allocation.users==0 then candidates+=((e,null,-1)))
      val result=Vector.newBuilder[Entry]
      var found=0L
      val ordered=candidates.sortBy((e,key,_) => (if key==null then 0 else 1,e.lastUse)).iterator
      while found<wanted && ordered.hasNext do
        val (e,key,index)=ordered.next()
        if key==null then old.remove(e)
        else
          val pair=byKey.get(key)
          pair(index)=null
          if pair.forall(_ == null) then byKey.remove(key)
        found+=e.allocation.bytes
        result+=e
      val out=result.result()
      if out.nonEmpty then { evictionCount.incrementAndGet();markTight() }
      out
    }

  /** Runs with the context current (from the keeper). */
  private def freeAll(device: Int): Unit =
    val all=entries.synchronized {
      registered.remove(device)
      val result=scala.collection.mutable.ArrayBuffer.empty[Entry]
      Option(entries.remove(device)).foreach(_.values().forEach(pair => pair.foreach(e => if e!=null then result+=e)))
      Option(retired.remove(device)).foreach(_.forEach(e => result+=e))
      result
    }
    arenasOf(device).foreach(_.releaseAll { (address,bytes) => CudaContextKeeper.freeMemory(address);heldCounter(device).addAndGet(-bytes) })
