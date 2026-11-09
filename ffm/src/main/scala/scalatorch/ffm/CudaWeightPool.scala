package scalatorch.ffm

import scalatorch.*

/** Device-resident weights shared by all sessions on a device (and by the sessions of different threads). A generation opens several sessions
  * and a batch of images many more; uploading the same few GB of weights every time cost more than the sampling of a small image.
  * Entries are keyed by the identity of the parameter and checked against the identity of the host tensor they were uploaded from.
  * They are freed together with the kept CUDA context ([[CudaContextKeeper]]) after its idle time, or explicitly. */
private[ffm] object CudaWeightPool:
  final class Entry(val allocation: CudaAllocation,val shape: Shape,val half: Boolean,val source: AnyRef)
  val lock=new Object
  private val entries=new java.util.HashMap[Int,java.util.IdentityHashMap[AnyRef,Array[Entry]]]()
  private val retired=new java.util.HashMap[Int,java.util.ArrayList[Entry]]()
  private val registered=new java.util.HashSet[Int]()

  private def slot(half: Boolean): Int = if half then 1 else 0

  def get(device: Int,key: AnyRef,half: Boolean,source: AnyRef): Entry =
    entries.synchronized {
      val byKey=entries.get(device)
      if byKey==null then null
      else
        val pair=byKey.get(key)
        val found=if pair==null then null else pair(slot(half))
        if found!=null && (found.source eq source) then found else null
    }

  def put(device: Int,key: AnyRef,entry: Entry): Unit =
    val firstOnDevice=entries.synchronized {
      val byKey=entries.computeIfAbsent(device,_ => new java.util.IdentityHashMap[AnyRef,Array[Entry]]())
      val pair=Option(byKey.get(key)).getOrElse(new Array[Entry](2))
      val old=pair(slot(entry.half))
      if old!=null then retired.computeIfAbsent(device,_ => new java.util.ArrayList[Entry]()).add(old) // may still be in use: freed with the rest
      pair(slot(entry.half))=entry
      byKey.put(key,pair)
      registered.add(device)
    }
    if firstOnDevice then CudaContextKeeper.onRelease(device,() => freeAll(device))

  /** Runs with the context current (from the keeper). */
  private def freeAll(device: Int): Unit =
    val all=entries.synchronized {
      registered.remove(device)
      val result=scala.collection.mutable.ArrayBuffer.empty[Entry]
      Option(entries.remove(device)).foreach(_.values().forEach(pair => pair.foreach(e => if e!=null then result+=e)))
      Option(retired.remove(device)).foreach(_.forEach(e => result+=e))
      result
    }
    all.foreach(e => CudaContextKeeper.freeMemory(e.allocation.address))
