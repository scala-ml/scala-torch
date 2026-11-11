package scalatorch.ffm

/** Device memory handed out from a few large chunks: best fit, split on allocation, neighbours merged on free.
  *
  * One driver allocation per tensor was wasteful in two ways. Freed blocks could only be reused for a tensor of exactly the same size, so a
  * session kept several times the memory it ever used at once; and on Windows every driver allocation is padded, about 0.4 MiB per tensor
  * for the thousands of weight tensors of a model. A chunk is one driver allocation; requests above half a chunk get a chunk of their own.
  *
  * All methods are synchronized: the arena of a session is used by one thread, the arena of the kept weights by every session of a device.
  * The driver calls are passed in, because they need the context of the calling session. */
private[ffm] final class CudaArena(chunkBytes: Long):
  private final class Block(val address: Long, var size: Long, val chunk: Long)
  private val chunks = new java.util.HashMap[Long, java.lang.Long]()                        // chunk address -> bytes
  private val free = new java.util.TreeMap[Long, Block]()                                   // by address
  private val bySize = new java.util.TreeMap[Long, java.util.TreeSet[java.lang.Long]]()     // size -> addresses of free blocks
  private val live = new java.util.HashMap[Long, Block]()
  private var owned = 0L
  private var used = 0L

  def ownedBytes: Long = synchronized(owned)
  def liveBytes: Long = synchronized(used)

  private def align(bytes: Long): Long = (math.max(bytes, 1L) + 255L) & ~255L

  private def addFree(block: Block): Unit =
    free.put(block.address, block)
    bySize.computeIfAbsent(block.size, _ => new java.util.TreeSet[java.lang.Long]()).add(block.address)

  private def removeFree(block: Block): Unit =
    free.remove(block.address)
    val set = bySize.get(block.size)
    if set != null then
      set.remove(block.address)
      if set.isEmpty then bySize.remove(block.size)

  private def fit(size: Long): Long =
    val entry = bySize.ceilingEntry(size)
    if entry == null then 0L
    else
      val block = free.get(entry.getValue.first())
      removeFree(block)
      if block.size - size >= 4096L then
        addFree(new Block(block.address + size, block.size - size, block.chunk))
        block.size = size
      live.put(block.address, block)
      used += block.size
      block.address

  /** A block of at least `bytes`. `room(n)` is called before a new chunk of n bytes is requested from the driver (it may free memory, also
    * memory of this arena); `raw(n)` allocates the chunk and throws when the device has no room. */
  def allocate(bytes: Long, room: Long => Unit, raw: Long => Long, rawFree: (Long, Long) => Unit): Long = synchronized {
    val size = align(bytes)
    val found = fit(size)
    if found != 0L then found
    else
      val chunk = if size > chunkBytes / 2 then size else chunkBytes
      // Empty chunks that together are as large as the new one go back first. The arena then settles on a few chunks of the largest
      // sizes (smaller requests are cut from them) instead of keeping one chunk per large size that ever occurred.
      if emptyBytes() >= chunk then trim(rawFree)
      room(chunk)
      val again = fit(size)
      if again != 0L then again
      else
        val granted = try (chunk, raw(chunk)) catch case error: Throwable =>
          if chunk == size then throw error
          (size, raw(size)) // a full chunk does not fit any more: take exactly what is needed
        chunks.put(granted._2, granted._1)
        owned += granted._1
        addFree(new Block(granted._2, granted._1, granted._2))
        fit(size)
  }

  private def emptyBytes(): Long =
    var total = 0L
    free.values().forEach { b => if b.address == b.chunk && b.size == chunks.get(b.chunk).longValue() then total += b.size }
    total

  /** Bytes of the live block at `address` (0 when the address is not a live block of this arena). */
  def blockBytes(address: Long): Long = synchronized { val b = live.get(address); if b == null then 0L else b.size }

  def release(address: Long): Unit = synchronized {
    val block = live.remove(address)
    require(block != null, "CUDA arena: released an unknown block")
    used -= block.size
    var start = block.address
    var size = block.size
    val before = free.lowerEntry(block.address)
    if before != null && before.getValue.chunk == block.chunk && before.getValue.address + before.getValue.size == block.address then
      removeFree(before.getValue)
      start = before.getValue.address
      size += before.getValue.size
    val after = free.get(block.address + block.size)
    if after != null && after.chunk == block.chunk then
      removeFree(after)
      size += after.size
    addFree(new Block(start, size, block.chunk))
  }

  /** Returns the chunks without a live block to the driver; the number of bytes freed. */
  def trim(rawFree: (Long, Long) => Unit): Long = synchronized {
    var freed = 0L
    val empty = new java.util.ArrayList[Block]()
    free.values().forEach { b => if b.address == b.chunk && b.size == chunks.get(b.chunk).longValue() then empty.add(b) }
    empty.forEach { b =>
      removeFree(b)
      chunks.remove(b.chunk)
      owned -= b.size
      freed += b.size
      rawFree(b.address, b.size)
    }
    freed
  }

  /** Forgets every block and returns all chunks to the driver (the owner guarantees that nothing is in use). */
  def releaseAll(rawFree: (Long, Long) => Unit): Unit = synchronized {
    chunks.forEach((address, bytes) => rawFree(address, bytes))
    chunks.clear(); free.clear(); bySize.clear(); live.clear()
    owned = 0L; used = 0L
  }
