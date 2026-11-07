package scalatorch.ffm

import java.lang.foreign.{Arena,MemorySegment}

/** Reusable confined native memory for synchronous CUDA host-to-device uploads. */
private[ffm] final class CudaHostStagingBuffer extends AutoCloseable:
  private var arena: Arena=null
  private var buffer: MemorySegment=null
  private var capacity=0L
  private var closed=false

  def segment(bytes: Long): MemorySegment =
    require(bytes>0,"staging segment must be non-empty")
    if closed then throw IllegalStateException("CUDA host staging buffer is closed")
    if buffer==null || bytes>capacity then
      val replacementArena=Arena.ofConfined()
      val replacement=try replacementArena.allocate(bytes,4)
      catch case error: Throwable =>
        try replacementArena.close() catch case cleanup: Throwable => error.addSuppressed(cleanup)
        throw error
      val previous=arena
      arena=replacementArena
      buffer=replacement
      capacity=bytes
      if previous!=null then previous.close()
    buffer.asSlice(0,bytes)

  def close(): Unit =
    if !closed then
      closed=true
      val previous=arena
      arena=null
      buffer=null
      capacity=0L
      if previous!=null then previous.close()
