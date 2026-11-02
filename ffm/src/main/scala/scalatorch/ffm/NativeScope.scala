package scalatorch.ffm

import java.lang.foreign.{Arena, MemorySegment}

/** Thread-confined host memory scope. Device operations need their own completion ownership. */
final class NativeScope extends AutoCloseable:
  private val arena = Arena.ofConfined()
  private var closed = false
  def allocate(bytes: Long, alignment: Long = 1L): MemorySegment =
    checkOpen()
    arena.allocate(bytes, alignment)
  private def checkOpen(): Unit =
    if closed then throw IllegalStateException("native scope is closed")
  override def close(): Unit =
    if !closed then
      arena.close()
      closed = true

object NativeScope:
  def use[A](f: NativeScope => A): A =
    val scope = new NativeScope()
    try f(scope)
    finally scope.close()
