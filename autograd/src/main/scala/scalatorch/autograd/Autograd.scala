package scalatorch.autograd

/** Graph recording state is local to the calling thread. */
object Autograd:
  private val disabled = new ThreadLocal[Boolean]:
    override def initialValue(): Boolean = false
  private[autograd] def recording: Boolean = !disabled.get()
  def noGrad[A](body: => A): A =
    val previous = disabled.get()
    disabled.set(true)
    try body
    finally disabled.set(previous)
