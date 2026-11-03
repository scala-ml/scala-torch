package scalatorch.nn

/** Request-local additive adapters keyed by the identity of a base Linear weight. */
object LinearAdapterScope:
  type Applied = (LowRankAdapter, Float)

  private val active = new ThreadLocal[Map[Parameter, Vector[Applied]]]:
    override def initialValue(): Map[Parameter, Vector[Applied]] = Map.empty

  def forWeight(weight: Parameter): Vector[Applied] = active.get().getOrElse(weight, Vector.empty)

  def parameters: Vector[Parameter] =
    active.get().valuesIterator.flatten.flatMap { case (adapter, _) => Iterator(adapter.a, adapter.b) }.toVector.distinct

  /** Nested scopes inherit unrelated targets and replace targets explicitly supplied by the inner scope. */
  def withAdapters[A](adapters: Map[Parameter, Vector[Applied]])(body: => A): A =
    adapters.valuesIterator.flatten.foreach { case (_, multiplier) =>
      require(multiplier.isFinite, "adapter multiplier must be finite")
    }
    val previous = active.get()
    active.set(previous ++ adapters)
    try body
    finally active.set(previous)
