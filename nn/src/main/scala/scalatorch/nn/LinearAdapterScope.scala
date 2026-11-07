package scalatorch.nn

/** Request-local additive adapters keyed by the identity of a base Linear weight. */
object LinearAdapterScope:
  type Applied = (LowRankAdapter, Float)

  private val active = new ThreadLocal[Map[Parameter, Vector[Applied]]]:
    override def initialValue(): Map[Parameter, Vector[Applied]] = Map.empty
  private val filter = new ThreadLocal[Applied => Boolean]:
    override def initialValue(): Applied => Boolean = _ => true
  private val multiplierOverride = new ThreadLocal[Map[LowRankAdapter, Float]]:
    override def initialValue(): Map[LowRankAdapter, Float] = Map.empty

  def forWeight(weight: Parameter): Vector[Applied] =
    val ovr = multiplierOverride.get()
    val raw = active.get().getOrElse(weight, Vector.empty).filter(filter.get())
    if ovr.isEmpty then raw
    else raw.flatMap { case (adapter, mult) =>
      val eff = ovr.getOrElse(adapter, mult)
      if eff == 0f then None else Some((adapter, eff))
    }

  /** Temporarily override per-adapter multipliers (e.g. for Composable LoRA branch evaluation). */
  def withMultipliers[A](overrides: Map[LowRankAdapter, Float])(body: => A): A =
    if overrides.isEmpty then body
    else
      val previous = multiplierOverride.get()
      multiplierOverride.set(previous ++ overrides)
      try body finally multiplierOverride.set(previous)

  /** Temporarily filter adapters without changing text-conditioning or outer scopes. */
  def withFilter[A](predicate: Applied => Boolean)(body: => A): A =
    val previous=filter.get()
    filter.set(item=>previous(item) && predicate(item))
    try body finally filter.set(previous)

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
