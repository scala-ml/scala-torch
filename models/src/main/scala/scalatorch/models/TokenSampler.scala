package scalatorch.models

/** Temperature zero selects greedy decoding. Top-k zero keeps the whole vocabulary. */
final case class SamplingConfig(temperature: Double = 0, topK: Int = 0, topP: Double = 1, seed: Long = 0):
  require(temperature.isFinite && temperature >= 0, "temperature must be finite and nonnegative")
  require(topK >= 0, "top-k must be nonnegative")
  require(topP > 0 && topP <= 1, "top-p must be in (0,1]")

/** One mutable seeded stream; use a separate instance per generation/request. */
final class TokenSampler(config: SamplingConfig):
  private val random = new java.util.Random(config.seed)
  def sample(logits: Array[Float]): Long =
    val probabilities = TokenSampler.probabilities(logits,config)
    if config.temperature == 0 then probabilities.indexWhere(_ > 0).toLong
    else
      val draw = random.nextDouble()
      var cumulative = 0.0
      var last = 0
      var i = 0
      while i < probabilities.length do
        if probabilities(i) > 0 then last = i
        cumulative += probabilities(i)
        if draw < cumulative then return i.toLong
        i += 1
      last.toLong // A rounded sum slightly below one must not select an excluded token.

object TokenSampler:
  /** Dense categorical probabilities, useful for inspecting the applied sampling policy. */
  def probabilities(logits: Array[Float], config: SamplingConfig): Array[Double] =
    require(logits.nonEmpty && logits.forall(_.isFinite), "sampling requires nonempty finite logits")
    val output = new Array[Double](logits.length)
    if config.temperature == 0 then
      var best = 0
      var i = 1
      while i < logits.length do
        if logits(i) > logits(best) then best = i
        i += 1
      output(best) = 1
    else
      val ranked = logits.indices.sortBy(i => (if logits(i) == 0f then 0.0 else -logits(i).toDouble,i))
      val count = if config.topK == 0 then logits.length else math.min(config.topK,logits.length)
      val max = logits(ranked.head).toDouble
      val weights = Array.tabulate(count)(i => math.exp((logits(ranked(i)).toDouble-max)/config.temperature))
      val threshold = config.topP * weights.sum
      var retained = 0
      var mass = 0.0
      while retained < count && (config.topP == 1 || retained == 0 || mass < threshold) do
        mass += weights(retained)
        retained += 1
      var i = 0
      while i < retained do
        output(ranked(i)) = weights(i)/mass
        i += 1
    output
