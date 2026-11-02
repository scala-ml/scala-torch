package scalatorch.nn

import scalatorch.*
import scala.collection.mutable

private[nn] final case class OptimizerSlot(step: Long, first: Vector[Double], second: Vector[Double])

abstract class Optimizer(input: Seq[Parameter]):
  protected val parameters = input.toVector.distinct.filter(_.trainable)
  private[nn] def optimized: Vector[Parameter] = parameters
  private[nn] def kind: String
  private[nn] def settings: Vector[(String, Double)]
  private[nn] def snapshot: Vector[Option[OptimizerSlot]]
  private[nn] def restore(saved: Vector[Option[OptimizerSlot]]): Unit
  private[nn] def validate(saved: Vector[Option[OptimizerSlot]]): Unit =
    require(saved.size == parameters.size, "optimizer slot count mismatch")
    saved.zip(parameters).foreach { (slot, p) => slot.foreach { s =>
      require(s.step > 0 && s.first.size.toLong == p.data.shape.numel && s.second.size == s.first.size,
        "invalid optimizer slot shape or counter")
      require(s.first.forall(_.isFinite) && s.second.forall(x => x.isFinite && x >= 0), "invalid optimizer moments")
    }}
  def zeroGrad(): Unit = parameters.foreach(_.zeroGrad())
  def step(): Unit
  protected def checked(parameter: Parameter): Option[(Array[Float], Array[Float])] = parameter.grad.map { gradient =>
    val weights = parameter.data.toFloatArray
    val grads = gradient.toFloatArray
    require(weights.forall(_.isFinite) && grads.forall(_.isFinite), "nonfinite optimizer input")
    (weights, grads)
  }
  protected def tensor(parameter: Parameter, values: Array[Float]): Tensor[Float] =
    require(values.forall(_.isFinite), "nonfinite optimizer update")
    Tensor.fromFloats(values, parameter.data.shape)

final class SGD(input: Seq[Parameter], learningRate: Double, weightDecay: Double = 0.0) extends Optimizer(input):
  require(learningRate.isFinite && learningRate > 0, "invalid learning rate")
  require(weightDecay.isFinite && weightDecay >= 0, "invalid weight decay")
  private[nn] def kind = "sgd"
  private[nn] def settings = Vector("learningRate" -> learningRate, "weightDecay" -> weightDecay)
  private[nn] def snapshot = Vector.fill(parameters.size)(None)
  override private[nn] def validate(saved: Vector[Option[OptimizerSlot]]): Unit =
    super.validate(saved)
    require(saved.forall(_.isEmpty), "SGD has no moment slots")
  private[nn] def restore(saved: Vector[Option[OptimizerSlot]]): Unit = validate(saved)
  def step(): Unit =
    val updates = parameters.flatMap { p => checked(p).map { (weights, grads) =>
      val next = weights.indices.map(i => (weights(i) - learningRate * (grads(i) + weightDecay * weights(i))).toFloat).toArray
      p -> tensor(p, next)
    }}
    updates.foreach((p, next) => p.update(next))

final class AdamW(input: Seq[Parameter], learningRate: Double, beta1: Double = 0.9, beta2: Double = 0.999,
    epsilon: Double = 1e-8, weightDecay: Double = 0.01) extends Optimizer(input):
  require(learningRate.isFinite && learningRate > 0, "invalid learning rate")
  require(beta1.isFinite && beta1 >= 0 && beta1 < 1 && beta2.isFinite && beta2 >= 0 && beta2 < 1, "invalid Adam betas")
  require(epsilon.isFinite && epsilon > 0 && weightDecay.isFinite && weightDecay >= 0, "invalid Adam epsilon or decay")
  private case class Moments(step: Long, first: Array[Double], second: Array[Double])
  private val state = mutable.Map.empty[Parameter, Moments]
  private[nn] def kind = "adamw"
  private[nn] def settings = Vector("learningRate" -> learningRate, "beta1" -> beta1, "beta2" -> beta2,
    "epsilon" -> epsilon, "weightDecay" -> weightDecay)
  private[nn] def snapshot = parameters.map(p => state.get(p).map(s => OptimizerSlot(s.step, s.first.toVector, s.second.toVector)))
  private[nn] def restore(saved: Vector[Option[OptimizerSlot]]): Unit =
    validate(saved)
    val next = parameters.zip(saved).flatMap((p, s) => s.map(slot =>
      p -> Moments(slot.step, slot.first.toArray, slot.second.toArray)))
    state.clear()
    state ++= next
  def step(): Unit =
    val updates = parameters.flatMap { p => checked(p).map { (weights, grads) =>
      val previous = state.getOrElse(p, Moments(0, new Array[Double](weights.length), new Array[Double](weights.length)))
      val t = Math.addExact(previous.step, 1L)
      val first = new Array[Double](weights.length)
      val second = new Array[Double](weights.length)
      val next = new Array[Float](weights.length)
      val correction1 = 1.0 - math.pow(beta1, t.toDouble)
      val correction2 = 1.0 - math.pow(beta2, t.toDouble)
      var i = 0
      while i < weights.length do
        val g = grads(i).toDouble
        first(i) = beta1 * previous.first(i) + (1 - beta1) * g
        second(i) = beta2 * previous.second(i) + (1 - beta2) * g * g
        next(i) = (weights(i) * (1 - learningRate * weightDecay) -
          learningRate * (first(i) / correction1) / (math.sqrt(second(i) / correction2) + epsilon)).toFloat
        i += 1
      (p, tensor(p, next), Moments(t, first, second))
    }}
    updates.foreach { (p, next, moments) => p.update(next); state(p) = moments }
