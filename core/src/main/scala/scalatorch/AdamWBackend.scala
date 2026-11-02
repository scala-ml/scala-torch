package scalatorch

final case class AdamWConfig(learningRate: Double = .001,beta1: Double = .9,beta2: Double = .999,
    epsilon: Double = 1e-8,weightDecay: Double = .01):
  require(learningRate.isFinite && learningRate>0,"invalid learning rate")
  require(beta1.isFinite && beta1>=0 && beta1<1 && beta2.isFinite && beta2>=0 && beta2<1,"invalid Adam betas")
  require(epsilon.isFinite && epsilon>0 && weightDecay.isFinite && weightDecay>=0,"invalid Adam epsilon or decay")

/** Immutable explicit host export; moment tensors use logical Double values. */
final case class AdamWMoments(step: Long,first: Tensor[Double],second: Tensor[Double]):
  def shape: Shape = first.shape
  private[scalatorch] def validate(): Unit =
    require(step>0 && first.shape==second.shape && shape.numel<=Int.MaxValue,"invalid Adam moment shape or counter")
    require(first.toDoubleArray.forall(_.isFinite) && second.toDoubleArray.forall(x => x.isFinite && x>=0),"invalid Adam moments")

/** Independently owned immutable moment snapshot; close is idempotent. */
trait AdamWState[T] extends AutoCloseable:
  def shape: Shape
  def step: Long
  def snapshot(): AdamWMoments

/** Caller owns both the new weight handle and moment state; inputs remain borrowed and unchanged. */
final case class AdamWUpdate[T](value: T,state: AdamWState[T])

trait AdamWBackend[T] extends TrainingBackend[T]:
  def adamW(weight: T,gradient: T,previous: Option[AdamWState[T]],config: AdamWConfig): AdamWUpdate[T]
  def restoreAdamW(saved: AdamWMoments): AdamWState[T]
