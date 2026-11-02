package scalatorch.cpu

import scalatorch.*

private[cpu] object CpuAdamW:
  private final class State(saved: AdamWMoments) extends AdamWState[Tensor[Float]]:
    private var closed=false
    private def check(): Unit = if closed then throw IllegalStateException("Adam state is closed")
    def shape: Shape = { check();saved.shape }
    def step: Long = { check();saved.step }
    def snapshot(): AdamWMoments = { check();saved }
    def close(): Unit = closed=true
  def restore(saved: AdamWMoments): AdamWState[Tensor[Float]] =
    saved.validate()
    new State(saved)
  def update(weight: Tensor[Float],gradient: Tensor[Float],previous: Option[AdamWState[Tensor[Float]]],config: AdamWConfig): AdamWUpdate[Tensor[Float]] =
    require(weight.shape==gradient.shape,"Adam gradient shape mismatch")
    val old=previous.map {
      case s: State =>
        require(s.shape==weight.shape,"Adam state shape mismatch")
        s.snapshot()
      case _ => throw IllegalArgumentException("Adam state belongs to another backend")
    }
    val weights=weight.toFloatArray;val grads=gradient.toFloatArray
    require(weights.forall(_.isFinite) && grads.forall(_.isFinite),"nonfinite optimizer input")
    val t=Math.addExact(old.fold(0L)(_.step),1L)
    val m=old.fold(new Array[Double](weights.length))(_.first.toDoubleArray)
    val v=old.fold(new Array[Double](weights.length))(_.second.toDoubleArray)
    val next=new Array[Float](weights.length)
    val c1=1.0-math.pow(config.beta1,t.toDouble);val c2=1.0-math.pow(config.beta2,t.toDouble)
    weights.indices.foreach { i =>
      val g=grads(i).toDouble
      m(i)=config.beta1*m(i)+(1-config.beta1)*g
      v(i)=config.beta2*v(i)+(1-config.beta2)*g*g
      next(i)=(weights(i)*(1-config.learningRate*config.weightDecay)-
        config.learningRate*(m(i)/c1)/(math.sqrt(v(i)/c2)+config.epsilon)).toFloat
    }
    require(next.forall(_.isFinite),"nonfinite optimizer update")
    val state=restore(AdamWMoments(t,Tensor.fromDoubles(m,weight.shape),Tensor.fromDoubles(v,weight.shape)))
    AdamWUpdate(Tensor.fromFloats(next,weight.shape),state)
