package scalatorch.autograd

import scalatorch.*
import org.apache.commons.numbers.gamma.Erfc

enum GeluApproximation:
  case Erf, Tanh

/** Gaussian-error GELU and explicitly selected tanh approximation, with first-order derivatives. */
private[autograd] object GeluOps:
  def apply(input: Variable,approximation: GeluApproximation): Variable =
    val x=input.data.toFloatArray
    val out=new Array[Float](x.length)
    val derivatives=new Array[Double](x.length)
    val root2=math.sqrt(2.0)
    val root2pi=math.sqrt(2.0*math.Pi)
    val tanhScale=math.sqrt(2.0/math.Pi)
    var i=0
    while i<x.length do
      val v=x(i).toDouble
      val (result,derivative)=
        if v==Double.NegativeInfinity then (-0.0,0.0)
        else if v==Double.PositiveInfinity then (Double.PositiveInfinity,1.0)
        else approximation match
          case GeluApproximation.Erf =>
            val cdf=.5*Erfc.value(-v/root2)
            (v*cdf,cdf+v*math.exp(-.5*v*v)/root2pi)
          case GeluApproximation.Tanh =>
            val t=math.tanh(tanhScale*(v+.044715*v*v*v))
            val half=.5*(1+t)
            (v*half,half+.5*v*(1-t*t)*tanhScale*(1+3*.044715*v*v))
      out(i)=result.toFloat
      derivatives(i)=derivative
      i+=1
    Variable.result(Tensor.fromFloats(out,input.data.shape),Vector(input -> ((g: Tensor[Float]) =>
      val upstream=g.toFloatArray
      var j=0
      while j<upstream.length do
        upstream(j)=(if derivatives(j)==0 then 0.0 else upstream(j).toDouble*derivatives(j)).toFloat
        j+=1
      Tensor.fromFloats(upstream,input.data.shape))))
