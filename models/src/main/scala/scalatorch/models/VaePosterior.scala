package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable

/** NCHW diagonal Gaussian; log variance is clipped to [-30,20] before reparameterization. */
final class VaePosterior private (moments: Variable):
  private val dims=moments.data.shape.dims
  require(dims.size==4 && dims(1)>0 && dims(1)%2==0 && dims(2)>0 && dims(3)>0,
    "VAE posterior requires NCHW moments with positive even channels and spatial dimensions")
  val mean: Variable=moments.narrow(1,0,dims(1)/2)
  val logVariance: Variable=moments.narrow(1,dims(1)/2,dims(1)/2).clamp(-30f,20f)
  val variance: Variable=logVariance.exp()
  val stddev: Variable=(logVariance*.5f).exp()
  def mode: Variable=mean

  /** Supplied standard-normal noise is a constant, enabling exact replay across runtimes. */
  def sample(noise: Tensor[Float]): Variable =
    require(noise.shape==mean.data.shape,"VAE posterior noise shape mismatch")
    mean+stddev*Variable(noise,requiresGrad=false)

  /** JVM java.util.Random Gaussian sequence; not a PyTorch seed compatibility promise. */
  def sample(seed: Long): Variable =
    val random=new java.util.Random(seed)
    val noise=Array.fill(mean.data.shape.numel.toInt)(random.nextGaussian().toFloat)
    sample(Tensor.fromFloats(noise,mean.data.shape))

  /** KL divergence to the standard normal, summed over channels and spatial axes per image. */
  def kl(): Variable =
    val one=Variable(Tensor.fromFloats(Array(1f),Shape.from(Seq.empty)),requiresGrad=false)
    (mean*mean+variance-one-logVariance).sum(Seq(1,2,3))*.5f

object VaePosterior:
  def apply(moments: Variable): VaePosterior=new VaePosterior(moments)
