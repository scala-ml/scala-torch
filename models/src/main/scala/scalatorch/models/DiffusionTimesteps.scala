package scalatorch.models

import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.nn.*

/** Deterministic CPU timestep features. Timesteps are conditioning constants, not trainable leaves. */
object DiffusionTimesteps:
  def sinusoidal(timesteps: Tensor[Float],width: Int,flipSinToCos: Boolean = false,frequencyShift: Double = 1.0,
      phaseScale: Double = 1.0,maxPeriod: Double = 10000.0,maxElements: Long = 16777216L): Tensor[Float] =
    require(timesteps.device==Device.Cpu && timesteps.dtype==DType.Float32 && timesteps.shape.rank==1,
      "timesteps must be a rank-one CPU Float32 tensor")
    val half=width/2
    require(width>=2 && frequencyShift.isFinite && half-frequencyShift>0,"invalid timestep width/frequency shift")
    require(phaseScale.isFinite && maxPeriod.isFinite && maxPeriod>=1,"invalid timestep phase scale/period")
    require(maxElements>=0,"negative timestep feature budget")
    val shape=Shape.from(Seq(timesteps.shape.dims.head,width.toLong))
    require(shape.numel<=maxElements && shape.numel<=Int.MaxValue,"timestep feature budget exceeded")
    val times=timesteps.toFloatArray
    require(times.forall(t => t.isFinite && (t.toDouble*phaseScale).isFinite),"nonfinite timestep phase")
    val out=new Array[Float](shape.numel.toInt)
    if times.nonEmpty then
      val frequencies=Array.tabulate(half)(i => math.exp(-math.log(maxPeriod)*i/(half-frequencyShift)))
      var row=0
      while row<times.length do
        var i=0
        while i<half do
          val phase=times(row).toDouble*phaseScale*frequencies(i)
          val sin=math.sin(phase).toFloat;val cos=math.cos(phase).toFloat
          out(row*width+i)=if flipSinToCos then cos else sin
          out(row*width+half+i)=if flipSinToCos then sin else cos
          i+=1
        row+=1
    Tensor.fromFloats(out,shape)

/** Learned Linear-SiLU-Linear timestep projection; supplied features can also be differentiable. */
final case class DiffusionTimeEmbedding(linear1: Linear,linear2: Linear) derives Module:
  require(linear1.weight.data.shape.dims.forall(_>0) && linear2.weight.data.shape.dims.forall(_>0) &&
    linear1.weight.data.shape.dims.head==linear2.weight.data.shape.dims(1),"incompatible timestep MLP widths")
  def apply(features: Variable): Variable =
    require(features.data.shape.rank==2 && features.data.shape.dims(1)==linear1.weight.data.shape.dims(1),
      "timestep MLP input must have matching batch/feature shape")
    linear2(linear1(features).silu())
