package scalatorch.models

import scalatorch.autograd.Variable

/** Euler ancestral update with caller-supplied standard-normal noise. */
final case class EulerAncestralScheduler(schedule: EulerSchedule,
    prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,maxElements: Long=134217728L):
  require(maxElements>=0,"negative Euler ancestral allocation limit")
  require(prediction!=DiffusionPrediction.Sample,"Euler ancestral sample prediction is not supported")

  def step(index: Int,modelOutput: Variable,sample: Variable,noise: Variable): EulerStep =
    require(index>=0 && index<schedule.timesteps.size,"Euler ancestral step index out of range")
    require(modelOutput.data.shape==sample.data.shape && noise.data.shape==sample.data.shape,
      "Euler ancestral tensor shape mismatch")
    require(sample.data.shape.numel<=maxElements && sample.data.shape.numel<=Int.MaxValue,
      "Euler ancestral tensor exceeds allocation limit")
    val sigma=schedule.sigmas(index)
    val next=schedule.sigmas(index+1)
    val original=prediction match
      case DiffusionPrediction.Epsilon => sample-modelOutput*sigma.toFloat
      case DiffusionPrediction.VPrediction =>
        modelOutput*(-sigma/math.hypot(sigma,1)).toFloat+sample*(1/(sigma*sigma+1)).toFloat
      case DiffusionPrediction.Sample => throw IllegalStateException("unsupported Euler ancestral prediction")
    val up=if next==0 then 0.0 else next*math.sqrt(math.max(0.0,1.0-(next/sigma)*(next/sigma)))
    val down=math.sqrt(math.max(0.0,next*next-up*up))
    val derivative=(sample-original)*(1/sigma).toFloat
    EulerStep(sample+derivative*(down-sigma).toFloat+noise*up.toFloat,original)
