package scalatorch.models

import scalatorch.autograd.Variable

final case class HeunPrediction(sample: Variable,predictor: Variable,derivative: Variable,predictedOriginal: Variable,
    requiresSecondEvaluation: Boolean)
final case class HeunStep(previous: Variable,predictedOriginal: Variable)

/** Stateless sigma-space Heun update. The caller evaluates the model at predictor when requested. */
final case class HeunScheduler(schedule: EulerSchedule,prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,
    maxElements: Long=134217728L):
  require(maxElements>=0,"negative Heun allocation limit")
  private def check(value: Variable): Unit =
    require(value.data.shape.numel<=maxElements && value.data.shape.numel<=Int.MaxValue,"Heun tensor exceeds allocation limit")
  private def pair(a: Variable,b: Variable): Unit =
    require(a.data.shape==b.data.shape,"Heun tensor shape mismatch")
    check(a);check(b)
  private def original(index: Int,output: Variable,sample: Variable): Variable =
    val sigma=schedule.sigmas(index)
    prediction match
      case DiffusionPrediction.Epsilon => sample-output*sigma.toFloat
      case DiffusionPrediction.Sample => output
      case DiffusionPrediction.VPrediction =>
        val alpha=1/math.hypot(sigma,1)
        sample*(1/(sigma*sigma+1)).toFloat-output*(sigma*alpha).toFloat
  def scaleModelInput(index: Int,sample: Variable): Variable =
    require(index>=0 && index<schedule.timesteps.size,"Heun step index out of range")
    check(sample)
    sample*(1/math.hypot(schedule.sigmas(index),1)).toFloat
  def predict(index: Int,modelOutput: Variable,sample: Variable): HeunPrediction =
    require(index>=0 && index<schedule.timesteps.size,"Heun step index out of range")
    pair(modelOutput,sample)
    val sigma=schedule.sigmas(index)
    val clean=original(index,modelOutput,sample)
    val derivative=(sample-clean)*(1/sigma).toFloat
    val delta=(schedule.sigmas(index+1)-sigma).toFloat
    val predictor=sample+derivative*delta
    HeunPrediction(sample,predictor,derivative,clean,schedule.sigmas(index+1)>0)
  def correct(index: Int,first: HeunPrediction,nextModelOutput: Variable): HeunStep =
    require(index>=0 && index<schedule.timesteps.size && first.requiresSecondEvaluation,"Heun correction is not required")
    pair(nextModelOutput,first.predictor)
    val nextSigma=schedule.sigmas(index+1)
    val nextOriginal=original(index+1,nextModelOutput,first.predictor)
    val nextDerivative=(first.predictor-nextOriginal)*(1/nextSigma).toFloat
    val delta=(nextSigma-schedule.sigmas(index)).toFloat
    HeunStep(first.sample+(first.derivative+nextDerivative)*(delta*0.5f),first.predictedOriginal)
