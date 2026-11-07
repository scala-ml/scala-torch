package scalatorch.models

import scalatorch.autograd.Variable

final case class DpmPlusPlusStep(previous: Variable,predictedOriginal: Variable)

/** Deterministic VP DPM++ order-two midpoint updates with caller-owned history.
  * Inputs are VP latents (no Euler input scaling); initial noise has unit scale.
  * Pass the previous step's predictedOriginal as history. Terminal zero uses order one.
  */
final case class DpmPlusPlus2M(sigmas: Vector[Double],
    prediction: DiffusionPrediction=DiffusionPrediction.Epsilon,maxElements: Long=134217728L):
  require(maxElements>=0,"negative DPM allocation limit")
  require(sigmas.size>=2 && sigmas.size<=1000001 && sigmas.last==0,"invalid DPM schedule")
  require(sigmas.dropRight(1).forall(s => s.isFinite && s>0 && s.toFloat.isFinite && s.toFloat>0),
    "invalid DPM sigma")
  require(sigmas.zip(sigmas.drop(1)).forall((a,b) => a>=b),"DPM sigmas must not increase")
  def step(index: Int,modelOutput: Variable,sample: Variable,
      previousOriginal: Option[Variable]=None): DpmPlusPlusStep =
    require(index>=0 && index<sigmas.size-1,"DPM step out of range")
    require(previousOriginal.isDefined==(index>0),"DPM history required exactly after first step")
    require(sample.data.shape.numel<=maxElements && sample.data.shape.numel<=Int.MaxValue,"DPM allocation limit")
    require(modelOutput.data.shape==sample.data.shape && previousOriginal.forall(_.data.shape==sample.data.shape),
      "DPM tensor shape mismatch")
    val s=sigmas(index)
    val alpha=1/math.hypot(s,1)
    val sigma=s*alpha
    val original=prediction match
      case DiffusionPrediction.Epsilon => (sample-modelOutput*sigma.toFloat)*(1/alpha).toFloat
      case DiffusionPrediction.Sample => modelOutput
      case DiffusionPrediction.VPrediction => sample*alpha.toFloat-modelOutput*sigma.toFloat
    val next=sigmas(index+1)
    if next==s then DpmPlusPlusStep(sample,original)
    else if next==0 then DpmPlusPlusStep(original,original)
    else
      val nextAlpha=1/math.hypot(next,1)
      val h=math.log(s)-math.log(next)
      require(h.isFinite && h>0,"unrepresentable DPM interval")
      val corrected=previousOriginal match
        case None => original
        case Some(old) if sigmas(index-1)>s =>
          val previousH=math.log(sigmas(index-1))-math.log(s)
          val coefficient=.5*h/previousH
          require(coefficient.toFloat.isFinite,"unrepresentable DPM history coefficient")
          original+(original-old)*coefficient.toFloat
        case Some(_) => original
      val result=sample*(next*nextAlpha/sigma).toFloat+
        corrected*(-nextAlpha*math.expm1(-h)).toFloat
      DpmPlusPlusStep(result,original)
