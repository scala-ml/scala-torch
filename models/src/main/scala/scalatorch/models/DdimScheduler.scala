package scalatorch.models

import scalatorch.autograd.Variable

/** DDIM inference timesteps and VP noise levels; NUI uses set_alpha_to_one=false. */
final class DdimSchedule private (val layout: EulerSchedule,private[models] val alphas: Vector[Double],
    private[models] val previousAlphas: Vector[Double])

object DdimSchedule:
  def apply(noise: DiffusionNoiseSchedule,steps: Int,spacing: DiffusionTimestepSpacing,stepsOffset: Int=0): DdimSchedule =
    val count=noise.betas.size
    require(steps>0 && steps<=10000 && steps<=count,"invalid DDIM inference step count")
    val times=Vector.tabulate(steps) { i =>
      val index=spacing match
        case DiffusionTimestepSpacing.Linspace => if steps==1 then 0 else math.rint((count-1).toDouble*(steps-1-i)/(steps-1)).toInt
        case DiffusionTimestepSpacing.Leading => (steps-1-i)*(count/steps)+stepsOffset
        case DiffusionTimestepSpacing.Trailing => math.rint(count.toDouble-i*count.toDouble/steps).toInt-1
      require(index>=0 && index<count,"DDIM timestep outside training schedule")
      index
    }
    val training=noise.floatTrainingSigmas.map(s=>1.0/(1.0+s.toDouble*s.toDouble))
    val alphas=times.map(training)
    val previous=times.map(t=>training(math.max(0,t-count/steps)))
    val sigmas=times.map(i=>noise.floatTrainingSigmas(i).toDouble) :+ 0.0
    new DdimSchedule(EulerSchedule(times.map(_.toFloat),sigmas,1.0),alphas,previous)

/** DDIM in VP coordinates; supplied noise follows the caller's seeded stream. */
final class DdimScheduler(schedule: DdimSchedule,prediction: DiffusionPrediction,clipSample: Boolean,clipRange: Float=1f):
  require(clipRange.isFinite && clipRange>0,"invalid DDIM clipping range")
  def step(index: Int,model: Variable,sample: Variable,eta: Double=0.0,noise: Option[Variable]=None): Variable =
    require(index>=0 && index<schedule.alphas.size,"DDIM step index out of range")
    require(model.data.shape==sample.data.shape,"DDIM tensor shape mismatch")
    require(eta.isFinite,"DDIM eta must be finite")
    require(eta<=0 || noise.exists(_.data.shape==sample.data.shape),"DDIM eta requires matching variance noise")
    val alpha=schedule.alphas(index)
    val previous=schedule.previousAlphas(index)
    val a=math.sqrt(alpha).toFloat
    val b=math.sqrt(1-alpha).toFloat
    val (original,epsilon)=prediction match
      case DiffusionPrediction.Epsilon => ((sample-model*b)*(1f/a),model)
      case DiffusionPrediction.Sample => (model,(sample-model*a)*(1f/b))
      case DiffusionPrediction.VPrediction => (sample*a-model*b,model*a+sample*b)
    val clean=if clipSample then original.clamp(-clipRange,clipRange) else original
    val variance=math.max(0.0,((1-previous)/(1-alpha))*(1-alpha/previous))
    val deviation=eta*math.sqrt(variance)
    val direction=1-previous-deviation*deviation
    require(direction>= -1e-12,"DDIM eta produces an invalid variance")
    val result=clean*math.sqrt(previous).toFloat+epsilon*math.sqrt(math.max(0.0,direction)).toFloat
    if eta>0 then result+noise.get*deviation.toFloat else result
