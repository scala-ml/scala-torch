package scalatorch.models

/** Detail Daemon's NUI step envelope and noise-to-timestep transformation. */
final case class DetailDaemon(mode: String="both",amount: Double=.1,start: Double=.2,end: Double=.8,bias: Double=.5,
    exponent: Double=1,startOffset: Double=0,endOffset: Double=0,fade: Double=0,smooth: Boolean=true,
    trainingSigmas: Vector[Double]):
  require(Set("both","cond","uncond")(mode),"invalid detail daemon mode")
  require(Seq(amount,start,end,bias,exponent,startOffset,endOffset,fade).forall(_.isFinite) &&
    amount>= -1 && amount<=1 && start>=0 && start<=1 && end>=0 && end<=1 && bias>=0 && bias<=1 &&
    exponent>=0 && exponent<=10 && startOffset>= -1 && startOffset<=1 && endOffset>= -1 && endOffset<=1 && fade>=0 && fade<=1,
    "invalid detail daemon settings")
  require(trainingSigmas.nonEmpty && trainingSigmas.forall(v=>v.isFinite && v>0) &&
    trainingSigmas.sliding(2).forall(v=>v.size<2 || v(0)<=v(1)),"invalid detail daemon training sigmas")
  private val logs=trainingSigmas.map(math.log)
  private def ramp(x: Double): Double =
    val value=if smooth then .5*(1-math.cos(x*math.Pi)) else x
    math.pow(value,exponent)
  private[models] def multipliers(steps: Int): Vector[Double] =
    require(steps>0,"invalid detail daemon step count")
    val s=math.min(start,end);val middle=s+bias*(end-s)
    val si=math.rint(s*(steps-1)).toInt;val mi=math.rint(middle*(steps-1)).toInt;val ei=math.rint(end*(steps-1)).toInt
    Vector.tabulate(steps) { i =>
      val raw=if i<si then startOffset
      else if i>ei then endOffset
      // The second NumPy ramp overwrites the shared midpoint.
      else if i>=mi then
        val x=if ei==mi then 1.0 else 1.0-(i-mi).toDouble/(ei-mi)
        endOffset+ramp(x)*(amount-endOffset)
      else
        val x=if mi==si then 0.0 else (i-si).toDouble/(mi-si)
        startOffset+ramp(x)*(amount-startOffset)
      raw*(1-fade)
    }
  private def interp(values: Vector[Double],x: Double): Double =
    val clamped=math.max(0,math.min(values.size-1.0,x));val lo=clamped.toInt;val hi=math.min(lo+1,values.size-1)
    values(lo)+(values(hi)-values(lo))*(clamped-lo)
  private[models] def adjustment(timestep: Float,multiplier: Double): Option[DetailDaemon.Adjustment] =
    if math.abs(multiplier)<1e-6 then None else
      val sigma=math.exp(interp(logs,timestep.toDouble))
      val adjusted=math.max(trainingSigmas.head,math.min(trainingSigmas.last,sigma*math.max(1-multiplier,.001)))
      val target=math.log(adjusted)
      var low=0;var high=logs.size-1
      while high-low>1 do
        val mid=(low+high)/2
        if logs(mid)<=target then low=mid else high=mid
      val t=if target<=logs.head then 0.0 else if target>=logs.last then logs.size-1.0
        else if logs(high)==logs(low) then low.toDouble else low+(target-logs(low))/(logs(high)-logs(low))
      val scale=math.sqrt(1+sigma*sigma)/math.sqrt(1+adjusted*adjusted)
      Some(DetailDaemon.Adjustment(mode,scale.toFloat,t.toFloat))

object DetailDaemon:
  private[models] final case class Adjustment(mode: String,scale: Float,timestep: Float)
