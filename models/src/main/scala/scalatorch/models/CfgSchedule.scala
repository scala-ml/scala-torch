package scalatorch.models

/** NUI CFG Schedule; progress refers to the full scheduler timetable. */
final case class CfgSchedule(startCfg: Double=7,endCfg: Double=3,curve: String="linear",start: Double=0,end: Double=1):
  require(Seq(startCfg,endCfg,start,end).forall(_.isFinite) && startCfg>=0 && endCfg>=0 &&
    start>=0 && start<=1 && end>=0 && end<=1,"invalid CFG schedule")
  require(Set("linear","cosine","square","sqrt")(curve),"invalid CFG schedule curve")
  def at(progress: Double): Float =
    val p=if progress<=start then startCfg else if progress>=end || end<=start then endCfg else
      val x=(progress-start)/(end-start)
      val t=curve match
        case "cosine" => .5*(1-math.cos(x*math.Pi))
        case "square" => x*x
        case "sqrt" => math.sqrt(x)
        case _ => x
      startCfg+(endCfg-startCfg)*t
    p.toFloat
