package scalatorch.models

import scalatorch.Tensor

/** Request-local NUI ReSharpen callback state. */
final case class Resharpen(sharpness: Double=.5,scaling: String="Flat"):
  require(sharpness.isFinite && sharpness>= -4 && sharpness<=4,"resharpen sharpness must be in -4..4")
  require(Set("flat","cos","sin","1-cos","1-sin")(scaling.toLowerCase(java.util.Locale.ROOT).replace(" ","")),"invalid resharpen scaling")
  private[models] def factor(completed: Int,total: Int): Float =
    val p=completed.toDouble/total
    val c=scaling.toLowerCase(java.util.Locale.ROOT).replace(" ","") match
      case "cos" => math.cos(p*math.Pi/2)
      case "sin" => math.sin(p*math.Pi/2)
      case "1-cos" => 1-math.cos(p*math.Pi/2)
      case "1-sin" => 1-math.sin(p*math.Pi/2)
      case _ => 1.0
    (sharpness*.1*c).toFloat
  private[models] def step(current: Tensor[Float],previous: Option[Tensor[Float]],completed: Int,total: Int): Tensor[Float] =
    if previous.isEmpty || completed>=total then current
    else
      val a=current.toFloatArray;val b=previous.get.toFloatArray
      require(a.length==b.length,"resharpen latent shape changed")
      val k=factor(completed,total)
      var i=0
      while i<a.length do {a(i)=a(i)+(a(i)-b(i))*k;i+=1}
      Tensor.fromFloats(a,current.shape)
