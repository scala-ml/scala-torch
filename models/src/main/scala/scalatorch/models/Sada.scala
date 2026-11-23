package scalatorch.models

/** Step skipping by the stability criterion of SADA (Jiang et al., "SADA: Stability-guided Adaptive Diffusion Acceleration",
  * ICML 2025, arXiv 2507.17135), re-implemented from the paper for the Euler sampler in sigma space.
  *
  * With y the change of the latent per step (y_i = epsilon_i * (sigma_{i+1} - sigma_i)), the paper compares the solver's next
  * latent with a third-order multistep estimate, x^ = x_i + 5/6 y_i + 5/6 y_{i-1} - 2/3 y_{i-2}, and calls the trajectory stable
  * when the difference is anti-aligned with the curvature of y: (x_{i+1} - x^) . (y_i - 2 y_{i-1} + y_{i-2}) < 0. For a stable
  * step the next U-Net call is skipped: the noise prediction is reused and the clean image is taken from x^, which is the
  * same as using epsilon' = epsilon_i + (x_{i+1} - x^) / sigma_{i+1}.
  *
  * Only this step-wise part is implemented; the token pruning and the Lagrange interpolation phase of the paper are not.
  * `start` and `end` are fractions of the steps between which a step may be skipped (the paper's range for 50 steps is 10 to 47),
  * `maxConsecutive` the longest run of skipped steps.
  */
final case class SadaConfig(start: Double=0.2,end: Double=0.9,maxConsecutive: Int=2):
  require(start>=0 && start<=1 && end>=0 && end<=1 && maxConsecutive>=1 && maxConsecutive<=8,"invalid SADA settings")

final class SadaState(config: SadaConfig,steps: Int):
  private var previous: Array[Float]=null       // y_{i-1}
  private var beforePrevious: Array[Float]=null // y_{i-2}
  private var pending: Array[Float]=null        // the prediction for the next step when its U-Net call is skipped
  private var consecutive=0
  private var skippedSteps=0
  def skipped: Int=skippedSteps

  /** The prediction to use for the step that starts now in place of a U-Net call, if the last step was found stable. */
  def planned: Option[Array[Float]]=Option(pending)

  /** After the prediction of step `index` is known (computed or substituted), before the latent is advanced. */
  def observe(index: Int,prediction: Array[Float],sigma: Double,next: Double,substituted: Boolean): Unit =
    pending=null
    if substituted then { consecutive+=1;skippedSteps+=1 } else consecutive=0
    val h=(next-sigma).toFloat
    val change=new Array[Float](prediction.length)
    var j=0
    while j<change.length do { change(j)=prediction(j)*h;j+=1 }
    val target=index+1
    if previous!=null && beforePrevious!=null && previous.length==change.length && beforePrevious.length==change.length &&
        next>0 && target>=math.ceil(config.start*steps) && target<math.floor(config.end*steps) && target<steps-1 &&
        consecutive<config.maxConsecutive then
      var aligned=0.0
      j=0
      while j<change.length do
        val error=change(j)/6f-previous(j)*(5f/6f)+beforePrevious(j)*(2f/3f) // x_{i+1} - x^
        val curvature=change(j)-2f*previous(j)+beforePrevious(j)
        aligned+=error.toDouble*curvature
        j+=1
      if aligned<0 then
        val reused=new Array[Float](change.length)
        val inverse=(1.0/next).toFloat
        j=0
        while j<change.length do
          val error=change(j)/6f-previous(j)*(5f/6f)+beforePrevious(j)*(2f/3f)
          reused(j)=prediction(j)+error*inverse
          j+=1
        pending=reused
    beforePrevious=previous
    previous=change
