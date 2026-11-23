package scalatorch.models

import scalatorch.*

final case class PostDiffConfig(scale: Double,portion: Double,snap: Boolean=false)

/** Spatial transition from the mixed-resolution denoising scheme of PostDiff (Du et al., ICCV 2025).
  * Input is a low-resolution clean prediction, not the noisy state. The caller supplies independent
  * full-resolution noise and owns both the timestep alignment and per-image random streams.
  */
private[models] object PostDiffTransition:
  /** The prediction at `transitionIndex` is the last low-resolution one. Its clean estimate is
    * enlarged and re-noised at the same sigma, and that noise is the prediction for the update. */
  final case class Plan(lowWidth: Int,lowHeight: Int,transitionIndex: Int):
    def lowPredictionCount: Int=transitionIndex+1

  def plan(width: Int,height: Int,steps: Int,scale: Double,portion: Double,alignment: Int): Plan =
    require(width>0 && height>0 && steps>=2 && alignment>0 && scale.isFinite && scale>0 && scale<1 &&
      portion.isFinite && portion>0 && portion<1,"invalid PostDiff schedule")
    val w=width.toDouble*scale
    val h=height.toDouble*scale
    require(w==math.rint(w) && h==math.rint(h) && w>=alignment && h>=alignment &&
      w<=Int.MaxValue && h<=Int.MaxValue && w.toInt%alignment==0 && h.toInt%alignment==0,
      "PostDiff low dimensions must be integral and aligned")
    val lowSteps=math.floor(steps*portion).toInt
    require(lowSteps>0 && lowSteps<steps,"PostDiff transition must leave steps on both sides")
    Plan(w.toInt,h.toInt,lowSteps-1)

  /** Snaps `width * scale` and `height * scale` to the nearest valid multiple of `alignment` for arbitrary resolutions. */
  def planSnapped(width: Int,height: Int,steps: Int,scale: Double,portion: Double,alignment: Int): Plan =
    require(width>alignment && height>alignment && steps>=2 && alignment>0 && scale.isFinite && scale>0 && scale<1 &&
      portion.isFinite && portion>0 && portion<1,"invalid PostDiff schedule")
    val rawW=width.toDouble*scale
    val rawH=height.toDouble*scale
    require(rawW>=alignment*0.5 && rawH>=alignment*0.5,"PostDiff low dimensions are too small for U-Net alignment")
    def snapDim(raw: Double,full: Int): Int =
      val aligned=(math.rint(raw/alignment).toInt)*alignment
      math.max(alignment,math.min(full-alignment,aligned))
    val w=snapDim(rawW,width)
    val h=snapDim(rawH,height)
    require(w<width && h<height && w%alignment==0 && h%alignment==0,"PostDiff snapped dimensions must shrink below target")
    val lowSteps=math.floor(steps*portion).toInt
    require(lowSteps>0 && lowSteps<steps,"PostDiff transition must leave steps on both sides")
    Plan(w,h,lowSteps-1)

  private def apply(clean: Tensor[Float],noise: Tensor[Float],sigma: Double,vp: Boolean): Tensor[Float] =
    val a=clean.shape.dims
    val b=noise.shape.dims
    require(a.size==4 && b.size==4 && a.forall(_>0) && b.forall(_>0) && a(0)==b(0) && a(1)==b(1),
      "PostDiff requires nonempty NCHW tensors with matching batch and channels")
    require(clean.device==Device.Cpu && noise.device==Device.Cpu && clean.dtype==DType.Float32 && noise.dtype==DType.Float32,
      "PostDiff requires CPU float32 tensors")
    require(b(2)<=Int.MaxValue && b(3)<=Int.MaxValue && b(2)>=a(2) && b(3)>=a(3),"invalid PostDiff target size")
    require(sigma.isFinite && sigma>=0,"invalid PostDiff sigma")
    val x=KohyaHrFix.interpolate2d(clean,b(2).toInt,b(3).toInt,"bilinear").toFloatArray
    val z=noise.toFloatArray
    require(x.forall(_.isFinite) && z.forall(_.isFinite),"PostDiff inputs must be finite")
    val alpha=if vp then 1.0/math.hypot(1.0,sigma) else 1.0
    val values=Array.tabulate(x.length)(i => (alpha*(x(i).toDouble+sigma*z(i).toDouble)).toFloat)
    require(values.forall(_.isFinite),"PostDiff output must be finite")
    Tensor.fromFloats(values,noise.shape)

  def euler(clean: Tensor[Float],noise: Tensor[Float],sigma: Double): Tensor[Float] = apply(clean,noise,sigma,false)
  def vp(clean: Tensor[Float],noise: Tensor[Float],sigma: Double): Tensor[Float] = apply(clean,noise,sigma,true)
