package scalatorch.models

import scalatorch.Tensor

/** NUI latent mirror/rotation callback; operates on NCHW floating tensors. */
final case class LatentMirroring(mode: String="Alternate Steps",style: String="Vertical Mirroring",maxStep: Double=.25,strength: Double=1):
  require(Set("Alternate Steps","Blend Average")(mode),"invalid mirroring mode")
  require(Set("Vertical Mirroring","Horizontal Mirroring","Horizontal+Vertical Mirroring","90 Degree Rotation","180 Degree Rotation")(style),"invalid mirroring style")
  require(maxStep.isFinite && maxStep>=0 && maxStep<=1 && strength.isFinite && strength>=0 && strength<=1,"invalid mirroring strength or step")
  private[models] def step(latent: Tensor[Float],completed: Int,total: Int): Tensor[Float] =
    if completed>maxStep*total || strength==0 || (mode=="Alternate Steps" && completed%2==0) then latent
    else
      val dims=latent.shape.dims;require(dims.size==4,"latent mirror requires NCHW")
      val h=dims(2).toInt;val w=dims(3).toInt;val plane=h*w
      val src=latent.toFloatArray;val out=new Array[Float](src.length)
      val s=style.toLowerCase(java.util.Locale.ROOT)
      for i <- src.indices do
        val channel=i/plane;val y=(i%plane)/w;val x=i%w
        val (sy,sx)=if s.startsWith("vertical") then (y,w-1-x)
          else if s.startsWith("horizontal") then (h-1-y,x)
          else if s.contains("90") && h==w then (x,w-1-y)
          else if s.contains("180") || s.contains("90") || s.contains("both") then (h-1-y,w-1-x)
          else (y,x)
        val mirror=src(channel*plane+sy*w+sx)
        val target=if mode=="Alternate Steps" then mirror else (src(i)+mirror)/2f
        out(i)=src(i)+(target-src(i))*strength.toFloat
      Tensor.fromFloats(out,latent.shape)
