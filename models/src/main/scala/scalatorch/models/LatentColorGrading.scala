package scalatorch.models

import scalatorch.Tensor

/** NUI's latent color recentering and final-step range normalization. */
final case class LatentColorGrading(recenter: Boolean=true,recenterStrength: Double=1,r: Double=0,g: Double=0,b: Double=0,
    normalize: Boolean=false,normalizeStrength: Double=1):
  require(Seq(recenterStrength,r,g,b,normalizeStrength).forall(_.isFinite) && recenterStrength>=0 && recenterStrength<=2 &&
    Seq(r,g,b).forall(v=>v>= -1 && v<=1) && normalizeStrength>=0 && normalizeStrength<=1,"invalid color grading options")
  private[models] def target(sdxl: Boolean): Array[Float] =
    val f=if sdxl then Array(Array(.3448,.4168,.4395),Array(-.1953,-.0290,.0250),Array(.1074,.0886,-.0163),Array(-.3730,-.2499,-.2088))
      else Array(Array(.298,.207,.208),Array(.187,.286,.173),Array(-.158,.189,.264),Array(-.184,-.271,-.473))
    val a=Array.ofDim[Double](3,4);val rgb=Array(r,g,b)
    for i <- 0 until 3 do
      for j <- 0 until 3 do a(i)(j)=f.indices.map(k=>f(k)(i)*f(k)(j)).sum+(if i==j then .05 else 0)
      a(i)(3)=rgb(i)
    for i <- 0 until 3 do
      val pivot=(i until 3).maxBy(row=>math.abs(a(row)(i)))
      val swap=a(i);a(i)=a(pivot);a(pivot)=swap
      val scale=a(i)(i)
      for j <- i until 4 do a(i)(j)/=scale
      for row <- 0 until 3 if row!=i do
        val k=a(row)(i)
        for j <- i until 4 do a(row)(j)-=k*a(i)(j)
    f.map(row=>(.5*(0 until 3).map(i=>row(i)*a(i)(3)).sum).toFloat)
  private[models] def step(input: Tensor[Float],completed: Int,total: Int,sdxl: Boolean): Tensor[Float] =
    if !recenter && (!normalize || completed!=total) then input
    else
      val dims=input.shape.dims;require(dims.size==4 && dims(1)<=4,"color grading requires at most four latent channels")
      val channels=dims(1).toInt;val plane=(dims(2)*dims(3)).toInt;val values=input.toFloatArray;val wanted=target(sdxl)
      for batch <- 0 until dims.head.toInt;channel <- 0 until channels do
        val offset=(batch*channels+channel)*plane
        if recenter && channel!=0 then
          var sum=0f;var i=0
          while i<plane do {sum+=values(offset+i);i+=1}
          val mean=sum/plane;val shift=((mean-wanted(channel))*recenterStrength.toFloat*(1.0/total).toFloat*2f)
          i=0;while i<plane do {values(offset+i)-=shift;i+=1}
        if normalize && completed==total then
          var lo=Float.PositiveInfinity;var hi=Float.NegativeInfinity;var i=0
          while i<plane do {lo=math.min(lo,values(offset+i));hi=math.max(hi,values(offset+i));i+=1}
          val mid=(hi+lo)/2f;val half=math.max((hi-lo)/2f,1e-6f)
          val target=if sdxl then 3.5f else 4f
          val k=normalizeStrength.toFloat;val factor=(1f-k)+k*math.min(target/half,1.5f)
          i=0;while i<plane do {values(offset+i)=mid+(values(offset+i)-mid)*factor;i+=1}
      Tensor.fromFloats(values,input.shape)
