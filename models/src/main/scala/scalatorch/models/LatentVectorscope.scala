package scalatorch.models

import scalatorch.Tensor

/** NUI Vectorscope CC latent correction, with a separate seeded noise stream. */
final case class LatentVectorscope(brightness: Double=0,contrast: Double=0,saturation: Double=0,
    r: Double=0,g: Double=0,b: Double=0,method: String="Straight",scaling: String="Flat",strength: Double=1):
  require(Seq(brightness,contrast,saturation,r,g,b).forall(v=>v.isFinite && v>= -5 && v<=5) &&
    strength.isFinite && strength>=0 && strength<=5,"invalid vectorscope settings")
  require(Set("Straight","Cross","N.Random","U.Random")(method),"invalid vectorscope method")
  private val curve=Resharpen(1,scaling)
  private[models] def state(seed: Long,sdxl: Boolean,batchSize: Int=1,batchIndex: Int=0): LatentVectorscope.State =
    new LatentVectorscope.State(this,seed,sdxl,curve,batchSize,batchIndex)

object LatentVectorscope:
  private[models] final class State(c: LatentVectorscope,seed: Long,sdxl: Boolean,curve: Resharpen,batchSize: Int,batchIndex: Int):
    require(batchSize>=1 && batchSize<=16 && batchIndex>=0 && batchIndex<batchSize,"invalid vectorscope batch")
    private val rng=new TorchCpuNormal((seed+17)&0xffffffffL)
    private def direction(r: Double,g: Double,b: Double): Array[Float] =
      // Color grading computes half the same regularized RGB-to-latent vector.
      LatentColorGrading(r=r/10,g=g/10,b=b/10).target(sdxl).map(_*20f)
    private val offset=direction(c.brightness+c.r,c.brightness+c.g,c.brightness+c.b)
    private val luma=direction(1,1,1)
    private val norm=math.sqrt(luma.map(v=>v.toDouble*v).sum).toFloat.max(1e-6f)
    for i <- luma.indices do luma(i)/=norm
    def step(input: Tensor[Float],completed: Int,total: Int): Tensor[Float] =
      val a=(c.strength*curve.factor(completed,total).toDouble*10/total).toFloat
      if a==0f then input else
        val dims=input.shape.dims;require(dims.size==4 && dims(1)<=4,"vectorscope requires at most four latent channels")
        val channels=dims(1).toInt;val plane=(dims(2)*dims(3)).toInt;val batch=dims.head.toInt
        val x=input.toFloatArray
        val noise=if c.method.contains("Random") then
          require(x.length.toLong*batchSize<=Int.MaxValue,"vectorscope noise exceeds allocation limit")
          val all=rng.normal(x.length*batchSize)
          Some(java.util.Arrays.copyOfRange(all,x.length*batchIndex,x.length*(batchIndex+1)))
        else None
        for n <- 0 until batch;ch <- 0 until channels do
          val start=(n*channels+ch)*plane
          var cross=0f
          if c.method=="Cross" then
            var i=0;while i<plane do {cross+=math.abs(x(start+i));i+=1};cross/=plane
          var i=0
          while i<plane do
            val multiplier=noise match
              case Some(values) if c.method=="U.Random" => math.abs(values(start+i))
              case Some(values) => 1f+.5f*values(start+i)
              case None if c.method=="Cross" => cross
              case _ => 1f
            x(start+i)+=(a*.15f)*(offset(ch)*multiplier);i+=1
          if c.contrast!=0 then
            var mean=0f;i=0
            while i<plane do {mean+=x(start+i);i+=1}
            mean/=plane;val factor=1f+a*.08f*c.contrast.toFloat;i=0
            while i<plane do {x(start+i)=mean+(x(start+i)-mean)*factor;i+=1}
        if c.saturation!=0 then
          val factor=1f+a*.08f*c.saturation.toFloat
          for n <- 0 until batch;i <- 0 until plane do
            var sum=0f;var ch=0
            while ch<channels do {sum+=x((n*channels+ch)*plane+i)*luma(ch);ch+=1}
            ch=0
            while ch<channels do
              val idx=(n*channels+ch)*plane+i;val along=sum*luma(ch)
              x(idx)=along+(x(idx)-along)*factor;ch+=1
        Tensor.fromFloats(x,input.shape)
