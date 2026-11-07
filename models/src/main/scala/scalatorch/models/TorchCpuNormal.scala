package scalatorch.models

/** PyTorch CPU float32 normal stream. MT19937 notice: META-INF/licenses/keyword-random.txt.
  * Transcendental operations may differ slightly from PyTorch's SIMD math.
  */
private[models] final class TorchCpuNormal(seed: Long):
  private val state=new Array[Int](624)
  state(0)=seed.toInt
  for i <- 1 until 624 do state(i)=1812433253*(state(i-1)^(state(i-1)>>>30))+i
  private var index=624
  private var cached=Option.empty[Double]

  private def nextWord(): Int =
    if index==624 then
      for i <- 0 until 624 do
        val bits=(state(i)&0x80000000)|(state((i+1)%624)&0x7fffffff)
        state(i)=state((i+397)%624)^(bits>>>1)^(if (bits&1)==0 then 0 else 0x9908b0df)
      index=0
    var value=state(index);index+=1
    value^=value>>>11
    value^=(value<<7)&0x9d2c5680
    value^=(value<<15)&0xefc60000
    value^(value>>>18)

  private def uniformFloat(): Float=(nextWord()&0xffffff).toFloat*(1.0f/16777216.0f)
  private def uniformDouble(): Double =
    val value=(nextWord().toLong<<32)|(nextWord().toLong&0xffffffffL)
    (value&0x1fffffffffffffL).toDouble*(1.0/9007199254740992.0)

  private def scalar(): Float = cached match
    case Some(value) => cached=None;value.toFloat
    case None =>
      val u1=uniformDouble();val u2=uniformDouble()
      val radius=math.sqrt(-2.0*math.log1p(-u2));val theta=2.0*math.Pi*u1
      cached=Some(radius*math.sin(theta))
      (radius*math.cos(theta)).toFloat

  def normal(count: Int): Array[Float] =
    require(count>=0,"normal sample count must be nonnegative")
    if count<16 then Array.fill(count)(scalar())
    else
      val output=Array.fill(count)(uniformFloat())
      def transform(offset: Int): Unit =
        for j <- 0 until 8 do
          val u1=1f-output(offset+j);val u2=output(offset+j+8)
          val radius=math.sqrt((-2f*math.log(u1).toFloat).toFloat).toFloat
          val theta=((2.0*math.Pi).toFloat*u2).toFloat
          output(offset+j)=(radius*math.cos(theta).toFloat).toFloat
          output(offset+j+8)=(radius*math.sin(theta).toFloat).toFloat
      var i=0
      while i+16<=count do
        transform(i);i+=16
      if count%16!=0 then
        val start=count-16
        for j <- 0 until 16 do output(start+j)=uniformFloat()
        transform(start)
      output
