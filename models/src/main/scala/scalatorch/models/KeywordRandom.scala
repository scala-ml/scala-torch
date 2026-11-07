package scalatorch.models

/** Integer-seeded CPython Random.choice compatibility, exclusively for keyword selection.
  * MT19937 algorithm: Matsumoto/Nishimura; see META-INF/licenses/keyword-random.txt.
  * This is not the Torch latent-noise generator.
  */
private[models] final class KeywordRandom(seed: Long):
  private val words=new Array[Int](624)
  private var cursor=624
  private val magnitude=BigInt(seed).abs
  private val key=Vector.tabulate(math.max(1,(magnitude.bitLength+31)/32))(i => (magnitude>>(32*i)).toInt)
  words(0)=19650218
  for i <- 1 until 624 do words(i)=1812433253*(words(i-1)^(words(i-1)>>>30))+i
  private var position=1
  for step <- 0 until 624 do
    val j=step%key.size
    words(position)=(words(position)^((words(position-1)^(words(position-1)>>>30))*1664525))+key(j)+j
    position+=1
    if position==624 then
      words(0)=words(623)
      position=1
  for step <- 0 until 623 do
    words(position)=(words(position)^((words(position-1)^(words(position-1)>>>30))*1566083941))-position
    position+=1
    if position==624 then
      words(0)=words(623)
      position=1
  words(0)=0x80000000

  private def nextWord(): Int=
    if cursor==624 then
      for i <- 0 until 624 do
        val joined=(words(i)&0x80000000)|(words((i+1)%624)&0x7fffffff)
        words(i)=words((i+397)%624)^(joined>>>1)^(if (joined&1)==0 then 0 else 0x9908b0df)
      cursor=0
    var value=words(cursor)
    cursor+=1
    value^=value>>>11
    value^=(value<<7)&0x9d2c5680
    value^=(value<<15)&0xefc60000
    value^(value>>>18)

  def below(bound: Int): Int=
    require(bound>0,"random bound must be positive")
    val bits=32-Integer.numberOfLeadingZeros(bound)
    var value=nextWord()>>>(32-bits)
    while value>=bound do value=nextWord()>>>(32-bits)
    value
