package scalatorch

/** Private mutable K/V storage. Append at a written prefix (including retry/truncate), never across gaps.
  * The returned attention output is independently owned by the caller.
  */
trait AttentionCache[T] extends AutoCloseable:
  def attend(query: T,key: T,value: T,position: Int,maxScoreElements: Long = 16777216L): T

private[scalatorch] final case class CacheLayout(batch: Int,heads: Int,capacity: Int,keyWidth: Int,valueWidth: Int,maxBytes: Long):
  require(batch>0 && heads>0 && capacity>0 && keyWidth>0 && valueWidth>0 && maxBytes>=0,"invalid attention cache dimensions/budget")
  private val keyCount=BigInt(batch)*heads*capacity*keyWidth
  private val valueCount=BigInt(batch)*heads*capacity*valueWidth
  require(keyCount<=Int.MaxValue && valueCount<=Int.MaxValue && (keyCount+valueCount)*4<=maxBytes,"attention cache allocation budget exceeded")
  val bytes: Long=((keyCount+valueCount)*4).toLong
  def keyShape(tokens: Long): Shape = Shape.from(Seq(batch.toLong,heads.toLong,tokens,keyWidth.toLong))
  def valueShape(tokens: Long): Shape = Shape.from(Seq(batch.toLong,heads.toLong,tokens,valueWidth.toLong))
  def validate(q: Shape,k: Shape,v: Shape,position: Int,written: Int,maxScores: Long): (Shape,Long,Int) =
    require(k.rank==4,"cache key must have rank 4")
    val chunk=k.dims(2)
    require(chunk>0 && k==keyShape(chunk) && v==valueShape(chunk),"cache append shape mismatch")
    require(q.rank==4 && q.dims(2)==chunk,"cache query/chunk mismatch")
    val end=position.toLong+chunk
    require(position>=0 && position<=written && end<=capacity,"cache append gap or capacity exceeded")
    val (output,scores)=AttentionLayout.validate(q,keyShape(end),valueShape(end),position,maxScores)
    (output,scores,end.toInt)
