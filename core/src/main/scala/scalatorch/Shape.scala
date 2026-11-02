package scalatorch

final class Shape private (val dims: Vector[Long], val numel: Long):
  def rank: Int = dims.size
  override def equals(other: Any): Boolean = other match
    case s: Shape => dims == s.dims
    case _ => false
  override def hashCode(): Int = dims.hashCode
  override def toString: String = dims.mkString("Shape(", ", ", ")")

object Shape:
  def from(dims: Seq[Long]): Shape =
    val ds = dims.toVector
    require(ds.forall(_ >= 0), "negative dimension")
    val n = if ds.contains(0L) then 0L else
      try ds.foldLeft(1L)(Math.multiplyExact)
      catch case _: ArithmeticException => throw IllegalArgumentException("shape size overflows Long")
    new Shape(ds, n)
