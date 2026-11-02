package scalatorch

private[scalatorch] final class Storage[A](val data: Array[A], val dtype: DType[A]):
  def length: Int = data.length
  def allocate(size: Int): Array[A] = (dtype match
    case DType.Float32 => new Array[Float](size)
    case DType.Float64 => new Array[Double](size)
    case DType.Int64 => new Array[Long](size)
    case DType.Bool => new Array[Boolean](size)
  ).asInstanceOf[Array[A]]
