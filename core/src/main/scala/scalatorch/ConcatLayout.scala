package scalatorch

/** Shared geometry for packed CPU and CUDA concatenation. */
private[scalatorch] object ConcatLayout:
  def outputShape(inputs: Seq[Shape],dim: Int): Shape =
    require(inputs.nonEmpty,"concatenation requires at least one input")
    val first=inputs.head
    require(dim>=0 && dim<first.rank,"concatenation axis out of range")
    inputs.foreach { shape =>
      require(shape.rank==first.rank && first.dims.indices.forall(i =>
        i==dim || shape.dims(i)==first.dims(i)),"concatenation dimensions mismatch")
    }
    val total=try inputs.foldLeft(0L)((n,s) => Math.addExact(n,s.dims(dim)))
      catch case _: ArithmeticException => throw IllegalArgumentException("concatenation axis overflows Long")
    val shape=Shape.from(first.dims.updated(dim,total))
    require(shape.numel<=Int.MaxValue,"concatenation output exceeds Int indexing")
    shape
