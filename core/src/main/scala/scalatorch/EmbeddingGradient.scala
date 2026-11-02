package scalatorch

private[scalatorch] object EmbeddingGradient:
  /** Validate logical IDs before narrowing or constructing device grouping metadata. */
  def indices(table: Shape,ids: Tensor[Long],gradient: Shape): Array[Int] =
    require(table.rank==2 && table.dims(1)>0,"embedding requires a matrix with positive width")
    require(table.numel<=Int.MaxValue && gradient.numel<=Int.MaxValue,"embedding gradient exceeds Int indexing")
    require(gradient==Shape.from(ids.shape.dims :+ table.dims(1)),"embedding gradient shape mismatch")
    val values=ids.toLongArray
    require(values.forall(id => id>=0 && id<table.dims(0)),"embedding token ID out of bounds")
    values.map(_.toInt)
