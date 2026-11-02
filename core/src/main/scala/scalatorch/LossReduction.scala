package scalatorch

enum LossReduction:
  case None, Sum, Mean

private[scalatorch] object CrossEntropyLayout:
  def targets(shape: Shape,targets: Tensor[Long],ignoreIndex: Long): Array[Int] =
    require(shape.rank>=1,"cross entropy requires a class axis")
    val classes=shape.dims.last
    require(classes>0 && classes<=Int.MaxValue && shape.numel<=Int.MaxValue,"cross entropy class/allocation limit exceeded")
    require(targets.shape.dims==shape.dims.dropRight(1),"cross entropy target shape mismatch")
    val ids=targets.toLongArray
    require(ids.forall(id => id==ignoreIndex || (id>=0 && id<classes)),"cross entropy target outside class range")
    ids.map(id => if id==ignoreIndex then -1 else id.toInt)
  def output(targets: Tensor[Long],reduction: LossReduction): Shape =
    if reduction==LossReduction.None then targets.shape else Shape.from(Seq.empty)
