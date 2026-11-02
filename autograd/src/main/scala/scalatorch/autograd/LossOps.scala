package scalatorch.autograd

import scalatorch.*

enum LossReduction:
  case None, Sum, Mean

private[autograd] object LossOps:
  def crossEntropy(logits: Variable, targets: Tensor[Long], reduction: LossReduction, ignoreIndex: Long): Variable =
    val shape = logits.data.shape
    require(shape.rank >= 1, "cross entropy requires a class axis")
    val classesLong = shape.dims.last
    require(classesLong > 0 && classesLong <= Int.MaxValue && shape.numel <= Int.MaxValue,
      "cross entropy class/allocation limit exceeded")
    require(targets.shape.dims == shape.dims.dropRight(1), "cross entropy target shape mismatch")
    val classes = classesLong.toInt
    val source = logits.data.toFloatArray
    val ids = targets.toLongArray
    require(source.forall(_.isFinite), "cross entropy requires finite logits")
    require(ids.forall(id => id == ignoreIndex || (id >= 0 && id < classesLong)), "cross entropy target outside class range")
    val maxima = new Array[Double](ids.length)
    val logSums = new Array[Double](ids.length)
    val losses = new Array[Float](ids.length)
    var count = 0
    var total = 0.0
    var row = 0
    while row < ids.length do
      if ids(row) != ignoreIndex then
        val offset = row*classes
        var max = source(offset).toDouble
        var c = 1
        while c < classes do
          max = math.max(max,source(offset+c).toDouble)
          c += 1
        var sum = 0.0
        c = 0
        while c < classes do
          sum += math.exp(source(offset+c).toDouble-max)
          c += 1
        val logSum = math.log(sum)
        val loss = logSum-(source(offset+ids(row).toInt).toDouble-max)
        maxima(row) = max
        logSums(row) = logSum
        losses(row) = loss.toFloat
        total += loss
        count += 1
      row += 1
    val output = reduction match
      case LossReduction.None => Tensor.fromFloats(losses,targets.shape)
      case LossReduction.Sum => Tensor.fromFloats(Array(total.toFloat),Shape.from(Seq.empty))
      case LossReduction.Mean => Tensor.fromFloats(Array(if count == 0 then Float.NaN else (total/count).toFloat),Shape.from(Seq.empty))
    Variable.result(output,Vector(logits -> ((gradient: Tensor[Float]) =>
      val upstream = gradient.toFloatArray
      val result = new Array[Float](source.length)
      var row = 0
      while row < ids.length do
        if ids(row) != ignoreIndex then
          val scale = reduction match
            case LossReduction.None => upstream(row).toDouble
            case LossReduction.Sum => upstream(0).toDouble
            case LossReduction.Mean => upstream(0).toDouble/count
          val offset = row*classes
          var c = 0
          while c < classes do
            val probability = math.exp((source(offset+c).toDouble-maxima(row))-logSums(row))
            result(offset+c) = ((probability-(if c == ids(row) then 1.0 else 0.0))*scale).toFloat
            c += 1
        row += 1
      Tensor.fromFloats(result,shape))))
