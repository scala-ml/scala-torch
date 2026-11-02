package scalatorch.cpu

import scalatorch.*

private[cpu] object CpuCrossEntropy:
  private def prepare(logits: Tensor[Float],targets: Tensor[Long],ignoreIndex: Long) =
    val ids=CrossEntropyLayout.targets(logits.shape,targets,ignoreIndex)
    val source=logits.toFloatArray
    require(source.forall(_.isFinite),"cross entropy requires finite logits")
    val width=logits.shape.dims.last.toInt
    val maxima=new Array[Double](ids.length)
    val logSums=new Array[Double](ids.length)
    ids.indices.foreach { row =>
      if ids(row)>=0 then
        val offset=row*width
        var max=source(offset).toDouble
        var c=1
        while c<width do
          max=math.max(max,source(offset+c).toDouble);c+=1
        var sum=0.0;c=0
        while c<width do
          sum+=math.exp(source(offset+c).toDouble-max);c+=1
        maxima(row)=max;logSums(row)=math.log(sum)
    }
    (ids,source,width,maxima,logSums)
  def forward(logits: Tensor[Float],targets: Tensor[Long],reduction: LossReduction,ignoreIndex: Long): Tensor[Float] =
    val (ids,source,width,maxima,logSums)=prepare(logits,targets,ignoreIndex)
    val losses=ids.indices.map { row =>
      if ids(row)<0 then 0.0 else logSums(row)-(source(row*width+ids(row)).toDouble-maxima(row))
    }
    val values=reduction match
      case LossReduction.None => losses.map(_.toFloat).toArray
      case LossReduction.Sum => Array(losses.sum.toFloat)
      case LossReduction.Mean => Array((losses.sum/ids.count(_>=0)).toFloat)
    Tensor.fromFloats(values,CrossEntropyLayout.output(targets,reduction))
  def backward(logits: Tensor[Float],targets: Tensor[Long],gradient: Tensor[Float],reduction: LossReduction,ignoreIndex: Long): Tensor[Float] =
    require(gradient.shape==CrossEntropyLayout.output(targets,reduction),"cross entropy gradient shape mismatch")
    val (ids,source,width,maxima,logSums)=prepare(logits,targets,ignoreIndex)
    val seed=gradient.toFloatArray
    val count=ids.count(_>=0)
    val result=new Array[Float](source.length)
    ids.indices.foreach { row =>
      if ids(row)>=0 then
        val factor=reduction match
          case LossReduction.None => seed(row).toDouble
          case LossReduction.Sum => seed(0).toDouble
          case LossReduction.Mean => seed(0).toDouble/count
        var c=0
        while c<width do
          val p=math.exp((source(row*width+c).toDouble-maxima(row))-logSums(row))
          result(row*width+c)=((p-(if c==ids(row) then 1.0 else 0.0))*factor).toFloat
          c+=1
    }
    Tensor.fromFloats(result,logits.shape)
