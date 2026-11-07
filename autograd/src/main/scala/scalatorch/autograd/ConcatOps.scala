package scalatorch.autograd

import scalatorch.*

private[autograd] object ConcatOps:
  def apply(inputs: Seq[Variable],dim: Int): Variable =
    val values=inputs.toVector
    val shape=ConcatLayout.outputShape(values.map(_.data.shape),dim)
    val total=shape.dims(dim)
    val out=new Array[Float](shape.numel.toInt)
    if shape.numel>0 then
      val suffix=Shape.from(shape.dims.drop(dim+1)).numel
      val rows=Shape.from(shape.dims.take(dim)).numel
      var offset=0L
      values.foreach { v =>
        val chunk=v.data.shape.dims(dim)*suffix
        if chunk>0 then
          val source=v.data.toFloatArray
          var row=0L
          while row<rows do
            System.arraycopy(source,(row*chunk).toInt,out,(row*total*suffix+offset*suffix).toInt,chunk.toInt)
            row+=1
        offset+=v.data.shape.dims(dim)
      }
    var offset=0L
    val edges=values.map { v =>
      val start=offset
      offset+=v.data.shape.dims(dim)
      val end=offset
      v -> ((g: Tensor[Float]) => g.slice(dim,start,end))
    }
    Variable.result(Tensor.fromFloats(out,shape),edges)
