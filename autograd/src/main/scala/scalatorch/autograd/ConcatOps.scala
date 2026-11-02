package scalatorch.autograd

import scalatorch.*

private[autograd] object ConcatOps:
  def apply(inputs: Seq[Variable],dim: Int): Variable =
    val values=inputs.toVector
    require(values.nonEmpty,"concatenation requires at least one input")
    val first=values.head.data.shape
    require(dim>=0 && dim<first.rank,"concatenation axis out of range")
    values.foreach { v =>
      require(v.data.shape.rank==first.rank && first.dims.indices.forall(i =>
        i==dim || v.data.shape.dims(i)==first.dims(i)),"concatenation dimensions mismatch")
    }
    val total=try values.foldLeft(0L)((n,v) => Math.addExact(n,v.data.shape.dims(dim)))
      catch case _: ArithmeticException => throw IllegalArgumentException("concatenation axis overflows Long")
    val shape=Shape.from(first.dims.updated(dim,total))
    require(shape.numel<=Int.MaxValue,"concatenation output exceeds Int indexing")
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
