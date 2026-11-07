package scalatorch.models

import scalatorch.*

/** Diffusers rescale_noise_cfg, per batch item over all non-batch axes. */
private[models] object GuidanceRescale:
  def apply(guided: Tensor[Float],positive: Tensor[Float],amount: Float): Tensor[Float] =
    require(guided.shape==positive.shape && guided.shape.rank>=2 && amount.isFinite && amount>=0 && amount<=1,
      "invalid CFG rescale inputs")
    if amount==0 then guided
    else
      val a=guided.toFloatArray;val b=positive.toFloatArray
      val batch=guided.shape.dims.head.toInt;val size=a.length/batch
      require(size>1,"CFG rescale requires more than one element per sample")
      val output=new Array[Float](a.length)
      for n <- 0 until batch do
        val start=n*size
        def deviation(values: Array[Float]): Float =
          var sum=0.0;var i=0
          while i<size do {sum+=values(start+i);i+=1}
          val mean=sum/size;var squares=0.0;i=0
          while i<size do {val d=values(start+i)-mean;squares+=d*d;i+=1}
          math.sqrt(squares/(size-1)).toFloat
        val ratio=deviation(b)/deviation(a)
        var i=0
        while i<size do
          val original=a(start+i)
          output(start+i)=amount*(original*ratio)+(1f-amount)*original
          i+=1
      Tensor.fromFloats(output,guided.shape)
