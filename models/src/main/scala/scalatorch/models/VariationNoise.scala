package scalatorch.models

import scalatorch.{Device,DType,Tensor}

/** NUI's variation interpolation is per spatial position across latent channels. */
private[models] object VariationNoise:
  def slerp(low: Tensor[Float],high: Tensor[Float],strength: Double): Tensor[Float] =
    require(strength.isFinite && strength>=0 && strength<=1,"variation strength must be in 0..1")
    require(low.shape==high.shape && low.shape.dims.size==4 && low.device==Device.Cpu && high.device==Device.Cpu &&
      low.dtype==DType.Float32 && high.dtype==DType.Float32,"variation noise shape/device mismatch")
    val dims=low.shape.dims
    val batches=dims(0).toInt;val channels=dims(1).toInt;val plane=(dims(2)*dims(3)).toInt
    val a=low.toFloatArray;val b=high.toFloatArray
    val dots=new Array[Double](batches*plane)
    var total=0.0
    for batch <- 0 until batches; pixel <- 0 until plane do
      var aa=0.0;var bb=0.0;var ab=0.0
      for channel <- 0 until channels do
        val i=(batch*channels+channel)*plane+pixel
        aa+=a(i).toDouble*a(i);bb+=b(i).toDouble*b(i);ab+=a(i).toDouble*b(i)
      val dot=if aa==0 || bb==0 then 1.0 else math.max(-1.0,math.min(1.0,ab/math.sqrt(aa*bb)))
      dots(batch*plane+pixel)=dot;total+=dot
    val linear=total/dots.length>0.9995
    val result=new Array[Float](a.length)
    for batch <- 0 until batches; pixel <- 0 until plane do
      val omega=math.acos(dots(batch*plane+pixel));val sin=math.sin(omega)
      val left=if linear then strength else if math.abs(sin)<1e-12 then 1-strength else math.sin((1-strength)*omega)/sin
      val right=if linear then 1-strength else if math.abs(sin)<1e-12 then strength else math.sin(strength*omega)/sin
      for channel <- 0 until channels do
        val i=(batch*channels+channel)*plane+pixel
        result(i)=(left*a(i)+right*b(i)).toFloat
    Tensor.fromFloats(result,low.shape)
