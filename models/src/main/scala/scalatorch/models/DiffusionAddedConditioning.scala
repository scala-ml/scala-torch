package scalatorch.models

import scalatorch.*

/** SDXL pooled text and per-image original size/crop/target size constants. */
final case class DiffusionAddedConditioning(pooledText: Tensor[Float],timeIds: Tensor[Float]):
  private[models] def features(batch: Long,timeWidth: Int,inputWidth: Long,maxElements: Long): Tensor[Float] =
    require(pooledText.device==Device.Cpu && timeIds.device==Device.Cpu &&
      pooledText.dtype==DType.Float32 && timeIds.dtype==DType.Float32,"added conditioning must be CPU Float32")
    val text=pooledText.shape;val ids=timeIds.shape
    require(text.rank==2 && ids.rank==2 && text.dims.head==batch && ids.dims.head==batch &&
      text.dims(1)>0 && ids.dims(1)>0,"added conditioning batch/rank mismatch")
    require(BigInt(text.dims(1))+BigInt(ids.dims(1))*timeWidth==inputWidth,"added conditioning width mismatch")
    val shape=Shape.from(Seq(batch,inputWidth))
    require(shape.numel<=maxElements && shape.numel<=Int.MaxValue,"added conditioning allocation limit")
    val pooled=pooledText.toFloatArray
    require(pooled.forall(_.isFinite),"nonfinite pooled text")
    val expanded=DiffusionTimesteps.sinusoidal(timeIds.reshape(Shape.from(Seq(ids.numel))),timeWidth,
      flipSinToCos=true,frequencyShift=0,maxElements=maxElements).toFloatArray
    val output=new Array[Float](shape.numel.toInt)
    val textWidth=text.dims(1).toInt;val expandedWidth=(ids.dims(1)*timeWidth).toInt
    for row <- 0 until batch.toInt do
      System.arraycopy(pooled,row*textWidth,output,row*inputWidth.toInt,textWidth)
      System.arraycopy(expanded,row*expandedWidth,output,row*inputWidth.toInt+textWidth,expandedWidth)
    Tensor.fromFloats(output,shape)
