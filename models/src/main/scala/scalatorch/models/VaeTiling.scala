package scalatorch.models

import scalatorch.{Shape,Tensor}
import java.util.concurrent.CancellationException

/** Diffusers-style overlapping VAE tiles with linear seams blended in output space. */
private[models] object VaeTiling:
  private def check(cancelled: () => Boolean): Unit =
    if cancelled() || Thread.currentThread().isInterrupted then throw new CancellationException("VAE tiling cancelled")

  private def dimensions(input: Tensor[Float]): (Int,Int,Int,Int,Int) =
    val shape=input.shape.dims
    require(input.device==scalatorch.Device.Cpu && input.dtype==scalatorch.DType.Float32 && shape.size==4,
      "VAE tiling requires CPU Float32 NCHW tensors")
    val batch=shape(0).toInt;val channels=shape(1).toInt;val height=shape(2).toInt;val width=shape(3).toInt
    require(batch>0 && channels>0 && height>0 && width>0 &&
      batch.toLong*channels*height*width<=Int.MaxValue,"VAE tile tensor dimensions exceed allocation limits")
    (batch,channels,height,width,shape(1).toInt)

  private def crop(input: Tensor[Float],top: Int,left: Int,height: Int,width: Int): Tensor[Float]=
    val (batch,channels,sourceH,sourceW,_)=dimensions(input)
    require(top>=0 && left>=0 && height>0 && width>0 && top.toLong+height<=sourceH && left.toLong+width<=sourceW,
      "VAE tile crop exceeds tensor bounds")
    val source=input.toFloatArray
    val target=new Array[Float](batch*channels*height*width)
    val sourcePlane=sourceH*sourceW;val targetPlane=height*width
    for b <- 0 until batch; c <- 0 until channels; y <- 0 until height; x <- 0 until width do
      target((b*channels+c)*targetPlane+y*width+x)=source((b*channels+c)*sourcePlane+(top+y)*sourceW+left+x)
    Tensor.fromFloats(target,Shape.from(Seq(batch.toLong,channels.toLong,height.toLong,width.toLong)))

  private def blendVertical(previous: Tensor[Float],current: Tensor[Float],extent: Int): Tensor[Float]=
    val (batch,channels,previousH,width,_)=dimensions(previous)
    val (_,_,currentH,currentW,_)=dimensions(current)
    require(width==currentW,"VAE vertical tile widths disagree")
    val amount=math.min(extent,math.min(previousH,currentH))
    if amount==0 then current
    else
      val before=previous.toFloatArray;val values=current.toFloatArray
      val planePrevious=previousH*width;val planeCurrent=currentH*width
      for b <- 0 until batch; c <- 0 until channels; y <- 0 until amount; x <- 0 until width do
        val weight=y.toFloat/amount
        val dst=(b*channels+c)*planeCurrent+y*width+x
        val src=(b*channels+c)*planePrevious+(previousH-amount+y)*width+x
        values(dst)=before(src)*(1f-weight)+values(dst)*weight
      Tensor.fromFloats(values,current.shape)

  private def blendHorizontal(previous: Tensor[Float],current: Tensor[Float],extent: Int): Tensor[Float]=
    val (batch,channels,height,previousW,_)=dimensions(previous)
    val (_,_,currentH,currentW,_)=dimensions(current)
    require(height==currentH,"VAE horizontal tile heights disagree")
    val amount=math.min(extent,math.min(previousW,currentW))
    if amount==0 then current
    else
      val before=previous.toFloatArray;val values=current.toFloatArray
      val plane=height*currentW;val previousPlane=height*previousW
      for b <- 0 until batch; c <- 0 until channels; y <- 0 until height; x <- 0 until amount do
        val weight=x.toFloat/amount
        val dst=(b*channels+c)*plane+y*currentW+x
        val src=(b*channels+c)*previousPlane+y*previousW+(previousW-amount+x)
        values(dst)=before(src)*(1f-weight)+values(dst)*weight
      Tensor.fromFloats(values,current.shape)

  private def concatenate(left: Tensor[Float],right: Tensor[Float],vertical: Boolean): Tensor[Float]=
    val (batch,channels,leftH,leftW,_)=dimensions(left)
    val (rightBatch,rightChannels,rightH,rightW,_)=dimensions(right)
    require(batch==rightBatch && channels==rightChannels &&
      (if vertical then leftW==rightW else leftH==rightH),"VAE output tile geometry disagrees")
    val height=if vertical then leftH+rightH else leftH
    val width=if vertical then leftW else leftW+rightW
    val output=new Array[Float](batch*channels*height*width)
    val a=left.toFloatArray;val b=right.toFloatArray
    val outPlane=height*width;val aPlane=leftH*leftW;val bPlane=rightH*rightW
    for n <- 0 until batch; c <- 0 until channels do
      val srcA=(n*channels+c)*aPlane;val srcB=(n*channels+c)*bPlane;val dst=(n*channels+c)*outPlane
      if vertical then
        for y <- 0 until leftH do System.arraycopy(a,srcA+y*leftW,output,dst+y*width,width)
        for y <- 0 until rightH do System.arraycopy(b,srcB+y*width,output,dst+(leftH+y)*width,width)
      else
        for y <- 0 until leftH do
          System.arraycopy(a,srcA+y*leftW,output,dst+y*width,leftW)
          System.arraycopy(b,srcB+y*rightW,output,dst+y*width+leftW,rightW)
    Tensor.fromFloats(output,Shape.from(Seq(batch.toLong,channels.toLong,height.toLong,width.toLong)))

  private def stitch(input: Tensor[Float],tileInputH: Int,tileInputW: Int,strideInputH: Int,strideInputW: Int,
      tileOutputH: Int,tileOutputW: Int,overlapOutputH: Int,overlapOutputW: Int,inputAlignment: Int,
      expectedOutputHeight: Int,expectedOutputWidth: Int,
      cancelled: () => Boolean)(runTile: Tensor[Float] => Tensor[Float]): Tensor[Float]=
    val (batch,channels,height,width,_)=dimensions(input)
    require(tileInputH>0 && tileInputW>0 && strideInputH>0 && strideInputW>0 &&
      tileOutputH>0 && tileOutputW>0 && inputAlignment>0 && height%inputAlignment==0 && width%inputAlignment==0 &&
      tileInputH%inputAlignment==0 && tileInputW%inputAlignment==0 && strideInputH%inputAlignment==0 && strideInputW%inputAlignment==0,
      "VAE tile geometry must align to the encoder scale")
    val rawRows=(0 until height by strideInputH).map { top =>
      check(cancelled)
      (0 until width by strideInputW).map { left =>
        check(cancelled)
        val tile=crop(input,top,left,math.min(tileInputH,height-top),math.min(tileInputW,width-left))
        val result=runTile(tile)
        check(cancelled)
        result
      }.toVector
    }.toVector
    val stitchedRows=rawRows.indices.map { rowIndex =>
      val row=rawRows(rowIndex)
      val stitchedTiles=scala.collection.mutable.ArrayBuffer.empty[Tensor[Float]]
      var previousBlended: Option[Tensor[Float]]=None
      row.indices.foreach { columnIndex =>
        var tile=row(columnIndex)
        if rowIndex>0 then tile=blendVertical(rawRows(rowIndex-1)(columnIndex),tile,overlapOutputH)
        previousBlended.foreach(previous=>tile=blendHorizontal(previous,tile,overlapOutputW))
        previousBlended=Some(tile)
        val (_,_,tileH,tileW,_)=dimensions(tile)
        stitchedTiles+=crop(tile,0,0,math.min(tileH,tileOutputH-overlapOutputH),math.min(tileW,tileOutputW-overlapOutputW))
      }
      tileConcat(stitchedTiles.toVector,vertical=false)
    }.toVector
    val output=tileConcat(stitchedRows,vertical=true)
    check(cancelled)
    require(output.shape.dims(0)==batch && output.shape.dims(2)==expectedOutputHeight &&
      output.shape.dims(3)==expectedOutputWidth,"VAE stitched output has unexpected size")
    output

  private def tileConcat(tiles: Vector[Tensor[Float]],vertical: Boolean): Tensor[Float]=
    require(tiles.nonEmpty,"cannot concatenate no VAE tiles")
    tiles.tail.foldLeft(tiles.head)((left,right)=>concatenate(left,right,vertical))

  def encode(input: Tensor[Float],sampleHeight: Int,sampleWidth: Int,factor: Int,
      cancelled: () => Boolean)(runTile: Tensor[Float] => Tensor[Float]): Tensor[Float]=
    val tileH=sampleHeight/factor;val tileW=sampleWidth/factor
    require(sampleHeight%factor==0 && sampleWidth%factor==0 && tileH>0 && tileW>0,
      "VAE sample size must be divisible by the encoder scale")
    val overlapH=tileH/4;val overlapW=tileW/4
    val inputH=dimensions(input)._3;val inputW=dimensions(input)._4
    if inputH<=sampleHeight && inputW<=sampleWidth then runTile(input)
    else stitch(input,sampleHeight,sampleWidth,sampleHeight-overlapH*factor,sampleWidth-overlapW*factor,
      tileH,tileW,overlapH,overlapW,factor,inputH/factor,inputW/factor,cancelled)(runTile)

  def decode(input: Tensor[Float],sampleHeight: Int,sampleWidth: Int,factor: Int,
      cancelled: () => Boolean)(runTile: Tensor[Float] => Tensor[Float]): Tensor[Float]=
    val tileH=sampleHeight/factor;val tileW=sampleWidth/factor
    require(sampleHeight%factor==0 && sampleWidth%factor==0 && tileH>0 && tileW>0,
      "VAE sample size must be divisible by the decoder scale")
    val overlapH=tileH/4;val overlapW=tileW/4
    val (_,_,inputH,inputW,_)=dimensions(input)
    if inputH<=tileH && inputW<=tileW then runTile(input)
    else stitch(input,tileH,tileW,tileH-overlapH,tileW-overlapW,
      sampleHeight,sampleWidth,overlapH*factor,overlapW*factor,1,inputH*factor,inputW*factor,cancelled)(runTile)
