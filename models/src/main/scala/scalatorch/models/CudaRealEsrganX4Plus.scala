package scalatorch.models

import scalatorch.*
import scalatorch.ffm.{CudaSession,CudaTensor}
import java.awt.image.BufferedImage
import java.nio.file.Path
import scala.util.Using

/** Tiled CUDA inference for the fixed Real-ESRGAN x4+ RRDBNet checkpoint. */
object CudaRealEsrganX4Plus:
  private val TileSize=128
  private val TilePad=32

  private final class Runtime(session: CudaSession,weights: Map[String,CudaTensor],cancelled: () => Boolean,
      fastConvolution: Boolean,blockCount: Int):
    private def check(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("Real-ESRGAN inference cancelled")
      session.allocatedBytes

    private def conv(input: CudaTensor,name: String): CudaTensor =
      check()
      Using.Manager { use =>
        val config=Conv2dConfig(padding=(1,1))
        val output=use(if fastConvolution then session.conv2dFast(input,weights(s"$name.weight"),config)
          else session.conv2d(input,weights(s"$name.weight"),config))
        val bias=use(session.reshape(weights(s"$name.bias"),Shape.from(Seq(1L,weights(s"$name.bias").shape.numel,1L,1L))))
        val result=use(session.addBroadcast(output,bias))
        session.reshape(result,result.shape)
      }.get

    private def activation(input: CudaTensor): CudaTensor = session.leakyRelu(input,.2f)

    private def convActivation(input: CudaTensor,name: String): CudaTensor =
      val output=conv(input,name)
      try activation(output) finally output.close()

    private def denseBlock(input: CudaTensor,root: String): CudaTensor =
      Using.Manager { use =>
        val first=use(convActivation(input,s"$root.conv1"))
        val secondIn=use(session.cat(Seq(input,first),1))
        val second=use(convActivation(secondIn,s"$root.conv2"))
        val thirdIn=use(session.cat(Seq(input,first,second),1))
        val third=use(convActivation(thirdIn,s"$root.conv3"))
        val fourthIn=use(session.cat(Seq(input,first,second,third),1))
        val fourth=use(convActivation(fourthIn,s"$root.conv4"))
        val fifthIn=use(session.cat(Seq(input,first,second,third,fourth),1))
        val fifth=use(conv(fifthIn,s"$root.conv5"))
        val scaled=use(session.scale(fifth,.2f))
        val result=use(session.add(input,scaled))
        session.reshape(result,result.shape)
      }.get

    private def rrdb(input: CudaTensor,index: Int): CudaTensor =
      var hidden=input
      var owned=false
      try
        var dense=1
        while dense<=3 do
          check()
          val next=denseBlock(hidden,s"RRDB_trunk.$index.RDB$dense")
          if owned then hidden.close()
          hidden=next;owned=true;dense+=1
        val scaled=session.scale(hidden,.2f)
        try session.add(input,scaled)
        finally scaled.close()
      finally if owned then hidden.close()

    private def upsample2(input: CudaTensor): CudaTensor =
      val shape=input.shape
      session.resizeNearest2d(input,Math.multiplyExact(shape.dims(2).toInt,2),Math.multiplyExact(shape.dims(3).toInt,2))

    def apply(input: CudaTensor): CudaTensor =
      val initial=conv(input,"conv_first")
      var trunk=initial
      var trunkOwned=false
      try
        var block=0
        while block<blockCount do
          check()
          val next=rrdb(trunk,block)
          if trunkOwned then trunk.close()
          trunk=next;trunkOwned=true;block+=1
        val trunkConv=conv(trunk,"trunk_conv")
        val features=try session.add(initial,trunkConv) finally trunkConv.close()
        try
          val up1in=upsample2(features)
          val up1=try convActivation(up1in,"upconv1") finally up1in.close()
          try
            val up2in=upsample2(up1)
            val up2=try convActivation(up2in,"upconv2") finally up2in.close()
            try
              val hr=convActivation(up2,"HRconv")
              try conv(hr,"conv_last") finally hr.close()
            finally up2.close()
          finally up1.close()
        finally features.close()
      finally
        if trunkOwned then trunk.close()
        initial.close()

  private def load(session: CudaSession,source: RealEsrganX4PlusWeights,cancelled: () => Boolean): Map[String,CudaTensor]=
    val result=scala.collection.mutable.LinkedHashMap.empty[String,CudaTensor]
    try
      source.tensors.foreach { (name,value) =>
        if cancelled() then throw new java.util.concurrent.CancellationException("Real-ESRGAN model loading cancelled")
        result(name)=session.upload(Tensor.fromFloats(value.values,Shape.from(value.shape)))
      }
      result.toMap
    catch
      case error: Throwable =>
        result.values.toVector.reverse.foreach(t => try t.close() catch case cleanup: Throwable => error.addSuppressed(cleanup))
        throw error

  private def tile(source: BufferedImage,x0: Int,y0: Int,x1: Int,y1: Int): Tensor[Float]=
    val width=x1-x0;val height=y1-y0;val plane=width*height
    val values=new Array[Float](plane*3)
    var y=0
    while y<height do
      var x=0
      while x<width do
        val rgb=source.getRGB(x0+x,y0+y);val index=y*width+x
        values(index)=((rgb>>>16)&255)/255f
        values(plane+index)=((rgb>>>8)&255)/255f
        values(2*plane+index)=(rgb&255)/255f
        x+=1
      y+=1
    Tensor.fromFloats(values,Shape.from(Seq(1L,3L,height.toLong,width.toLong)))

  private final class CpuRuntime(weights: Map[String,Tensor[Float]],cancelled: () => Boolean,blockCount: Int):
    private def check(): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("Real-ESRGAN inference cancelled")

    private def addScaled(a: Tensor[Float],b: Tensor[Float],scale: Float=1f): Tensor[Float] =
      val aArr=a.toFloatArray;val bArr=b.toFloatArray
      Tensor.fromFloats(Array.tabulate(aArr.length)(i => aArr(i)+bArr(i)*scale),a.shape)

    private def conv(input: Tensor[Float],name: String): Tensor[Float] = scalatorch.autograd.Autograd.noGrad {
      check()
      val config=Conv2dConfig(padding=(1,1))
      val w=scalatorch.autograd.Variable(weights(s"$name.weight"),requiresGrad=false)
      val b=scalatorch.autograd.Variable(weights(s"$name.bias").reshape(Shape.from(Seq(1L,weights(s"$name.bias").shape.numel,1L,1L))),requiresGrad=false)
      (scalatorch.autograd.Variable(input,requiresGrad=false).conv2d(w,config)+b).data
    }

    private def leakyRelu(input: Tensor[Float]): Tensor[Float] =
      val arr=input.toFloatArray.clone()
      var i=0
      while i<arr.length do
        if arr(i)<0f then arr(i)=arr(i)*0.2f
        i+=1
      Tensor.fromFloats(arr,input.shape)

    private def convActivation(input: Tensor[Float],name: String): Tensor[Float] =
      leakyRelu(conv(input,name))

    private def cat1(tensors: Seq[Tensor[Float]]): Tensor[Float] =
      val h=tensors.head.shape.dims(2).toInt
      val w=tensors.head.shape.dims(3).toInt
      val plane=h*w
      val totalC=tensors.map(_.shape.dims(1).toInt).sum
      val out=new Array[Float](totalC*plane)
      var offset=0
      tensors.foreach { t =>
        val arr=t.toFloatArray
        System.arraycopy(arr,0,out,offset,arr.length)
        offset+=arr.length
      }
      Tensor.fromFloats(out,Shape.from(Seq(1L,totalC.toLong,h.toLong,w.toLong)))

    private def denseBlock(input: Tensor[Float],root: String): Tensor[Float] =
      val first=convActivation(input,s"$root.conv1")
      val second=convActivation(cat1(Seq(input,first)),s"$root.conv2")
      val third=convActivation(cat1(Seq(input,first,second)),s"$root.conv3")
      val fourth=convActivation(cat1(Seq(input,first,second,third)),s"$root.conv4")
      val fifth=conv(cat1(Seq(input,first,second,third,fourth)),s"$root.conv5")
      addScaled(input,fifth,0.2f)

    private def rrdb(input: Tensor[Float],index: Int): Tensor[Float] =
      var hidden=input
      var dense=1
      while dense<=3 do
        check()
        hidden=denseBlock(hidden,s"RRDB_trunk.$index.RDB$dense")
        dense+=1
      addScaled(input,hidden,0.2f)

    private def upsample2(input: Tensor[Float]): Tensor[Float] =
      val dims=input.shape.dims
      val c=dims(1).toInt;val h=dims(2).toInt;val w=dims(3).toInt
      val outH=h*2;val outW=w*2
      val src=input.toFloatArray
      val out=new Array[Float](c*outH*outW)
      for ch <- 0 until c; y <- 0 until outH; x <- 0 until outW do
        out((ch*outH+y)*outW+x)=src((ch*h+(y/2))*w+(x/2))
      Tensor.fromFloats(out,Shape.from(Seq(1L,c.toLong,outH.toLong,outW.toLong)))

    def apply(input: Tensor[Float]): Tensor[Float] =
      val initial=conv(input,"conv_first")
      var trunk=initial
      var block=0
      while block<blockCount do
        check()
        trunk=rrdb(trunk,block)
        block+=1
      val features=addScaled(initial,conv(trunk,"trunk_conv"),1f)
      val up1=convActivation(upsample2(features),"upconv1")
      val up2=convActivation(upsample2(up1),"upconv2")
      val hr=convActivation(up2,"HRconv")
      conv(hr,"conv_last")

  /** Runs a bounded-memory, overlapped 4x pass on CPU and returns an RGB image. */
  def upscaleCpu(source: BufferedImage,modelPath: Path,cancelled: () => Boolean=() => false): BufferedImage =
    val width=source.getWidth;val height=source.getHeight
    require(width>0 && height>0 && width.toLong*height<=16777216L,"Real-ESRGAN source image exceeds pixel limit")
    val weights=RealEsrganX4PlusWeights.read(modelPath)
    val outputWidth=Math.multiplyExact(width,4);val outputHeight=Math.multiplyExact(height,4)
    require(outputWidth.toLong*outputHeight<=67108864L,"Real-ESRGAN output exceeds pixel limit")
    val output=new BufferedImage(outputWidth,outputHeight,BufferedImage.TYPE_INT_RGB)
    val cpuWeights=weights.tensors.map { (name,value) =>
      if cancelled() then throw new java.util.concurrent.CancellationException("Real-ESRGAN model loading cancelled")
      name -> Tensor.fromFloats(value.values,Shape.from(value.shape))
    }
    val runtime=new CpuRuntime(cpuWeights,cancelled,weights.blockCount)
    var y=0
    while y<height do
      var x=0
      val coreY1=math.min(height,y+TileSize)
      val inputY0=math.max(0,y-TilePad);val inputY1=math.min(height,coreY1+TilePad)
      while x<width do
        if cancelled() || Thread.currentThread().isInterrupted then
          throw new java.util.concurrent.CancellationException("Real-ESRGAN tiling cancelled")
        val coreX1=math.min(width,x+TileSize)
        val inputX0=math.max(0,x-TilePad);val inputX1=math.min(width,coreX1+TilePad)
        val cpuTile=tile(source,inputX0,inputY0,inputX1,inputY1)
        val cpu=runtime(cpuTile)
        val values=cpu.toFloatArray
        val tileWidth=inputX1-inputX0;val tileHeight=inputY1-inputY0
        val plane=tileWidth*tileHeight*16
        val cropX=(x-inputX0)*4;val cropY=(y-inputY0)*4
        for oy <- 0 until (coreY1-y)*4 do
          for ox <- 0 until (coreX1-x)*4 do
            val sourceOffset=(cropY+oy)*(tileWidth*4)+cropX+ox
            val red=math.max(0f,math.min(1f,values(sourceOffset)))
            val green=math.max(0f,math.min(1f,values(plane+sourceOffset)))
            val blue=math.max(0f,math.min(1f,values(2*plane+sourceOffset)))
            val packed=((red*255f+.5f).toInt<<16)|((green*255f+.5f).toInt<<8)|(blue*255f+.5f).toInt
            output.setRGB(x*4+ox,y*4+oy,packed)
        x=coreX1
      y=coreY1
    output

  /** Runs a bounded-memory, overlapped 4x pass and returns an RGB image. */
  def upscale(source: BufferedImage,modelPath: Path,cancelled: () => Boolean=() => false,
      deviceIndex: Int=0,maxCudaBytes: Long=1073741824L,fastConvolution: Boolean=true): BufferedImage =
    val width=source.getWidth;val height=source.getHeight
    require(width>0 && height>0 && width.toLong*height<=16777216L,"Real-ESRGAN source image exceeds pixel limit")
    val weights=RealEsrganX4PlusWeights.read(modelPath)
    val outputWidth=Math.multiplyExact(width,4);val outputHeight=Math.multiplyExact(height,4)
    require(outputWidth.toLong*outputHeight<=67108864L,"Real-ESRGAN output exceeds pixel limit")
    val output=new BufferedImage(outputWidth,outputHeight,BufferedImage.TYPE_INT_RGB)
    CudaSession.withDevice(deviceIndex,maxCudaBytes) { session =>
      val uploaded=load(session,weights,cancelled)
      val runtime=new Runtime(session,uploaded,cancelled,fastConvolution,weights.blockCount)
      try
        var y=0
        while y<height do
          var x=0
          val coreY1=math.min(height,y+TileSize)
          val inputY0=math.max(0,y-TilePad);val inputY1=math.min(height,coreY1+TilePad)
          while x<width do
            if cancelled() || Thread.currentThread().isInterrupted then
              throw new java.util.concurrent.CancellationException("Real-ESRGAN tiling cancelled")
            val coreX1=math.min(width,x+TileSize)
            val inputX0=math.max(0,x-TilePad);val inputX1=math.min(width,coreX1+TilePad)
            val cpuTile=tile(source,inputX0,inputY0,inputX1,inputY1)
            val input=session.upload(cpuTile)
            val predicted=try runtime(input) finally input.close()
            val cpu=try predicted.toCpu finally predicted.close()
            val values=cpu.toFloatArray
            val tileWidth=inputX1-inputX0;val tileHeight=inputY1-inputY0
            val plane=tileWidth*tileHeight*16
            val cropX=(x-inputX0)*4;val cropY=(y-inputY0)*4
            for oy <- 0 until (coreY1-y)*4 do
              for ox <- 0 until (coreX1-x)*4 do
                val sourceOffset=(cropY+oy)*(tileWidth*4)+cropX+ox
                val red=math.max(0f,math.min(1f,values(sourceOffset)))
                val green=math.max(0f,math.min(1f,values(plane+sourceOffset)))
                val blue=math.max(0f,math.min(1f,values(2*plane+sourceOffset)))
                val packed=((red*255f+.5f).toInt<<16)|((green*255f+.5f).toInt<<8)|(blue*255f+.5f).toInt
                output.setRGB(x*4+ox,y*4+oy,packed)
            x=coreX1
          y=coreY1
      finally uploaded.values.toVector.reverse.foreach(_.close())
    }
    output
