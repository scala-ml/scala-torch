package scalatorch.models

import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.ffm.{CudaSession,CudaTensor}
import scalatorch.io.SafeTensorSet
import scalatorch.nn.{Conv2d,Module,Parameter}

/** Frozen TAESD encoder using the standard Diffusers layer layout. */
final case class TinyAutoencoderEncoder private (
    input: Conv2d,blocks: Vector[Vector[TinyAutoencoderBlock]],
    downsample: Vector[Conv2d],output: Conv2d) derives Module:
  private def prepare(image: Tensor[Float],cancelled: () => Unit): Tensor[Float] =
    val dims=image.shape.dims
    require(image.device==Device.Cpu && image.dtype==DType.Float32 && dims.size==4 &&
      dims(0)==1 && dims(1)==3 && dims(2)>0 && dims(3)>0,"TAESD expects one CPU Float32 RGB image")
    require(BigInt(dims(2))*dims(3)*64<=134217728,"TAESD encoder feature map exceeds its allocation limit")
    val values=image.toFloatArray
    require(values.forall(_.isFinite),"TAESD image contains nonfinite values")
    cancelled()
    Tensor.fromFloats(values.map(v=>(v+1f)/2f),image.shape)

  def encode(image: Tensor[Float],cancelled: () => Unit = () => ()): Tensor[Float] =
    var hidden=input(Variable(prepare(image,cancelled),false))
    for i<-blocks.indices do
      if i>0 then
        cancelled()
        hidden=downsample(i-1)(hidden)
      for block<-blocks(i) do
        cancelled()
        hidden=block(hidden)
    cancelled()
    output(hidden).data

  def encodeCuda(image: Tensor[Float],deviceIndex: Int=0,maxBytes: Long=1073741824L,
      cancelled: () => Unit = () => ()): Tensor[Float] =
    val prepared=prepare(image,cancelled)
    CudaSession.withDevice(deviceIndex,maxBytes) { session =>
      Using.Manager { use =>
        val runtime=use(CudaDiffusionRuntime.upload(session,this,()=>{cancelled();false},includeAdapters=false))
        var hidden=use(session.upload(prepared))
        def next(value: CudaTensor): Unit =
          val owned=use(value)
          hidden.close()
          hidden=owned
          cancelled()
        next(runtime.conv(hidden,input))
        for i<-blocks.indices do
          if i>0 then next(runtime.conv(hidden,downsample(i-1)))
          for block<-blocks(i) do
            cancelled()
            val residual=use(session.reshape(hidden,hidden.shape))
            next(runtime.conv(hidden,block.conv1));next(session.leakyRelu(hidden,0f))
            next(runtime.conv(hidden,block.conv2));next(session.leakyRelu(hidden,0f))
            next(runtime.conv(hidden,block.conv3))
            next(session.add(hidden,residual));residual.close()
            next(session.leakyRelu(hidden,0f))
        next(runtime.conv(hidden,output))
        val result=hidden.toCpu
        cancelled()
        result
      }.get
    }

object TinyAutoencoderEncoder:
  def load(directory: Path): TinyAutoencoderEncoder =
    Using.resource(SafeTensorSet.open(directory,filename="diffusion_pytorch_model.safetensors")) { archive =>
      val used=mutable.Set.empty[String]
      def parameter(name: String,dims: Seq[Int]): Parameter =
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing TAESD tensor: $name"))
        require(info.shape==Shape.from(dims.map(_.toLong)),s"TAESD tensor shape mismatch: $name")
        require(used.add(name),s"duplicate TAESD tensor: $name")
        Parameter(archive.readFloat(name),false)
      def conv(layer: String,in: Int,out: Int,bias: Boolean,stride: Int=1): Conv2d =
        Conv2d(parameter(s"encoder.layers.$layer.weight",Seq(out,in,3,3)),
          Option.when(bias)(parameter(s"encoder.layers.$layer.bias",Seq(out))),
          Conv2dConfig(stride=(stride,stride),padding=(1,1)))
      def block(index: Int): TinyAutoencoderBlock =
        TinyAutoencoderBlock(conv(s"$index.conv.0",64,64,true),conv(s"$index.conv.2",64,64,true),
          conv(s"$index.conv.4",64,64,true))
      val input=conv("0",3,64,true)
      val blocks=Vector(Vector(1),Vector(3,4,5),Vector(7,8,9),Vector(11,12,13)).map(_.map(block))
      val downsample=Vector("2","6","10").map(i=>conv(i,64,64,false,stride=2))
      val output=conv("14",64,4,true)
      require(used.toSet==archive.tensors.keySet.filter(_.startsWith("encoder.")),"unsupported TAESD encoder tensor layout")
      new TinyAutoencoderEncoder(input,blocks,downsample,output)
    }
