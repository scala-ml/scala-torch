package scalatorch.models

import java.nio.file.Path
import scala.collection.mutable
import scala.util.Using
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.io.SafeTensorSet
import scalatorch.nn.{Conv2d,Module,Parameter}
import scalatorch.ffm.{CudaSession,CudaTensor}

/** Frozen TAESD decoder using the standard Diffusers layer layout. */
final case class TinyAutoencoderDecoder private (
    input: Conv2d,blocks: Vector[Vector[TinyAutoencoderBlock]],
    intermediate: Vector[Conv2d],output: Conv2d) derives Module:
  private def prepare(latent: Tensor[Float],cancelled: () => Unit): Tensor[Float] =
    val dims=latent.shape.dims
    require(latent.device==Device.Cpu && latent.dtype==DType.Float32 && dims.size==4 &&
      dims(0)==1 && dims(1)==4 && dims(2)>0 && dims(3)>0,"TAESD expects one CPU Float32 four-channel latent")
    require(BigInt(dims(2))*dims(3)*64*64<=134217728,"TAESD feature map exceeds its allocation limit")
    val values=latent.toFloatArray
    require(values.forall(_.isFinite),"TAESD latent contains nonfinite values")
    cancelled()
    Tensor.fromFloats(values.map(v=>(math.tanh(v.toDouble/3)*3).toFloat),latent.shape)

  def decode(latent: Tensor[Float],cancelled: () => Unit = () => ()): Tensor[Float] =
    var x=Variable(prepare(latent,cancelled),false)
    x=input(x).relu()
    for i<-blocks.indices do
      for block<-blocks(i) do
        cancelled()
        x=block(x)
      if i<intermediate.size then
        cancelled()
        val shape=x.data.shape.dims
        x=Variable(Resize2d.nearest(x.data,(shape(2)*2).toInt,(shape(3)*2).toInt),false)
        x=intermediate(i)(x)
    cancelled()
    val result=output(x).data
    Tensor.fromFloats(result.toFloatArray.map(value=>value*2f-1f),result.shape)

  def decodeCuda(latent: Tensor[Float],deviceIndex: Int=0,maxBytes: Long=1073741824L,
      cancelled: () => Unit = () => ()): Tensor[Float] =
    val prepared=prepare(latent,cancelled)
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
        next(session.leakyRelu(hidden,0f))
        for i<-blocks.indices do
          for block<-blocks(i) do
            cancelled()
            val residual=use(session.reshape(hidden,hidden.shape))
            next(runtime.conv(hidden,block.conv1));next(session.leakyRelu(hidden,0f))
            next(runtime.conv(hidden,block.conv2));next(session.leakyRelu(hidden,0f))
            next(runtime.conv(hidden,block.conv3))
            next(session.add(hidden,residual));residual.close()
            next(session.leakyRelu(hidden,0f))
          if i<intermediate.size then
            val shape=hidden.shape.dims
            next(session.resizeNearest2d(hidden,(shape(2)*2).toInt,(shape(3)*2).toInt))
            next(runtime.conv(hidden,intermediate(i)))
        next(runtime.conv(hidden,output))
        val result=hidden.toCpu
        cancelled()
        Tensor.fromFloats(result.toFloatArray.map(value=>value*2f-1f),result.shape)
      }.get
    }

object TinyAutoencoderDecoder:
  def load(directory: Path): TinyAutoencoderDecoder =
    Using.resource(SafeTensorSet.open(directory,filename="diffusion_pytorch_model.safetensors")) { archive =>
      val used=mutable.Set.empty[String]
      def parameter(name: String,dims: Seq[Int]): Parameter =
        val info=archive.tensors.getOrElse(name,throw IllegalArgumentException(s"missing TAESD tensor: $name"))
        require(info.shape==Shape.from(dims.map(_.toLong)),s"TAESD tensor shape mismatch: $name")
        require(used.add(name),s"duplicate TAESD tensor: $name")
        Parameter(archive.readFloat(name),false)
      def conv(layer: String,in: Int,out: Int,bias: Boolean): Conv2d =
        Conv2d(parameter(s"decoder.layers.$layer.weight",Seq(out,in,3,3)),
          Option.when(bias)(parameter(s"decoder.layers.$layer.bias",Seq(out))),Conv2dConfig(padding=(1,1)))
      def block(index: Int): TinyAutoencoderBlock =
        TinyAutoencoderBlock(conv(s"$index.conv.0",64,64,true),conv(s"$index.conv.2",64,64,true),
          conv(s"$index.conv.4",64,64,true))
      val input=conv("0",4,64,true)
      val blocks=Vector(Vector(2,3,4),Vector(7,8,9),Vector(12,13,14),Vector(17)).map(_.map(block))
      val intermediate=Vector("6","11","16").map(i=>conv(i,64,64,false))
      val output=conv("18",64,3,true)
      val decoderKeys=archive.tensors.keySet.filter(_.startsWith("decoder."))
      require(used.toSet==decoderKeys,"unsupported TAESD decoder tensor layout")
      new TinyAutoencoderDecoder(input,blocks,intermediate,output)
    }
