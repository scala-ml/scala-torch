package scalatorch.models.worker

import java.nio.file.{Files,Path}
import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.io.{PyTorchArchiveLimits,PyTorchTensorArchive}
import scalatorch.models.PyTorchCompatibility
import scalatorch.nn.{Conv2d,Parameter}

/** Eight-layer A1111 VAE approximation, loaded through the restricted tensor archive reader. */
private[worker] final class NativeApproxNn private (layers: Vector[Conv2d]):
  def decodeTensor(latent: Tensor[Float],cancelled: () => Unit): Tensor[Float] =
    val dims=latent.shape.dims
    require(latent.device==Device.Cpu && dims.size==4 && dims(0)==1 && dims(1)==4 && dims(2)>0 && dims(3)>0 &&
      (dims(2)*2+22)*(dims(3)*2+22)*64<=134217728L,"invalid Approx NN latent dimensions")
    require(latent.toFloatArray.forall(_.isFinite),"nonfinite latent")
    cancelled()
    var value=Variable(latent,false).resizeNearest2d((dims(2)*2).toInt,(dims(3)*2).toInt).pad2d(Padding2d(11,11,11,11))
    layers.foreach { layer =>
      cancelled()
      val result=layer(value).data
      value=Variable(Tensor.fromFloats(result.toFloatArray.map(x=>if x>=0 then x else x*.1f),result.shape),false)
    }
    cancelled()
    value.data

private[worker] object NativeApproxNn:
  def load(path: Path): NativeApproxNn =
    require(Files.isRegularFile(path) && Files.size(path)<=16L*1024*1024,"Approx NN weights missing or too large")
    val limits=PyTorchCompatibility.limits(PyTorchArchiveLimits(maxArchiveBytes=16L*1024*1024,
      maxEntries=64,maxEntryBytes=16L*1024*1024,maxTotalUncompressedBytes=16L*1024*1024,maxTensors=16))
    val entries=PyTorchTensorArchive.read(path,limits).value match
      case PyTorchTensorArchive.DictValue(values) => values
      case _ => throw IllegalArgumentException("Approx NN weights must be a tensor dictionary")
    val inputs=Vector(4,8,16,32,64,32,16,8)
    val outputs=Vector(8,16,32,64,32,16,8,3)
    val kernels=Vector(7,5,3,3,3,3,3,3)
    val expected=(1 to 8).flatMap(i=>Vector(s"conv$i.weight",s"conv$i.bias")).toSet
    require(entries.keySet==expected,"unsupported Approx NN checkpoint keys")
    def parameter(name: String,shape: Seq[Int]): Parameter =
      val value=entries(name) match
        case PyTorchTensorArchive.TensorValue(tensor) => tensor
        case _ => throw IllegalArgumentException(s"invalid Approx NN tensor: $name")
      require(value.shape==shape.map(_.toLong).toVector && value.values.forall(_.isFinite),s"invalid Approx NN shape or values: $name")
      Parameter(Tensor.fromFloats(value.values,Shape.from(shape.map(_.toLong))),false)
    val layers=(0 until 8).map { i =>
      val k=kernels(i)
      Conv2d(parameter(s"conv${i+1}.weight",Seq(outputs(i),inputs(i),k,k)),
        Some(parameter(s"conv${i+1}.bias",Seq(outputs(i)))),Conv2dConfig())
    }.toVector
    new NativeApproxNn(layers)

private[worker] final class NativeApproxNnVaeTools(encoder: NativeVaeToolsCodec,model: NativeApproxNn)
    extends NativeVaeToolsCodec:
  def encode(image: java.awt.image.BufferedImage,cancelled: () => Unit): Tensor[Float] = encoder.encode(image,cancelled)
  override def encodeSeeded(image: java.awt.image.BufferedImage,seed: Long,cancelled: () => Unit): Tensor[Float] =
    encoder.encodeSeeded(image,seed,cancelled)
  def decode(latent: Tensor[Float],cancelled: () => Unit): java.awt.image.BufferedImage =
    NativePreview.decodedRgb(model.decodeTensor(latent,cancelled))
