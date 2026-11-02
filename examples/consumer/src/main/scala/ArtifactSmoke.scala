import scalatorch.*
import scalatorch.autograd.Variable
import scalatorch.models.{PretrainedClip,PretrainedUNet}
import scalatorch.nn.Module
import java.nio.file.Path

/** Compiles only against Maven artifacts, never a dependsOn link to the source build. */
object ArtifactSmoke:
  def main(args: Array[String]): Unit =
    val x=Variable(Tensor.fromFloats(Array(1f,2f,3f),Shape.from(Seq(3))),true)
    (x*x).sum().backward()
    require(x.grad.get.toFloatArray.toVector==Vector(2f,4f,6f),"artifact autograd mismatch")
    println("Artifact CPU/autograd smoke passed")
    require(args.isEmpty || args.length==2,"provide both CLIP and U-Net checkpoint directories, or neither")
    if args.length==2 then
      val clip=PretrainedClip.load(Path.of(args(0)))
      val unet=PretrainedUNet.load(Path.of(args(1)))
      require(Module.parameters(clip).size==84 && Module.parameters(unet).size==304,"expected pinned tiny-sd-pipe weights")
      val ids=Tensor.fromLongs(Array.tabulate(77)(i => ((i*17+3)%997).toLong),Shape.from(Seq(1,77)))
      val image=Variable(Tensor.fromFloats(Array.tabulate(60)(i => math.sin(i*.17).toFloat),Shape.from(Seq(1,4,3,5))),false)
      val time=Tensor.fromFloats(Array(7.5f),Shape.from(Seq(1)))
      val result=unet(image,time,clip(ids))
      require(result.data.shape==image.data.shape && !result.requiresGrad && result.data.toFloatArray.forall(_.isFinite))
      println("Artifact public CLIP-to-U-Net smoke passed")
